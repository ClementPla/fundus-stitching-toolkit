import warnings

import albumentations as A
import cv2
import numpy as np
import timm
import torch
from skimage.feature import SIFT
from skimage.morphology import isotropic_erosion, skeletonize

from fundus_stitching_toolkit.utils.config import Descriptors, Keypoints
from fundus_stitching_toolkit.utils.morpho import getSkeletonIntersection


class _KeypointsDetector:
    def __init__(self, processor, K=512, window_size=3, resize=(1024, 1024)):
        self.p = processor
        self._features = []
        self.K = K
        self.window_size = window_size // 2
        if (
            self.p.descriptor_type == Descriptors.DINOV2
            or self.p.descriptor_type == Descriptors.CNN
        ):
            if self.p.descriptor_type == Descriptors.DINOV2:
                model = timm.create_model(
                    "hf_hub:ClementP/FundusDRGrading-vit_base_patch14_reg4_dinov2.lvd142m",
                    # "vit_base_patch14_reg4_dinov2.lvd142m",
                    pretrained=True,
                    global_pool="",
                    img_size=(1024, 1024),
                ).cuda()
            if self.p.descriptor_type == Descriptors.CNN:
                model = timm.create_model(
                    "hf_hub:ClementP/FundusDRGrading-convnext_base",
                    pretrained=True,
                    num_classes=1,
                ).cuda()

            for i, img in enumerate(self.p.images):
                model.eval()
                img = A.Normalize()(image=img)["image"]
                img = np.transpose(img, (2, 0, 1))
                img = np.expand_dims(img, axis=0)
                img = torch.from_numpy(img).float().cuda()
                # Normalize the image

                with torch.no_grad():
                    if self.p.descriptor_type == Descriptors.DINOV2:
                        output = model.get_intermediate_layers(img, 1, reshape=True)
                        fmap = output[-1]

                    if self.p.descriptor_type == Descriptors.CNN:
                        fmap = model.forward_features(img)

                    fmap = torch.nn.functional.interpolate(
                        fmap,
                        size=resize,
                        mode="bilinear",
                        align_corners=True,
                    )
                    self._features.append(fmap.cpu().numpy().squeeze(0))

    def extract_keypoints(self):
        match self.p.keypoints_type:
            case Keypoints.VESSELS_CROSSING:
                self.extract_keypoints_from_vessels()
            case Keypoints.SIFT:
                self.extract_SIFT_keypoints()
            case Keypoints.ORB:
                self.extract_ORB_keypoints()
            case Keypoints.MODEL_KLARGEST:
                self.extract_model_klargest_keypoints()
            case Keypoints.SURF:
                raise NotImplementedError(
                    "SURF keypoints are not implemented yet due to patent issues"
                )

    def compute_descriptors(self):
        match self.p.descriptor_type:
            case Descriptors.ORB:
                self.compute_ORB_descriptors()

            case Descriptors.TOPO_BASED:
                self.compute_topological_descriptors()

            case Descriptors.SIFT:
                warnings.warn(
                    "SIFT descriptors are calculated automatically when extracting keypoints"
                )
                return

            case Descriptors.FREAK:
                self.compute_FREAK_descriptors()
            case Descriptors.DINOV2 | Descriptors.CNN:
                self.compute_CNN_descriptors()
            case _:
                raise ValueError("Invalid descriptor type")

    def extract_model_klargest_keypoints(self):
        self.p.keypoints = []
        for i, feature_map in enumerate(self._features):
            h, w = self.p.images[i].shape[:2]
            fnorm = np.linalg.norm(feature_map, axis=0)
            fnorm = cv2.resize(fnorm, (w, h), interpolation=cv2.INTER_CUBIC)
            fnorm *= self.p.masks_vessels[i]
            k = self.K
            # Get the k largest values
            idx = np.argpartition(fnorm.flatten(), -k)[-k:]
            y, x = np.unravel_index(idx, (h, w))
            self.p.keypoints.append(np.stack((y, x), axis=1))

    def extract_SIFT_keypoints(self):
        self.p.keypoints = []
        self.p.descriptors = {}
        sift = cv2.SIFT_create(contrastThreshold=0.02)
        for i, img in enumerate(self.p.images):
            kp, desc = sift.detectAndCompute(
                img, mask=isotropic_erosion(self.p.rois[i], radius=25).astype(np.uint8)
            )
            kp = np.asarray([(point.pt[1], point.pt[0]) for point in kp])
            self.p.keypoints.append(kp)
            self.p.descriptors[i] = np.array(desc)

    def extract_ORB_keypoints(self):
        self.p.keypoints = []
        self.p.descriptors = {}
        orb = cv2.ORB_create(fastThreshold=0, edgeThreshold=11)
        for i, img in enumerate(self.p.images):
            kp, desc = orb.detectAndCompute(
                img, mask=isotropic_erosion(self.p.rois[i], radius=25).astype(np.uint8)
            )

            kp = np.asarray([(point.pt[1], point.pt[0]) for point in kp])
            self.p.keypoints.append(kp)

            self.p.descriptors[i] = np.array(desc)

    def extract_keypoints_from_vessels(self, neighborhood_size=3):
        """
        Compute the skeleton of the vessels masks and extract keypoints, which are defined as the bifurcation points.
        """

        vessels_skeletons = [
            (skeletonize(mask) > 0).astype(np.uint8)
            for mask, odmask in zip(self.p.masks_vessels, self.p.masks_od)
        ]

        self.p.keypoints = []

        for i, skeleton in enumerate(vessels_skeletons):
            # Compute the distance transform

            kps = getSkeletonIntersection(skeleton)
            self.p.keypoints.append(np.asarray(kps))

    def compute_ORB_descriptors(self):
        assert self.p.keypoints is not None, "You need to extract keypoints first"

        orb = cv2.ORB_create()
        self.p.descriptors = {}
        for i, (img, keypoints) in enumerate(zip(self.p.images, self.p.keypoints)):
            kp = [
                cv2.KeyPoint(x=point[1], y=point[0], size=20)
                for point in keypoints.astype(np.float32)
            ]
            _, des = orb.compute(img, kp)
            self.p.descriptors[i] = des

    def compute_topological_descriptors(self):
        self.p.descriptors = {}
        od_centers = self.OD_centers()
        for i, keypoints in enumerate(self.p.keypoints):
            od_center = od_centers[i]
            self.p.descriptors[i] = keypoints - od_center

    def compute_FREAK_descriptors(self):
        assert self.p.keypoints is not None, "You need to extract keypoints first"

        freak = cv2.xfeatures2d.FREAK_create()
        self.p.descriptors = {}
        for i, (img, keypoints) in enumerate(zip(self.p.images, self.p.keypoints)):
            kp = [
                cv2.KeyPoint(x=point[1], y=point[0], size=20)
                for point in keypoints.astype(np.float32)
            ]
            _, des = freak.compute(img, kp)
            self.p.descriptors[i] = des

    def compute_CNN_descriptors(self):
        assert self.p.keypoints is not None, "You need to extract keypoints first"
        self.p.descriptors = {}
        for i, feature_map in enumerate(self._features):
            hsmall, wsmall = feature_map.shape[1:]

            h, s = self.p.images[i].shape[:2]
            keypoints = self.p.keypoints[i]
            descriptors = []
            for point in keypoints:
                y, x = point
                y = y * hsmall / h
                x = x * wsmall / s
                x, y = int(x), int(y)

                fmap = feature_map[
                    :,
                    y - self.window_size : y + self.window_size,
                    x - self.window_size : x + self.window_size,
                ].max(axis=(1, 2))
                # Normalize the descriptor
                fmap = fmap / np.linalg.norm(fmap)
                descriptors.append(fmap)
            self.p.descriptors[i] = np.array(descriptors)

    def OD_centers(self):
        return [np.mean(np.argwhere(mask), axis=0) for mask in self.p.masks_od]
