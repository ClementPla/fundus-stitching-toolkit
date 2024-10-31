import warnings

import cv2
import numpy as np
from fundus_data_toolkit.utils.image_processing import fundus_precise_autocrop
from plotly import graph_objects as go
from scipy.ndimage import distance_transform_edt
from skimage import transform
from skimage.feature import match_descriptors
from skimage.measure import ransac
from skimage.morphology import isotropic_erosion
from skimage.registration import optical_flow_ilk

from fundus_stitching_toolkit.io import save_img
from fundus_stitching_toolkit.registration.detector import _KeypointsDetector
from fundus_stitching_toolkit.registration.helper import choose_reference_image
from fundus_stitching_toolkit.segment import segment_od_mask, segment_vessels
from fundus_stitching_toolkit.utils.config import (
    Descriptors,
    Keypoints,
    ReferenceChoice,
)
from fundus_stitching_toolkit.utils.preprocess import resize_and_pad
from fundus_stitching_toolkit.utils.visu import imshow


class FundusRegistration:
    def __init__(
        self,
        images,
        reference_choice: ReferenceChoice,
        descriptor_type: Descriptors = Descriptors.TOPO_BASED,
        keypoints_type: Keypoints = Keypoints.VESSELS_CROSSING,
        transform=transform.SimilarityTransform,
        max_ratio: float = 0.8,
        max_dimensions=None,
        alpha_stitch: float = 1,
        with_elastic_registration: bool = False,
        margin: int = 0,
        **kwargs,
    ):
        self._referenceChoice = reference_choice
        self.descriptor_type = descriptor_type
        self.keypoints_type = keypoints_type
        self.max_ratio = max_ratio

        images_rois = [fundus_precise_autocrop(image=image) for image in images]

        if self.descriptor_type in [Descriptors.DINOV2, Descriptors.CNN]:
            warnings.warn(
                "With DINO or CNN descriptors, images are resized/padded to be at resolution 1024x1024"
            )
            for i, (data) in enumerate(images_rois):
                images_rois[i] = dict(
                    roi=resize_and_pad(data["roi"], (1024, 1024)),
                    image=resize_and_pad(data["image"], (1024, 1024)),
                )

        self.images = [f["image"] for f in images_rois]
        self.rois = [f["roi"] for f in images_rois]
        self.masks_od, self.masks_macula = segment_od_mask(self.images)
        self.masks_vessels = segment_vessels(self.images)
        self.margin = margin
        self.reference_index = choose_reference_image(
            self.masks_od, self.masks_macula, self.masks_vessels, reference_choice
        )

        self.transform = transform
        self.keypoints = None
        self.descriptors = None
        self.matched_keypoints = None
        self.warped_rois = {}
        self.warped_images = {}
        self.trfm_list = []
        self._ignore_index = []
        if max_dimensions is None:
            max_dimensions = [
                int(_ * len(self.images) * 0.8) for _ in self.images[0].shape[:2]
            ]
        self.max_dimensions = max_dimensions
        self.alpha_stitch = alpha_stitch
        self.detector = _KeypointsDetector(self, **kwargs)
        self.inliers_kpts = {}
        self.inliers_matches = {}
        self.with_elastic_registration = with_elastic_registration

    def register(self):
        self.extract_keypoints()
        self.compute_descriptors()
        if self.matched_keypoints is None:
            self.match_all_descriptors()
        self.registration()

        # We don't call this function for now as it doesn't seem to work
        if self.with_elastic_registration:
            self.elastic_registration()
        return self.stitch()

    def extract_keypoints(self):
        if self.keypoints is None:
            self.detector.extract_keypoints()

    def compute_descriptors(self, force=False):
        if force:
            self.descriptors = None
        if self.descriptors is None or (
            self.descriptor_type != Descriptors.SIFT
            and self.keypoints_type == Keypoints.SIFT
        ):
            self.detector.compute_descriptors()

    def match_all_descriptors(self):
        self.matched_keypoints = {}
        ref_descriptors = self.descriptors[self.reference_index]
        for i, descriptor in self.descriptors.items():
            self.matched_keypoints[i] = match_descriptors(
                ref_descriptors, descriptor, max_ratio=self.max_ratio, cross_check=True
            )

    def stitch(self):
        """
        Stitch the warped images using alpha blending
        """

        # Determine the output shape from the warped images
        rois = np.asarray(
            [roi for roi in self.warped_rois.values() if np.sum(roi) > 1e3]
        ).astype(bool)
        eroded_rois = np.asarray([isotropic_erosion(roi, 32) for roi in rois]).astype(
            bool
        )

        distances = np.asarray([distance_transform_edt(roi) for roi in eroded_rois])
        distances -= distances.min((1, 2), keepdims=True)
        distances /= distances.max((1, 2), keepdims=True)  # Distance between 0 and 1

        # distances = sigmoid(distances, alpha=self.alpha_stitch)
        distances = np.exp(self.alpha_stitch * distances)
        distances = distances / distances.sum(0)

        result = np.asarray(
            [
                img
                for i, img in self.warped_images.items()
                if np.sum(self.warped_rois[i]) > 1e3
            ]
        )
        distances[~eroded_rois] = np.nan

        result = distances[:, :, :, None] * result
        eps = 1e-8
        result = np.nansum(result, 0) / (np.nansum(distances, 0)[:, :, None] + eps)
        result[np.isnan(result)] = 0
        # result -= np.nanmin(result)
        # result /= np.nanmax(result)

        result = np.clip(result, 0, 1)

        return result

    def registration(self):
        self.trfm_list = {}
        self.warped_images = {}
        self.warped_rois = {}
        self.inliers_kpts = {}
        self.inliers_matches = {}
        ref_keypoints = self.keypoints[self.reference_index]
        minx, miny = np.inf, np.inf
        maxx, maxy = -np.inf, -np.inf

        # First pass: calculate all transformations and find global bounds
        for i, img in enumerate(self.images):
            # Get matched keypoints
            dst = ref_keypoints[self.matched_keypoints[i][:, 0]]
            src = self.keypoints[i][self.matched_keypoints[i][:, 1]]

            # Convert keypoints from (y, x) to (x, y)
            src_xy = src[:, ::-1]
            dst_xy = dst[:, ::-1]

            # We only keep they keypoints that are in the same region

            transform_model, inliers = ransac(
                (dst_xy, src_xy),
                self.transform,
                min_samples=min(3, src_xy.shape[0]),
                residual_threshold=5,
                max_trials=1000,
            )
            # H, mask = cv2.findHomography(dst_xy, src_xy, cv2.RANSAC, 5.0)
            # transform_model = transform.ProjectiveTransform(matrix=H)

            if transform_model is None:
                continue

            inverse_model = transform_model.inverse

            self.trfm_list[i] = transform_model

            # Map inliers to the reference image

            self.inliers_matches[i] = self.matched_keypoints[i]

            # Calculate transformed corners
            h, w = img.shape[:2]
            corners = np.array([[0, 0], [0, w], [h, 0], [h, w]])
            transformed_corners = inverse_model(corners)
            minx = min(minx, transformed_corners[:, 0].min())
            miny = min(miny, transformed_corners[:, 1].min())
            maxx = max(maxx, transformed_corners[:, 0].max())
            maxy = max(maxy, transformed_corners[:, 1].max())

        minx = max(minx, -self.max_dimensions[1])
        maxx = min(maxx, self.max_dimensions[1])
        miny = max(miny, -self.max_dimensions[0])
        maxy = min(maxy, self.max_dimensions[0])

        width = int(-minx + maxx)
        height = int(-miny + maxy)
        height = min(height, self.max_dimensions[0])
        width = min(width, self.max_dimensions[1])
        output_shape = (height + 2 * self.margin, width + 2 * self.margin)
        global_transform = transform.SimilarityTransform(translation=[-minx, -miny])
        # Second pass: apply transformations
        for i, img in enumerate(self.images):
            if i not in self.trfm_list:
                continue

            t = global_transform.inverse + self.trfm_list[i]
            # Warp image
            img = img / 255.0
            img = transform.warp(
                img, t, output_shape=output_shape, mode="constant", cval=0
            )
            img = np.nan_to_num(img)
            if img.max() < 0.05:
                del self.trfm_list[i]
                # Some images are pitch black after transformation. No need to keep them
                continue

            self.warped_images[i] = img
            roi = self.rois[i]
            roi = transform.warp(
                roi,
                t,
                output_shape=output_shape,
                mode="constant",
                cval=0,
            )

            self.warped_rois[i] = roi

            self.inliers_kpts[i] = t.inverse(self.keypoints[i][:, ::-1])[:, ::-1]

    def elastic_registration(self):
        ref_keypoints = self.inliers_kpts[self.reference_index]
        for i, img in self.warped_images.items():
            if i == self.reference_index:
                continue
            dst = ref_keypoints[self.inliers_matches[i][:, 0]]
            src = self.inliers_kpts[i][self.inliers_matches[i][:, 1]]

            # Convert keypoints from (y, x) to (x, y)
            src_xy = src[:, ::-1]
            dst_xy = dst[:, ::-1]

            # We only keep they keypoints that are in the same region
            distance = np.linalg.norm(src_xy - dst_xy, axis=1)
            src_xy_fitted = src_xy[distance < 15]
            dst_xy_fitted = dst_xy[distance < 15]
            transform_model = transform.PolynomialTransform(dimensionality=2)

            success = transform_model.estimate(dst_xy_fitted, src_xy_fitted, order=2)

            # Estimate is the transformation is too large:
            after = transform_model(src_xy)
            distance = np.linalg.norm(after - src_xy, axis=1)
            success = success and np.mean(distance) < 25

            if not success:
                continue
            self.warped_images[i] = transform.warp(
                img,
                transform_model,
                output_shape=self.warped_images[self.reference_index].shape,
                mode="constant",
                cval=0,
            )
            self.warped_rois[i] = transform.warp(
                self.warped_rois[i],
                transform_model,
                output_shape=self.warped_rois[self.reference_index].shape,
                mode="constant",
                cval=0,
            )

    def plot(self, index=None, with_masks=True, with_keypoints=True):
        """
        Plot the images with the keypoints.
        """
        if index is None:
            index = list(range(len(self.images)))

        if not isinstance(index, list):
            index = [index]

        for i in index:
            img = self.images[i]
            mask_od = self.masks_od[i]
            mask_mac = self.masks_macula[i]
            mask_vessels = self.masks_vessels[i]
            keypoints = self.keypoints[i]
            fig = imshow(
                img,
                mask=[
                    mask_od,
                    mask_mac,
                    mask_vessels,
                ]
                if with_masks
                else None,
                keypoints=keypoints if with_keypoints else None,
                title="Keypoints",
                show=False,
            )
            if i == self.reference_index:
                fig.add_trace(
                    go.Scatter(
                        x=[0, img.shape[1], img.shape[1], 0, 0],
                        y=[0, 0, img.shape[0], img.shape[0], 0],
                        mode="lines",
                        opacity=0.8,
                        showlegend=False,
                        line=dict(color="blue", width=10),
                    )
                )
            fig.show(config={"scrollZoom": True, "displaylogo": False})

    def plot_warped(self, index=None):
        """
        Plot the images with the keypoints.
        """
        if index is None:
            index = list(range(len(self.images)))

        if not isinstance(index, list):
            index = [index]

        for i, img in self.warped_images.items():
            fig = imshow(
                img,
                show=False,
            )

            if i == self.reference_index:
                fig.add_trace(
                    go.Scatter(
                        x=[0, img.shape[1], img.shape[1], 0, 0],
                        y=[0, 0, img.shape[0], img.shape[0], 0],
                        mode="lines",
                        opacity=0.8,
                        showlegend=False,
                        line=dict(color="blue", width=10),
                    )
                )
            fig.show(config={"scrollZoom": True, "displaylogo": False})

    def plot_correspondances(self):
        self.extract_keypoints()
        self.compute_descriptors()
        if not self.matched_keypoints:
            self.match_all_descriptors()
        ref_img = self.images[self.reference_index]
        h, w, c = ref_img.shape
        keypoints_ref = self.keypoints[self.reference_index]

        ref_od = self.masks_od[self.reference_index]
        ref_mac = self.masks_macula[self.reference_index]
        ref_vessels = self.masks_vessels[self.reference_index]
        for i, (img, mask_od, mask_mac, mask_vessels) in enumerate(
            zip(self.images, self.masks_od, self.masks_macula, self.masks_vessels)
        ):
            if i == self.reference_index:
                continue
            img = self.images[i]
            img = cv2.resize(img, (w, h))
            mask_od = cv2.resize(mask_od.astype(np.uint8), (w, h))
            mask_mac = cv2.resize(mask_mac.astype(np.uint8), (w, h))
            mask_vessels = cv2.resize(mask_vessels.astype(np.uint8), (w, h))

            sideBysideImg = np.concatenate(([ref_img, img]), axis=1)
            sideBysideMaskOD = np.concatenate(([ref_od, mask_od]), axis=1)
            sideBysideMaskMac = np.concatenate(([ref_mac, mask_mac]), axis=1)
            sideBysideMaskVessels = np.concatenate(
                ([ref_vessels, mask_vessels]), axis=1
            )

            keypoints = self.keypoints[i].copy()

            keypoints[:, 1] += w
            fig = imshow(
                sideBysideImg,
                mask=[
                    sideBysideMaskOD,
                    sideBysideMaskMac,
                    sideBysideMaskVessels,
                ],
                keypoints=np.concatenate(
                    [
                        keypoints_ref[self.matched_keypoints[i][:, 0]],
                        keypoints[self.matched_keypoints[i][:, 1]],
                    ],
                    0,
                ),
                title=f"Correspondances {self.reference_index} - {i}: {len(self.matched_keypoints[i])}",
                show=False,
            )
            start_line = keypoints_ref[self.matched_keypoints[i][:, 0]]
            end_line = keypoints[self.matched_keypoints[i][:, 1]]
            for start, end in zip(start_line, end_line):
                fig.add_trace(
                    go.Scatter(
                        x=[start[1], end[1]],
                        y=[start[0], end[0]],
                        mode="lines",
                        legend=None,
                        opacity=0.8,
                        showlegend=False,
                    )
                )
            fig.show(config={"scrollZoom": True, "displaylogo": False})

    def plot_result(self):
        result = self.register()
        imshow(result, title="Result")

    def save_result(self, path, with_metadata=False):
        result = self.register()
        result = (result * 255).astype(np.uint8)
        if with_metadata:
            metadata = [
                self._referenceChoice.name,
                self.descriptor_type.name,
                self.keypoints_type.name,
                f"alpha_{self.alpha_stitch}",
            ]
            metadata = "_".join(metadata)
            save_img(result, path, suffix=f"_{metadata}")
        else:
            save_img(result, path)
