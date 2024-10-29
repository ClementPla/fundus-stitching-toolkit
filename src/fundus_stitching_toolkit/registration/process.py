import cv2
import numpy as np
from fundus_data_toolkit.utils.image_processing import fundus_precise_autocrop
from plotly import graph_objects as go
from scipy.ndimage import distance_transform_edt
from skimage import transform
from skimage.feature import match_descriptors
from skimage.measure import ransac
from skimage.morphology import isotropic_dilation, isotropic_erosion, skeletonize
from skimage.registration import optical_flow_ilk

from fundus_stitching_toolkit.registration.helper import choose_reference_image
from fundus_stitching_toolkit.segment import segment_od_mask, segment_vessels
from fundus_stitching_toolkit.utils.config import Descriptors, ReferenceChoice
from fundus_stitching_toolkit.utils.visu import imshow


class FundusRegistration:
    def __init__(
        self,
        images,
        reference_choice: ReferenceChoice,
        descriptor_type: Descriptors = Descriptors.TOPO_BASED,
        transform=transform.SimilarityTransform,
        max_ratio: float = 0.8,
    ):
        images_rois = [fundus_precise_autocrop(image=image) for image in images]

        self.images = [f["image"] for f in images_rois]
        self.rois = [f["roi"] for f in images_rois]
        self.masks_od, self.masks_macula = segment_od_mask(self.images)
        self.masks_vessels = segment_vessels(self.images)

        self.reference_index = choose_reference_image(
            self.masks_od, self.masks_macula, self.masks_vessels, reference_choice
        )
        self.max_ratio = max_ratio
        self.descriptor_type = descriptor_type
        self.transform = transform
        self.keypoints = None
        self.descriptors = None
        self.matched_keypoints = {}
        self.warped_rois = {}
        self.warped_images = {}
        self.trfm_list = []
        self._ignore_index = []

    def OD_centers(self):
        return [np.mean(np.argwhere(mask), axis=0) for mask in self.masks_od]

    def register(self):
        self.extract_keypoints_from_vessels()
        self.compute_descriptors()
        self.match_all_descriptors()
        self.registration()

        # We don't call this function for now as it doesn't seem to work
        # self.elastic_registration()
        return self.stitch()

    def extract_keypoints_from_vessels(self, neighborhood_size=3):
        """
        Compute the skeleton of the vessels masks and extract keypoints, which are defined as the bifurcation points.
        """

        vessels_skeletons = [
            skeletonize(mask).astype(np.uint8) for mask, odmask in zip(self.masks_vessels, self.masks_od)
        ]

        self.keypoints = []
        kernel = np.ones((neighborhood_size, neighborhood_size), np.uint8)
        for skeleton in vessels_skeletons:
            # Compute the distance transform

            neighbors = cv2.filter2D(skeleton, -1, kernel)
            local_maxima = (skeleton > 0) & (neighbors > 3)

            self.keypoints.append(np.argwhere(local_maxima))

    def compute_descriptors(self):
        match self.descriptor_type:
            case Descriptors.ORB:
                self.compute_ORB_descriptors()

            case Descriptors.TOPO_BASED:
                self.compute_topological_descriptors()

            case _:
                raise ValueError("Invalid descriptor type")

    def compute_ORB_descriptors(self):
        assert self.keypoints is not None, "You need to extract keypoints first"

        orb = cv2.ORB_create()
        self.descriptors = {}
        for i, (img, keypoints) in enumerate(zip(self.images, self.keypoints)):
            kp = [cv2.KeyPoint(x=point[1], y=point[0], size=20) for point in keypoints.astype(np.float32)]
            _, des = orb.compute(img, kp)
            self.descriptors[i] = des

    def compute_topological_descriptors(self):
        self.descriptors = {}
        od_centers = self.OD_centers()
        for i, keypoints in enumerate(self.keypoints):
            od_center = od_centers[i]
            self.descriptors[i] = keypoints - od_center

    def match_all_descriptors(self):
        self.matched_keypoints = {}
        ref_descriptors = self.descriptors[self.reference_index]
        for i, descriptor in self.descriptors.items():
            self.matched_keypoints[i] = match_descriptors(ref_descriptors, descriptor, max_ratio=self.max_ratio)

    def stitch(self, alpha=1):
        """
        Stitch the warped images using alpha blending
        """

        # Determine the output shape from the warped images
        rois = np.asarray([roi for roi in self.warped_rois.values()]).astype(bool)
        eroded_rois = np.asarray([isotropic_erosion(roi, 16) for roi in rois]).astype(bool)
        rois = eroded_rois
        distances = np.asarray([distance_transform_edt(roi) for roi in rois])
        distances -= distances.min((1, 2), keepdims=True)
        distances /= distances.max((1, 2), keepdims=True)  # Distance between 0 and 1
        distances = distances * 2 - 1  # Distance between -1 and 1

        def sigmoid(x, alpha=1):
            return 1 / (1 + np.exp(-alpha * x))

        distances = sigmoid(distances, alpha=alpha)

        # Probas of the rois
        # For each image, where the ROIs of the other images is 0, the proba is necessarily 1
        softmax_distances = np.clip(distances, 0, 1)
        softmax_distances[~rois] = np.nan

        result = np.asarray(list(self.warped_images.values()))

        result = softmax_distances[:, :, :, None] * result
        result = np.nanmean(result, 0) / np.nanmean(softmax_distances, 0)[:, :, None]

        # result[np.isnan(result)] = 0
        # result -= np.nanmin(result)
        # result /= np.nanmax(result)

        result = np.clip(result, 0, 1)

        return result

    def registration(self):
        self.trfm_list = {}
        self.warped_images = {}
        ref_keypoints = self.keypoints[self.reference_index]

        minx, miny = 0, 0
        maxx, maxy = self.images[self.reference_index].shape[1], self.images[self.reference_index].shape[0]
        h, w, c = self.images[self.reference_index].shape
        for i, img in enumerate(self.images):
            # Get matched keypoints
            dst = ref_keypoints[self.matched_keypoints[i][:, 0]]
            src = self.keypoints[i][self.matched_keypoints[i][:, 1]]

            # Convert keypoints from (y, x) to (x, y)
            src_xy = src[:, ::-1]  # Flip the columns
            dst_xy = dst[:, ::-1]  # Flip the columns

            # Estimate transform with (x, y) coordinates
            transform_model, inliers = ransac(
                (dst_xy, src_xy),
                self.transform,
                min_samples=min(3, src_xy.shape[0]),
                residual_threshold=10,
                max_trials=1000,
            )

            # Store transform
            if transform_model is None:
                continue
            self.trfm_list[i] = transform_model

            # Calculate transformed corners
            y, x = img.shape[:2]
            corners = np.array([[0, 0], [0, y], [x, 0], [x, y]])
            transformed_corners = transform_model(corners)
            # Update min and max coordinates
            minx = min(minx, transformed_corners[:, 0].min())
            miny = min(miny, transformed_corners[:, 1].min())
            maxx = max(maxx, transformed_corners[:, 0].max())
            maxy = max(maxy, transformed_corners[:, 1].max())

        # Calculate global transformation to shift all images to positive coordinate space
        glob_trns = np.eye(3)
        glob_trns[:2, 2] = -minx, -miny
        out_shape = (int(np.abs(maxy) + np.abs(miny)), int(np.abs(maxy) + np.abs(minx)))  # TODO: Fix this
        global_transform = self.transform(matrix=glob_trns)

        # Apply global transformation and warp images
        for i, img in enumerate(self.images):
            if i not in self.trfm_list:
                continue
            full_transform = self.trfm_list[i]
            img = img / 255.0
            img = transform.warp(
                img,
                (global_transform.inverse + full_transform),
                output_shape=out_shape,
                mode="constant",
                cval=0,
            )
            self.warped_rois[i] = transform.warp(
                self.rois[i],
                (global_transform.inverse + full_transform),
                output_shape=out_shape,
                mode="constant",
                cval=0,
            )
            # img = transform.warp(img, global_transform.inverse, mode="constant", cval=0)

            self.warped_images[i] = img
            img = transform.warp(img, global_transform, mode="constant", cval=0)

    def elastic_registration(self):
        ref_img = self.warped_images[self.reference_index]

        nr, nc, _ = ref_img.shape

        row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing="ij")

        for i, img in enumerate(self.images):
            print(f"Computing optical flow for image {i}")
            if i == self.reference_index:
                continue
            v, u = optical_flow_ilk(ref_img.mean(-1), self.warped_images[i].mean(-1))

            for j in range(3):
                self.warped_images[i][:, :, j] = transform.warp(
                    self.warped_images[i][:, :, j], np.array([row_coords + v, col_coords + u]), mode="edge"
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

        for i in index:
            img = self.warped_images[i]
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
        if self.keypoints is None:
            self.extract_keypoints_from_vessels()
        if self.descriptors is None:
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
            sideBysideMaskVessels = np.concatenate(([ref_vessels, mask_vessels]), axis=1)

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
                    [keypoints_ref[self.matched_keypoints[i][:, 0]], keypoints[self.matched_keypoints[i][:, 1]]], 0
                ),
                title="Correspondances",
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

    def plot_result(self, debug=False):
        result = self.register(debug=debug)
        imshow(result, title="Result")
