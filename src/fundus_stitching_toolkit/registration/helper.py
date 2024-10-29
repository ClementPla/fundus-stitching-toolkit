import numpy as np

from fundus_stitching_toolkit.utils.config import ReferenceChoice


def choose_reference_image(masks_OD, masks_macula, masks_vessels, choice: ReferenceChoice) -> int:
    """
    Choose the reference image for registration.
    If method is OD, choose the image with the OD mask closest to the center.
    If method is MACULA, choose the image with the macula mask closest to the center.
    If method is MAX_VESSELS, choose the image with the most vessels.
    return the index of the chosen image.
    """
    h, w = masks_OD[0].shape
    center = (h // 2, w // 2)

    def compute_all_distances_to_center(masks):
        distances = []
        for mask in masks:
            if np.sum(mask) == 0:
                distances.append(np.inf)
                continue
            # Compute the distance between the center and the mask
            center_masks = np.mean(np.argwhere(mask), axis=0)
            distance = np.linalg.norm(center_masks - center)
            distances.append(distance)

        return distances

    match choice:
        case ReferenceChoice.OD:
            distances = compute_all_distances_to_center(masks_OD)
            return np.argmin(distances)

        case ReferenceChoice.MACULA:
            distances = compute_all_distances_to_center(masks_macula)
            return np.argmin(distances)
        case ReferenceChoice.MAX_VESSELS:
            num_vessels = [np.sum(mask) for mask in masks_vessels]
            return np.argmax(num_vessels)
        case _:
            raise ValueError("Invalid method")
