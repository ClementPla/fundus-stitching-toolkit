from pathlib import Path

import cv2


def save_img(img, path, suffix=None):
    """Save an image to a path.

    Args:
        img (ndarray): Image to save.
        path (Path): Path to save the image.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if suffix:
        path = str(path.with_suffix("")) + suffix + str(path.suffix)
    path = str(path)
    cv2.imwrite(path, img[:, :, ::-1])
