from pathlib import Path
from typing import List

import cv2


def read_imgs(paths: List[Path]):
    """Read images from paths.

    Args:
        paths (list): List of image paths.

    Returns:
        list: List of images.
    """
    imgs = []
    for path in paths:
        path = str(path)
        img = cv2.imread(path)[:, :, ::-1]
        imgs.append(img)
    return imgs


def read_imgs_from_folder(folder_path: Path):
    """Read images from a folder.

    Args:
        folder_path (Path): Path to the folder.

    Returns:
        list: List of images.
    """
    folder_path = Path(folder_path)
    types = ["*.png", "*.jpg", "*.jpeg"]
    paths = []
    for t in types:
        paths.extend(list(folder_path.glob(t)))
    imgs = read_imgs(paths)
    return imgs
