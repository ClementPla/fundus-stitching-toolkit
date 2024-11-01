import cv2
import numpy as np


def resize_and_pad(image, resolution):
    """
    Scale the image to the desired resolution by respecting the aspect ratio
    and pad the image to the desired resolution if necessary.

    Parameters:
    image (numpy.ndarray): Input image.
    resolution (tuple): Desired resolution (width, height).

    Returns:
    numpy.ndarray: Resized and padded image.
    """
    target_width, target_height = resolution
    original_height, original_width = image.shape[:2]

    # Calculate aspect ratio
    aspect_ratio = original_width / original_height

    # Determine new dimensions
    if target_width / target_height > aspect_ratio:
        new_height = target_height
        new_width = int(aspect_ratio * target_height)
    else:
        new_width = target_width
        new_height = int(target_width / aspect_ratio)

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height))

    # Create a new image with the target resolution and fill it with black
    if len(image.shape) == 2:
        padded_image = np.zeros((target_height, target_width), dtype=np.uint8)
    else:
        padded_image = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    # Calculate padding offsets
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2

    # Place the resized image in the center of the padded image
    padded_image[y_offset : y_offset + new_height, x_offset : x_offset + new_width] = resized_image

    return padded_image
