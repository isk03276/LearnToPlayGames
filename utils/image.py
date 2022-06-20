import numpy as np
import cv2


def rgb_to_gray(rgb_image:np.ndarray)-> np.ndarray:
    """
    Convert rgb image to gray image
    Args:
        rgb_image (np.ndarray): input image(rgb)

    Returns:
        np.ndarray: gray scaled image
    """
    r, g, b = rgb_image[:,:,0], rgb_image[:,:,1], rgb_image[:,:,2]
    gray_image = 0.2989 * r + 0.5870 * g + 0.1140 * b
    gray_image = np.nan_to_num(gray_image).astype(np.uint8)
    gray_image = gray_image.astype(np.uint8)
    return gray_image

def resize_image(image:np.ndarray, width:int, height:int)-> np.ndarray:
    """
    Resize image
    Args:
        image (np.ndarray): input image
        width (int): width to resize
        height (int): height to resize

    Returns:
        np.ndarray: resized image
    """
    resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    resized_image = np.reshape(resized_image, (width, height, -1))
    return resized_image