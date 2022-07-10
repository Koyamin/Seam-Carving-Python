import cv2
import numpy as np
import seam_carving


class ImageData:
    """ Image Data class. """
    def __init__(self) -> None:
        self.img = None

    def set_image(self, img_matrix: np.ndarray) -> None:
        """ Set image. """
        self.img = img_matrix

    def height(self) -> int:
        """ Get Height of image. """
        return self.img.shape[0]

    def width(self) -> int:
        """ Get Width of image. """
        return self.img.shape[1]

    def is_set(self) -> bool:
        """ Judge image is set. """
        return self.img is None

    def resize(self, width: int, height: int, keep_mask: np.ndarray = None):
        """ Resize Image. """
        self.img = seam_carving.resize(self.img, width, height, keep_mask=keep_mask)
        self.img = np.ascontiguousarray(self.img)

    def remove_object(self, drop_mask: np.ndarray, keep_mask: np.ndarray = None):
        """ Remove object. """
        original_width, original_height = self.width(), self.height()
        self.img = seam_carving.remove_object(self.img, drop_mask=drop_mask,
                                              keep_mask=keep_mask)
        self.img = np.ascontiguousarray(self.img)
        self.img = seam_carving.resize(self.img, original_width, original_height, keep_mask=None)
        self.img = np.ascontiguousarray(self.img)


