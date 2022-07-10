from typing import Tuple, Optional

import numpy as np
from scipy.ndimage import sobel


DROP_MASK_ENERGY = 1e5
KEEP_MASK_ENERGY = 1e5


def _rgb2gray(rgb: np.ndarray) -> np.ndarray:
    """ Convert an RGB image to a grayscale image. """
    coeffs = np.array([0.2125, 0.7154, 0.0721], dtype=np.float32)
    res = np.matmul(rgb, coeffs).astype(rgb.dtype)
    return res


def _get_energy(gray: np.ndarray) -> np.ndarray:
    """ Get backward energy map from the source image. """
    assert gray.ndim == 2
    gray = gray.astype(np.float32)
    grad_x, grad_y = sobel(gray, axis=1), sobel(gray, axis=0)
    energy = np.abs(grad_x) + np.abs(grad_y)
    return energy


def _get_seam_mask(src: np.ndarray, seam: np.ndarray) -> np.ndarray:
    """ Convert a list of column seam indices to a mask. """
    return ~np.eye(src.shape[1], dtype=np.bool)[seam]


def _remove_seam_mask(src: np.ndarray, seam_mask: np.ndarray) -> np.ndarray:
    """ Remove a seam from the source image according to the given seam mask. """
    if src.ndim == 3:
        h, w, c = src.shape
        seam_mask = np.dstack([seam_mask] * c)
        res = src[seam_mask].reshape((h, w - 1, c))
    else:
        h, w = src.shape
        res = src[seam_mask].reshape((h, w - 1))
    return res


# def _remove_seam(src: np.ndarray, seam: np.ndarray) -> np.ndarray:
#     """ Remove a seam from the source image, given a list of seam columns. """
#     seam_mask = _get_seam_mask(src, seam)
#     res = _remove_seam_mask(src, seam_mask)
#     return res


def _get_seam(energy: np.ndarray) -> np.ndarray:
    """ Get the minimum vertical seam from the energy map. """
    assert energy.size > 0 and energy.ndim == 2
    h, w = energy.shape
    # cost: save each row of total energy map
    cost = energy[0]
    # parent: record parents of seams
    parent = np.empty((h, w), dtype=np.int32)
    # base_idx: [-1,0,1,...,w-2]
    base_idx = np.arange(-1, w - 1, dtype=np.int32)

    # dynamic programming
    for r in range(1, h):
        # left_shift: [cost[1],cost[2],...,cost[w-1],np.inf]
        left_shift = np.hstack((cost[1:], np.inf))
        # right_shift: [np.inf,cost[1],cost[2],...cost[w-2]]
        right_shift = np.hstack((np.inf, cost[:-1]))
        # get the correct parent index of seam
        min_idx = np.argmin([right_shift, cost, left_shift],
                            axis=0) + base_idx
        parent[r] = min_idx
        cost = cost[min_idx] + energy[r]

    # get the seam
    c = np.argmin(cost)
    seam = np.empty(h, dtype=np.int32)
    for r in range(h - 1, -1, -1):
        seam[r] = c
        c = parent[r, c]

    return seam


def _get_seams(gray: np.ndarray, num_seams: int,
                        keep_mask: Optional[np.ndarray]) -> np.ndarray:
    """ Get the minimum N vertical seams. """
    h, w = gray.shape
    # seams_mask: record all the pixels need to be removed
    seams_mask = np.zeros((h, w), dtype=np.bool)
    # rows: [0,1,2,...,h-1]
    rows = np.arange(0, h, dtype=np.int32)
    # idx_map: [[0,1,2,...,w-1] for _ in range(h) ]
    idx_map = np.tile(np.arange(0, w, dtype=np.int32), h).reshape((h, w))
    energy = _get_energy(gray)

    # remove N seams
    for _ in range(num_seams):
        # use keep mask to keep objects
        if keep_mask is not None:
            energy[keep_mask] += KEEP_MASK_ENERGY
        seam = _get_seam(energy)
        # record pixels need to be removed
        seams_mask[rows, idx_map[rows, seam]] = True

        # remove a seam
        seam_mask = _get_seam_mask(gray, seam)
        gray = _remove_seam_mask(gray, seam_mask)
        idx_map = _remove_seam_mask(idx_map, seam_mask)
        if keep_mask is not None:
            keep_mask = _remove_seam_mask(keep_mask, seam_mask)

        # Only need to re-compute the energy in the bounding box of the seam
        _, cur_w = energy.shape
        lo = max(0, np.min(seam) - 1)
        hi = min(cur_w, np.max(seam) + 1)
        pad_lo = 1 if lo > 0 else 0
        pad_hi = 1 if hi < cur_w - 1 else 0
        mid_block = gray[:, lo - pad_lo:hi + pad_hi]
        _, mid_w = mid_block.shape
        mid_energy = _get_energy(mid_block)[:, pad_lo:mid_w - pad_hi]
        energy = np.hstack((energy[:, :lo], mid_energy, energy[:, hi + 1:]))

    return seams_mask


def _reduce_width(src: np.ndarray, delta_width: int,
                  keep_mask: Optional[np.ndarray]) -> np.ndarray:
    """ Reduce width of image by delta width pixels. """
    assert src.ndim in (2, 3) and delta_width >= 0
    if src.ndim == 2:
        gray = src
        src_h, src_w = src.shape
        res_shape = (src_h, src_w - delta_width)
    else:
        gray = _rgb2gray(src)
        src_h, src_w, src_c = src.shape
        res_shape = (src_h, src_w - delta_width, src_c)

    seams_mask = _get_seams(gray, delta_width, keep_mask)
    res = src[~seams_mask].reshape(res_shape)
    return res


def _expand_width(src: np.ndarray, delta_width: int,
                  keep_mask: Optional[np.ndarray]) -> np.ndarray:
    """ Expand width of image by delta width pixels. """
    assert src.ndim in (2, 3) and delta_width >= 0
    if src.ndim == 2:
        gray = src
        src_h, src_w = src.shape
        res_shape = (src_h, src_w + delta_width)
    else:
        gray = _rgb2gray(src)
        src_h, src_w, src_c = src.shape
        res_shape = (src_h, src_w + delta_width, src_c)

    # get delta width seams
    seams_mask = _get_seams(gray, delta_width, keep_mask)
    res = np.empty(res_shape, dtype=np.uint8)

    # insert pixel
    for row in range(src_h):
        res_col = 0
        for src_col in range(src_w):
            if seams_mask[row, src_col]:
                lo = max(0, src_col - 1)
                hi = src_col + 1
                # the value of inserted pixel
                res[row, res_col] = src[row, lo:hi].mean(axis=0)
                res_col += 1
            res[row, res_col] = src[row, src_col]
            res_col += 1
        assert res_col == src_w + delta_width

    return res


def _resize_width(src: np.ndarray, width: int,
                  keep_mask: Optional[np.ndarray]) -> np.ndarray:
    """ Resize the width of image by removing vertical seams. """
    assert src.size > 0 and src.ndim in (2, 3)
    assert width > 0

    src_w = src.shape[1]
    if src_w < width:
        res = _expand_width(src, width - src_w, keep_mask)
    else:
        res = _reduce_width(src, src_w - width, keep_mask)
    return res


def _resize_height(src: np.ndarray, height: int,
                   keep_mask: Optional[np.ndarray]) -> np.ndarray:
    """ Resize the height of image by removing horizontal seams. """
    assert src.ndim in (2, 3) and height > 0
    if src.ndim == 3:
        if keep_mask is not None:
            keep_mask = keep_mask.T
        src = _resize_width(src.transpose((1, 0, 2)), height, keep_mask).transpose((1, 0, 2))
    else:
        src = _resize_width(src.T, height, keep_mask.T).T
    return src


def _check_mask(mask: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """ Ensure the mask to be a 2D grayscale map of specific shape. """
    mask = np.asarray(mask, dtype=np.bool)
    if mask.ndim != 2:
        raise ValueError('Invalid mask of shape {}: expected to be a 2D '
                         'binary map'.format(mask.shape))
    if mask.shape != shape:
        raise ValueError('The shape of mask must match the image: expected {}, '
                         'got {}'.format(shape, mask.shape))
    return mask


def _check_src(src: np.ndarray) -> np.ndarray:
    """ Ensure the source to be RGB or grayscale. """
    src = np.asarray(src, dtype=np.uint8)
    if src.size == 0 or src.ndim not in (2, 3):
        raise ValueError('Invalid src of shape {}: expected an 3D RGB image or '
                         'a 2D grayscale image'.format(src.shape))
    return src


def resize(src: np.ndarray, width: int, height: int,
           keep_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """ Resize image with seam-carving algorithm. """
    src = _check_src(src)
    src_h, src_w = src.shape[:2]

    # Ensure that Width and Height are INTEGER
    width, height = int(round(width)), int(round(height))

    # Check if Width and Height is VALID
    if width <= 0 or height <= 0:
        raise ValueError('Invalid image size {}: expected > 0'.format((width, height)))

    # Check expended Size cannot be too LARGE
    if width >= 2 * src_w:
        raise ValueError('Invalid target width {}: expected less than twice '
                         'the source width (< {})'.format(width, 2 * src_w))
    if height >= 2 * src_h:
        raise ValueError('Invalid target height {}: expected less than twice '
                         'the source height (< {})'.format(height, 2 * src_h))

    # Check if Keep Mask is VALID
    if keep_mask is not None:
        keep_mask = _check_mask(keep_mask, (src_h, src_w))

    # Resize Image
    if width != src_w:
        src = _resize_width(src, width, keep_mask)
    if height != src_h:
        if src.shape[0] == keep_mask.shape[0] and src.shape[1] == keep_mask.shape[1]:
            src = _resize_height(src, height, keep_mask)

    return src


def remove_object(src: np.ndarray, drop_mask: np.ndarray,
                  keep_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Remove an object on the source image."""
    src = _check_src(src)

    drop_mask = _check_mask(drop_mask, src.shape[:2])

    if keep_mask is not None:
        keep_mask = _check_mask(keep_mask, src.shape[:2])

    gray = src if src.ndim == 2 else _rgb2gray(src)

    # drop pixels until all the pixels are dropped
    while drop_mask.any():
        energy = _get_energy(gray)
        # mark pixels need to be dropped
        energy[drop_mask] -= DROP_MASK_ENERGY
        # keep object
        if keep_mask is not None:
            energy[keep_mask] += KEEP_MASK_ENERGY
        # remove object
        seam = _get_seam(energy)
        seam_mask = _get_seam_mask(src, seam)
        gray = _remove_seam_mask(gray, seam_mask)
        # drop mask should also drop pixels
        drop_mask = _remove_seam_mask(drop_mask, seam_mask)
        src = _remove_seam_mask(src, seam_mask)
        if keep_mask is not None:
            keep_mask = _remove_seam_mask(keep_mask, seam_mask)

    return src
