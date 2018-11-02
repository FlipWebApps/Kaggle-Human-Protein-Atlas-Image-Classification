"""
This module contains Kaggle submission related code.

"""
import numpy as np

# from tgssalt_challenge import features
# from tgssalt_challenge.io import IO


def rle_decode(rle_mask):
    """
    rle_mask: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    """
    if type(rle_mask) == float:
        return np.zeros([101, 101])
    s = rle_mask.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(101 * 101, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(101, 101)


def rle_encode(im, order='F', format=True):
    """
    used for converting the decoded image to rle mask
    im: numpy array, 1 - mask, 0 - background
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not
    Returns run length as string formated
    """
    pixels = im.flatten(order=order)
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    if format:
        return ' '.join(str(x) for x in runs)
    else:
        return runs