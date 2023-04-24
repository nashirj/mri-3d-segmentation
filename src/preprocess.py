import nibabel as nib
import numpy as np

from src import constants


def normalize_img(data):
    data_min = np.min(data)
    return (data - data_min) / (np.max(data) - data_min)


def get_masks(mask):
    # Adapted from https://www.kaggle.com/code/asmahekal/brain-tumor-mri-segmentation
    # WT = Whole tumor
    mask_WT = mask.copy()
    mask_WT[mask_WT == constants.EDEMA] = 1
    mask_WT[mask_WT == constants.NON_ENHANCING] = 1
    mask_WT[mask_WT == constants.ENHANCING] = 1

    # TC = Tumor core
    mask_TC = mask.copy()
    mask_TC[mask_TC == constants.EDEMA] = 0
    mask_TC[mask_TC == constants.NON_ENHANCING] = 1
    mask_TC[mask_TC == constants.ENHANCING] = 1

    # ET = Enhancing tumor
    mask_ET = mask.copy()
    mask_ET[mask_ET == constants.EDEMA] = 0
    mask_ET[mask_ET == constants.NON_ENHANCING] = 0
    mask_ET[mask_ET == constants.ENHANCING] = 1

    mask = np.stack([mask_WT, mask_TC, mask_ET])

    return mask


def binarize_mask(mask, threshold=0.5):
    mask[mask > threshold] = 1
    mask[mask <= threshold] = 0
    return mask.astype(np.uint8)


def load_stacked_mri_img(path):
    """Returns a 4D numpy array with shape [4, 240, 240, 155].
    
    Channel order: FLAIR, T1_W, T1_GD, T2_W
    """
    test_image=nib.load(path).get_fdata()
    # Change order of channels from [240, 240, 155, 4] to [4, 240, 240, 155]
    test_image = np.transpose(test_image, (3, 0, 1, 2))
    return test_image


def split_mri_img(img):
    # Channel order: FLAIR, T1_W, T1_GD, T2_W
    return [img[i, :, :, :] for i in range(4)]


def load_mask(path):
    return nib.load(path).get_fdata().astype(np.uint8)
