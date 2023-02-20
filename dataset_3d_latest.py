import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import nibabel as nib
import os
import SimpleITK as sitk
from skimage import transform

# soft tissues: W:350–400 L:20–60


def clip_image(image):
    # window width = 400
    # window level = 50
    # so upper x = 50 +(400/2) = 250
    #  lower y = 50 - (400/2) = -200
    # clip images -200 to 250
    np_img = image
    np_img = np.clip(np_img, -200., 250.).astype(np.float32)
    return np_img


def normalise_zero_one(image):
    """Image normalisation. Normalises image to fit [0, 1] range."""

    image = image.astype(np.float32)

    minimum = np.min(image)
    maximum = np.max(image)

    if maximum > minimum:
        ret = (image - minimum) / (maximum - minimum)
    else:
        ret = image * 0.
    return ret


def resample_img(itk_image, out_spacing=[0.7, 0.7, 1.5], is_label=False):
    # resample images to 2mm spacing with simple itk

    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [
        int(np.round(original_size[0] *
            (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] *
            (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)


class HecktorDataset_one_patient(torch.utils.data.Dataset):
    def __init__(self, image_path, mask_path):
        self.image_path = image_path
        self.mask_path = mask_path

    def __getitem__(self, idx):
        img_sitk = sitk.ReadImage(
            self.image_path[idx], sitk.sitkFloat32)  # Reading CT
        mask_sitk = sitk.ReadImage(
            self.mask_path[idx], sitk.sitkInt32)  # Reading CT
        img_sitk = resample_img(img_sitk)
        mask_sitk = resample_img(mask_sitk, is_label=True)
        img_sitk = sitk.GetArrayFromImage(img_sitk)  # get array
        mask_sitk = sitk.GetArrayFromImage(mask_sitk)  # get array
        img_sitk = clip_image(img_sitk)
        slice_center = int(img_sitk.shape[0]/2)
        img_center = int(img_sitk.shape[1]/2)
        if img_sitk.shape[1] > 480 and img_sitk.shape[0] > 125:
            img_sitk = img_sitk[slice_center-60:slice_center+60,
                                img_center-240:img_center+192, img_center-216:img_center+216]
            mask_sitk = mask_sitk[slice_center-60:slice_center-60,
                                  img_center-240:img_center+192, img_center-216:img_center+216]
        else :
            img_sitk = img_sitk[img_sitk.shape[0]-121:img_sitk.shape[0]-1,
                                  :,:]
            mask_sitk = mask_sitk[mask_sitk.shape[0]-121:mask_sitk.shape[0]-1,
                                  :,:]
        img_sitk = normalise_zero_one(img_sitk)
        mask_sitk = np.where(mask_sitk == 1, 1, 0)
        img_sitk = transform.resize(img_sitk, (120, 144, 144))
        img_sitk = np.expand_dims(img_sitk,axis=0)
        mask_sitk = transform.resize(mask_sitk, (120, 144, 144), order=0)
        mask_sitk = np.expand_dims(mask_sitk,axis=0)
        return img_sitk, mask_sitk

    def __len__(self):
        return len(self.image_path)

