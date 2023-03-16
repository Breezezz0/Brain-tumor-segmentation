import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import nibabel as nib
import os
import SimpleITK as sitk
from skimage import transform
import cv2


def clip_image(image):
    # window width = 80
    # window level = 40
    # so upper x = 40 +(80/2) = 80
    #  lower y = 40 - (80/2) = 0
    # clip images -200 to 250
    np_img = image
    np_img = np.clip(np_img, 0., 80.).astype(np.float32)
    return np_img

#檢測背景
def crop_and_resize_image(image, label, size, display=False):
    if np.min(image) < 0 :
        image = clip_image(image)
    cropped_image = np.zeros((image.shape[0], size, size))
    cropped_label = np.zeros((image.shape[0], size, size))
    for slice in range(image.shape[0]) :

        # Create a mask with the background pixels
        img_slice = image[slice,:,:]
        label_slice = label[slice,:,:]
        mask = img_slice == 0



        # Find the brain area

        coords = np.array(np.nonzero(~mask))

        top_left = np.min(coords, axis=1)

        bottom_right = np.max(coords, axis=1)
        cropped_image = img_slice[top_left[0]:bottom_right[0],
                                top_left[1]:bottom_right[1]]
        cropped_label = label_slice[top_left[0]:bottom_right[0],
                                top_left[1]:bottom_right[1]]
        
        h, w = cropped_image.shape[:2]
        c = cropped_image.shape[2] if len(cropped_image.shape)>2 else 1
        if h == w: 
            return cv2.resize(cropped_image, size, cv2.INTER_AREA)

        dif = h if h > w else w

        interpolation = cv2.INTER_AREA if dif > (size[0]+size[1])//2 else cv2.INTER_CUBIC
        x_pos = (dif - w)//2
        y_pos = (dif - h)//2
        if len(cropped_image.shape) == 2:
            mask_image = np.zeros((dif, dif), dtype=cropped_image.dtype)
            mask_image[y_pos:y_pos+h, x_pos:x_pos+w] = cropped_image[:h, :w]
            mask_label = np.zeros((dif, dif), dtype=cropped_image.dtype)
            mask_label[y_pos:y_pos+h, x_pos:x_pos+w] = cropped_image[:h, :w]
        else:
            mask_image = np.zeros((dif, dif, c), dtype=cropped_image.dtype)
            mask_image[y_pos:y_pos+h, x_pos:x_pos+w, :] = cropped_image[:h, :w, :]
            mask_label = np.zeros((dif, dif), dtype=cropped_image.dtype)
            mask_label[y_pos:y_pos+h, x_pos:x_pos+w] = cropped_image[:h, :w]
        image_processed = cv2.resize(mask_image, (size,size), interpolation)
        label_processed = cv2.resize(mask_label, (size,size), interpolation)
        # image_processed = transform.resize(img_slice[top_left[0]:bottom_right[0],
        #                         top_left[1]:bottom_right[1]], (size, size))
        # label_processed = transform.resize(label_slice[top_left[0]:bottom_right[0],
        #                         top_left[1]:bottom_right[1]], (size, size), order=0)


        # Remove the background

        cropped_image[slice,:,:] = image_processed
        cropped_label[slice,:,:] = label_processed
    return cropped_image, cropped_label

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
    def __init__(self, image_path, mask_path, preprocessing = True):
        self.image_path = image_path
        self.mask_path = mask_path
        self.preprocessing = preprocessing
    def __getitem__(self, idx):
        img_sitk = sitk.ReadImage(
            self.image_path[idx], sitk.sitkFloat32)  # Reading CT
        mask_sitk = sitk.ReadImage(
            self.mask_path[idx], sitk.sitkInt32)  # Reading CT
        img_sitk = resample_img(img_sitk)
        mask_sitk = resample_img(mask_sitk, is_label=True)
        img_sitk = sitk.GetArrayFromImage(img_sitk)  # get array
        mask_sitk = sitk.GetArrayFromImage(mask_sitk)  # get array
        if self.preprocessing :
            img_sitk, mask_sitk = crop_and_resize_image(img_sitk, mask_sitk, 256)
        else :
            img_sitk = transform.resize(img_sitk, (80, 256, 256))
            mask_sitk = transform.resize(mask_sitk, (80, 256, 256), order=0)
        img_center = int(img_sitk.shape[1]/2)
        # img_sitk = img_sitk[img_sitk.shape[0]-81:img_sitk.shape[0]-1,
        #                     img_center-250:img_center+182, img_center-216:img_center+216]
        # mask_sitk = mask_sitk[mask_sitk.shape[0]-81:mask_sitk.shape[0]-1,
        #                       img_center-250:img_center+182, img_center-216:img_center+216]
        img_sitk = normalise_zero_one(img_sitk)
        mask_sitk = np.where(mask_sitk == 1, 1, 0)
        # img_sitk, mask_sitk = crop_image(img_sitk, mask_sitk)
        # img_sitk = transform.resize(img_sitk, (80, 256, 256))
        # mask_sitk = transform.resize(mask_sitk, (80, 256, 256), order=0)
        return img_sitk, mask_sitk

    def __len__(self):
        return len(self.image_path)

if __name__ == '__main__':
    image_dir = r'hecktor2022_training_corrected_v3\hecktor2022_training\hecktor2022\imagesTr'
    label_dir = r'hecktor2022_training_corrected_v3\hecktor2022_training\hecktor2022\labelsTr'
    image_path = []
    label_path = []
    for filename in sorted(os.listdir(image_dir)):
        if 'CT' in filename:
            fullpath = os.path.join(image_dir, filename)
            if os.path.isfile(fullpath):
                image_path.append(fullpath)
    for filename in sorted(os.listdir(label_dir)):
        fullpath = os.path.join(label_dir, filename)
        if os.path.isfile(fullpath):
            label_path.append(fullpath)
    tumor_slice_count = 0
    dataset_org = HecktorDataset_one_patient(image_path[:10],label_path[:10],preprocessing=False)
    dataset_processed = HecktorDataset_one_patient(image_path[:10],label_path[:10])
    # for i in range(len(dataset)) :
    #     image , mask = dataset[i]
    #     for slice in range(mask.shape[0]) :
    #         if np.sum(mask[slice,:,:]) > 0 :
    #             tumor_slice_count += 1
    image_original, mask_org = dataset_org[0]
    image_processed, mask_processed = dataset_processed[0]

    # for i in mask.flatten():
    #     if i != 0 and i != 1:
    #         print('oops')
    f, axarr = plt.subplots(2, 3, figsize=(15, 15))
    axarr[0,0].imshow(np.squeeze(image_original[10, :, :]), cmap='gray', origin='lower')
    axarr[0,0].set_ylabel('Axial View', fontsize=14)
    axarr[0,0].set_xticks([])
    axarr[0,0].set_yticks([])
    axarr[0,0].set_title('CT_org', fontsize=14)

    axarr[0,1].imshow(np.squeeze(mask_org[10, :, :]), cmap='jet', origin='lower')
    axarr[0,1].axis('off')
    axarr[0,1].set_title('Mask', fontsize=14)

    axarr[0,2].imshow(np.squeeze(image_original[10, :, :]),
                    cmap='gray', alpha=1, origin='lower')
    axarr[0,2].imshow(np.squeeze(mask_org[10, :, :]),
                    cmap='jet', alpha=0.5, origin='lower')
    axarr[0,2].axis('off')
    axarr[0,2].set_title('Overlay', fontsize=14)
    axarr[1,0].imshow(np.squeeze(image_processed[10, :, :]), cmap='gray', origin='lower')
    axarr[1,0].set_ylabel('Axial View', fontsize=14)
    axarr[1,0].set_xticks([])
    axarr[1,0].set_yticks([])
    axarr[1,0].set_title('CT_processed', fontsize=14)

    axarr[1,1].imshow(np.squeeze(mask_processed[10, :, :]), cmap='jet', origin='lower')
    axarr[1,1].axis('off')
    axarr[1,1].set_title('Mask', fontsize=14)

    axarr[1,2].imshow(np.squeeze(image_processed[10, :, :]),
                    cmap='gray', alpha=1, origin='lower')
    axarr[1,2].imshow(np.squeeze(mask_processed[10, :, :]),
                    cmap='jet', alpha=0.5, origin='lower')
    axarr[1,2].axis('off')
    axarr[1,2].set_title('Overlay', fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()