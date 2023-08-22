import torch
import numpy as np
import cv2 as cv
import math
import nibabel as nib
from einops import rearrange
import os

def to_uint8(vol):
    vol = vol.astype(float)
    vol[vol < 0] = 0    
    return ((vol - vol.min()) * 255.0 / vol.max()).astype(np.uint8)


def IR_to_uint8(vol):
    vol = vol.astype(np.float)
    vol[vol < 0] = 0
    return ((vol - 800) * 255.0 / vol.max()).astype(np.uint8)


def histeq_vol(vol):
    for ind in range(vol.shape[0]):
        vol[ind, :, :] = cv.equalizeHist(vol[ind, :, :])
    return vol


def read_vol(PATH):
    return nib.load(PATH).get_data().transpose(2, 0, 1)


def edge_vol(vol, kernel_size=(3, 3), sigmaX=0):
    edge = np.zeros(vol.shape, np.uint8)
    for ind in range(vol.shape[0]):
        edge[ind, :, :] = cv.Canny(vol[ind, :, :], 1, 1)
        edge[ind, :, :] = cv.GaussianBlur(edge[ind, :, :], kernel_size, sigmaX)
    return edge



def stack_vol(vol, stack_num):
    assert stack_num % 2 == 1, 'stack numbers must be odd!'
    vol = np.expand_dims(vol, axis=1)
    N = range(stack_num // 2, -(stack_num // 2 + 1), -1)
    stacked_vol = np.roll(vol, N[0], axis=0)
    for n in N[1:]:
        stacked_vol = np.concatenate((stacked_vol, np.roll(vol, n, axis=0)), axis=1)
    return stacked_vol


# crop
def calc_ceil_pad(x, devider):
    return math.ceil(x / float(devider)) * devider


def crop_vol(vol, crop_region):
    l_range, r_range, c_range = crop_region
    cropped_vol = vol[l_range[0]: l_range[1], r_range[0]: r_range[1],
                  c_range[0]: c_range[1]]
    return cropped_vol


def get_mask_region(vols, scale1, scale2):
    mask = np.zeros(vols[0].shape[:])
    for vol in vols:
        l, r, c = np.where(vol > 0)
        mask[l, r, c] = 1

    l, r, c = np.where(mask > 0)
    min_l, min_r, min_c = l.min(), r.min(), c.min()
    max_l, max_r, max_c = l.max(), r.max(), c.max()
    max_r = min_r + calc_ceil_pad(max_r - min_r, scale1)
    max_c = min_c + calc_ceil_pad(max_c - min_c, scale1)
    max_l = min_l + calc_ceil_pad(max_l - min_l, scale2)

    pad_r = 0
    pad_c = 0
    if (max_r - min_r) < 96:
        pad_r = (96 - (max_r - min_r))//2
    if (max_c - min_c) < 128:
        pad_c = (128 - (max_c - min_c))//2

    return [(min_l, max_l), (min_r - pad_r, max_r + pad_r), (min_c - pad_c, max_c + pad_c)]


class BrainData:
    def __init__(self):
        self.vols = []
        
    def read(self, PATH, IDs, suffix):
        for id_vol in IDs:
            self.vols.append(to_uint8(read_vol(os.path.join(PATH, id_vol + '_' + suffix))))

    # def adaptive_adjust_gamma(self):
        # self.vols = [adaptive_adjust_gamma_vol(vol) for vol in self.vols]

    def histeq(self):
        self.vols = [histeq_vol(vol) for vol in self.vols]

    def crop(self, crop_region):
        for ind in range(len(self.vols)):
            cropped_vol = crop_vol(self.vols[ind], crop_region[ind])
            self.vols[ind] = cropped_vol

    # def normalize(self):

    #     transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    #     self.vols = [rearrange(transform(vol), 'w c h -> c h w') for vol in self.vols]
    def transform(self):
        mean = [np.mean(vol) for vol in self.vols]
        mean = np.mean(mean)
        if self.vols[0].ndim == 2:
            self.vols = [torch.from_numpy((np.expand_dims(vol, 0).astype(np.float) - mean) / 255.0).float()
                         for vol in self.vols]
        else:
            self.vols = [torch.from_numpy((vol.astype(np.float) - mean) / 255.0).float()
                         for vol in self.vols]

    def stack(self, stack_num):
        self.vols = [stack_vol(vol, stack_num) for vol in self.vols]


    def split(self):
        temp = []
        for vol in self.vols:
            temp.extend(np.split(vol, vol.shape[0], axis=0))
        self.vols = temp

    def expand(self, times):
        new_vols = []
        for i in range(times):
            new_vols.extend(self.vols)
        self.vols = new_vols


class Brain_Img(BrainData):
    def __init__(self):
        super().__init__()
    def crop(self, crop_region):
        for ind in range(len(self.vols)):
            cropped_vol = crop_vol(self.vols[ind], crop_region)
            self.vols[ind] = cropped_vol

class Brain_Label(Brain_Img):
    def __init__(self):
        super().__init__()

    def trans_vol_label(self, label_s, label_t):
        assert len(label_s) == len(label_t), 'length must be same!'
        self.num_classes = max(label_t) + 1
        for i in range(len(self.vols)):
            vol = np.zeros(self.vols[i].shape)
            for j in range(len(label_s)):
                l, r, c = np.where(self.vols[i] == label_s[j])
                vol[l, r, c] = label_t[j]
            self.vols[i] = vol

    def split(self):
        vols = []
        for vol in self.vols:
            vols.extend(np.split(vol, vol.shape[0], axis=0))
        self.vols = vols

##配准后读取 
def read_IBSR_vol(path):
    vol = nib.load(path).get_fdata().transpose(1, 2, 0)
    return np.flip(vol, axis=1)
    
#原图读取 
#def read_IBSR_vol(path):
#    vol = nib.load(path).get_fdata().squeeze().transpose(2, 1, 0)
#    return np.flip(vol, axis=2)

class IBSR_Img(Brain_Img):
    def __init__(self):
        super().__init__()
    def read(self, PATH, IDs):
        for id_vol in IDs:
            self.vols.append(to_uint8(read_IBSR_vol(os.path.join(
                PATH, id_vol +'_img_re.nii'.format(id_vol)))))
        #self.vols.append(to_uint8(read_IBSR_vol(os.path.join(
                #PATH, 'IBSR_{}_ana_stripped.nii'.format(id_vol)))))
            self.affine = nib.load(os.path.join(
                PATH, id_vol +'_img_re.nii'.format(id_vol))).affine    
class IBSR_Label(Brain_Label):
    def __init__(self):
        super().__init__()
    def read(self, PATH, IDs):
        for id_vol in IDs:
            self.vols.append(read_IBSR_vol(os.path.join(PATH, id_vol + '_seg_6_re.nii'.format(id_vol))))
            #self.vols.append(read_IBSR_vol(os.path.join(PATH, 'IBSR_{}_seg.nii'.format(id_vol))))

#原图 
#def read_MALC_vol(path):
#    vol = nib.load(path).get_fdata().transpose(2, 1, 0)
#    return vol

##配准后
def read_MALC_vol(path):
    vol = nib.load(path).get_fdata().transpose(1, 2, 0)
    return np.flip(vol, axis=1)

class MALC_Img(Brain_Img):
    def __init__(self):
        super().__init__()
    def read(self, PATH, IDs):
        for id_vol in IDs:
            self.vols.append(to_uint8(read_MALC_vol(os.path.join(
                PATH, id_vol +'_img_re.nii'.format(id_vol)))))
                #PATH, 'skull_strip-{}_img.nii'.format(id_vol)))))
            self.affine = nib.load(os.path.join(
                PATH, id_vol +'_img_re.nii'.format(id_vol))).affine 


class MALC_Label(Brain_Label):
    def __init__(self):
        super().__init__()
    def read(self, PATH, IDs):
        for id_vol in IDs:
            self.vols.append(read_MALC_vol(os.path.join(PATH, id_vol + '_seg_6_re.nii'.format(id_vol)))) 
            #self.vols.append(read_MALC_vol(os.path.join(PATH, id_vol + '_seg.nii'.format(id_vol))))     
