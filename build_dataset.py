from torch.utils import data
from einops import rearrange, reduce
from yacs.config import CfgNode as CN
import random

from utils.brain_data import *
from utils.transforms import img_to_tensor, to_tensor, random_flip_rotate, random_flip_rotate_aug
from utils.fft import FDA_source_to_target_np
from utils.utils import setup_seed  
from experiment_config import EXPERIMENTS    
from torchvision.transforms import functional as TF
from skimage.metrics import structural_similarity as ssim
import numpy as np


class Dataset(data.Dataset):
    def __init__(self, img, label, transform=None):
        self.img = img
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.img)

    def __getitem__(self, index):
        img, label = to_tensor(self.img[index], self.label[index])
        if self.transform:
            img, label = self.transform(img, label)
           
        label = rearrange(label, '1 h w -> h w')
        
        return img, label


class Dataset_DA(data.Dataset):
    def __init__(self, source_img, source_label, target_img, target_label, transform=None):
        self.src_img = source_img
        self.src_label = source_label
        self.tgt_img = target_img
        self.tgt_label = target_label
        self.transform = transform
    def __len__(self):
        return len(self.src_img)

    def __getitem__(self, index):

        
        src_img, src_label = self.src_img[index], self.src_label[index]
        tgt_img, tgt_label = self.tgt_img[index], self.tgt_label[index]
        src_img, src_label = to_tensor(src_img, src_label)
        tgt_img, tgt_label = to_tensor(tgt_img, tgt_label)
        if self.transform:
            src_img, src_label = self.transform(src_img, src_label)
            tgt_img, tgt_label = self.transform(tgt_img, tgt_label)

        src_label = rearrange(src_label, '1 h w -> h w')
        tgt_label = rearrange(tgt_label, '1 h w -> h w')

        return src_img, src_label, tgt_img, tgt_label


def random_colorjitter(img, range1, range2, range3):
    b_min, b_max = 1 - range1, 1 + range1
    c_min, c_max = 1 - range2, 1 + range2
    s_min, s_max = 1 - range3, 1 + range3
    
    b = torch.rand(1) * (b_max - b_min) + b_min
    c = torch.rand(1) * (c_max - c_min) + c_min
    s = torch.rand(1) * (s_max - s_min) + s_min
    
    img_aug = img.clone()
    if random.random() < 0.5:
        img_aug = TF.adjust_brightness(img_aug, b)
    if random.random() < 0.5:
        img_aug = TF.adjust_contrast(img_aug, c)
    if random.random() < 0.5:
        img_aug = TF.adjust_saturation(img_aug, s)

    # img_aug_min = reduce(img_aug, 'c h w -> c 1 1', 'min')
    # img_aug_max = reduce(img_aug, 'c h w -> c 1 1', 'max')
    # img_aug = (img_aug - img_aug_min).div(img_aug_max - img_aug_min)    

    return img_aug


def align_vols(src_vols, tgt_vols):
    new_src_vols = []
    new_tgt_vols = []
    l_src = len(src_vols)
    l_tgt = len(tgt_vols)
    
    for vol in src_vols:
        for i in range(l_tgt):
            new_src_vols.append(vol)
            
    for i in range(l_src):
        new_tgt_vols.extend(tgt_vols)
        
    return new_src_vols, new_tgt_vols

# def align_vols_ssim(src_img_list, tgt_img_list, tgt_label_list):
#     new_tgt_img_list = []
#     new_tgt_label_list = []
#     for i in range(len(src_img_list)):
#         src_img = src_img_list[i].squeeze()
#         index_range = range(i-2, i+3)
#         sim = []
#         for index in index_range:
#             if index >= len(src_img_list):
#                  continue
#             sim.append(ssim(src_img, tgt_img_list[index].squeeze()))
#         ind = np.argsort(sim)[-1]
#         new_tgt_img_list.append(tgt_img_list[index_range[ind]])
#         new_tgt_label_list.append(tgt_label_list[index_range[ind]])
    
#     return new_tgt_img_list, new_tgt_label_list


def align_vols_corrcoef_img(src_img_list, tgt_img_list, tgt_label_list):
    new_tgt_img_list = []
    new_tgt_label_list = []
    for i in range(len(src_img_list)):
        src_img = src_img_list[i]
        index_range = range(i-2, i+3)
        sim = []
        for index in index_range:
            if index >= len(src_img_list):
                 continue
            sim.append(np.corrcoef(src_img.flat, tgt_img_list[index].flat)[0, 1])
        ind = np.argsort(sim)[-1]
        new_tgt_img_list.append(tgt_img_list[index_range[ind]])
        new_tgt_label_list.append(tgt_label_list[index_range[ind]])
    
    return new_tgt_img_list, new_tgt_label_list


def align_vols_corrcoef(src_label_list, tgt_img_list, tgt_label_list):
    new_tgt_img_list = []
    new_tgt_label_list = []
    for i in range(len(src_label_list)):
        src_label_i = src_label_list[i]
        src_label_i = np.clip(src_label_i, a_min=0.004, a_max=1).flat
        index_range = range(i-2, i+3)
        sim = []
        for index in index_range:
            if index >= len(src_label_list):
                 continue
            tgt_label_i = tgt_label_list[index]
            tgt_label_i = np.clip(tgt_label_i, a_min=0.004, a_max=1).flat
            sim.append(np.corrcoef(src_label_i, tgt_label_i)[0, 1])
        ind = np.argsort(sim)[-1]
        new_tgt_img_list.append(tgt_img_list[index_range[ind]])
        new_tgt_label_list.append(tgt_label_list[index_range[ind]])
    
    return new_tgt_img_list, new_tgt_label_list  


# def align_vols_corrcoef(tgt_label_list, src_img_list, src_label_list):
#     new_src_img_list = []
#     new_src_label_list = []
#     for i in range(len(tgt_label_list)):
#         tgt_label_i = tgt_label_list[i]
#         tgt_label_i = np.clip(tgt_label_i, a_min=0.004, a_max=1).flat
#         index_range = range(i-2, i+3)
#         sim = []
#         for index in index_range:
#             if index >= len(src_label_list):
#                  continue
#             src_label_i = src_label_list[index]
#             src_label_i = np.clip(src_label_i, a_min=0.004, a_max=1).flat
#             sim.append(np.corrcoef(tgt_label_i, src_label_i)[0, 1])
#         ind = np.argsort(sim)[-1]
#         new_src_img_list.append(src_img_list[index_range[ind]])
#         new_src_label_list.append(src_label_list[index_range[ind]])
    
#     return new_src_img_list, new_src_label_list  


# def align_vols_ssim(src_label_list, tgt_img_list, tgt_label_list):
#     new_tgt_img_list = []
#     new_tgt_label_list = []
#     for i in range(len(src_label_list)):
#         src_label_i = src_label_list[i]
#         src_label_i = np.clip(src_label_i, a_min=0.004, a_max=1).squeeze()
#         index_range = range(i-2, i+3)
#         sim = []
#         for index in index_range:
#             if index >= len(src_label_list):
#                  continue
#             tgt_label_i = tgt_label_list[index]
#             tgt_label_i = np.clip(tgt_label_i, a_min=0.004, a_max=1).squeeze()
#             sim.append(ssim(src_label_i, tgt_label_i))
#         ind = np.argsort(sim)[-1]
#         new_tgt_img_list.append(tgt_img_list[index_range[ind]])
#         new_tgt_label_list.append(tgt_label_list[index_range[ind]])
    
#     return new_tgt_img_list, new_tgt_label_list  

def build_dataset_preDA(source, target, transform=None):

    i_source, l_source = source.dataset + '_Img', source.dataset + '_Label'
    i_target, l_target = target.dataset + '_Img', target.dataset + '_Label'

    source_train_img, source_train_label = globals()[i_source](), globals()[l_source]()
    target_train_img, target_train_label = globals()[i_target](), globals()[l_target]()
    test_img, test_label = globals()[i_target](), globals()[l_target]()
    eval_img, eval_label = globals()[i_target](), globals()[l_target]()
    
    source_train_img.read(source.PATH, source.IDs_train)
    source_train_label.read(source.PATH, source.IDs_train)        
    #source_train_label.trans_vol_label(source.label_s, source.label_t)
    train_mask_region = get_mask_region(source_train_label.vols, 16, 1)
    source_train_img.crop(train_mask_region)
    source_train_label.crop(train_mask_region)
    
    target_train_img.read(target.PATH, target.IDs_train)
    target_train_label.read(target.PATH, target.IDs_train)
    #train_target_mask_region = get_mask_region(target_train_label.vols, 16, 1)  # 不应该有目标域label
    target_train_img.crop(train_mask_region)
    target_train_label.crop(train_mask_region)

    source_train_img.vols, target_train_img.vols = align_vols(source_train_img.vols, target_train_img.vols)
    source_train_label.vols, target_train_label.vols = align_vols(source_train_label.vols, target_train_label.vols)
    
    source_train_img.histeq()
    source_train_img.split()
    source_train_label.split()    
    
#    source_train_img.expand(2)
#    source_train_label.expand(2)

    target_train_img.histeq()
    target_train_img.split()
    target_train_label.split()     

    eval_img.read(target.PATH, target.IDs_eval)
    eval_label.read(target.PATH, target.IDs_eval)    
    eval_img.crop(train_mask_region)
    eval_label.crop(train_mask_region)
#    eval_label.trans_vol_label(target.label_s, target.label_t)

    eval_img.histeq()
    eval_img.split()
    eval_label.split()
    
    test_img.read(target.PATH, target.IDs_test)
    test_label.read(target.PATH, target.IDs_test)

    test_img.crop(train_mask_region)
    test_label.crop(train_mask_region)
#    test_label.trans_vol_label(target.label_s, target.label_t)

    test_img.histeq()
    test_img.split()
    test_label.split()

    train_dataset = Dataset_DA(source_train_img.vols, source_train_label.vols, target_train_img.vols, target_train_label.vols, transform)
    test_dataset  = Dataset(test_img.vols, test_label.vols)
    eval_dataset = Dataset(eval_img.vols, eval_label.vols)
    
    return train_dataset, test_dataset, eval_dataset    

def build_dataset_DA(train_config, transform=None):

    i_source, l_source = train_config.source.dataset + '_Img', train_config.source.dataset + '_Label'
    i_target, l_target = train_config.target.dataset + '_Img', train_config.target.dataset + '_Label'

    source_train_img, source_train_label = globals()[i_source](), globals()[l_source]()
    target_train_img, target_train_label = globals()[i_target](), globals()[l_target]()
    test_img, test_label = globals()[i_target](), globals()[l_target]()
    eval_img, eval_label = globals()[i_target](), globals()[l_target]()
    
    source_train_img.read(train_config.source.PATH, train_config.source.IDs_train)
    source_train_label.read(train_config.source.PATH, train_config.source.IDs_train)
    train_mask_region = get_mask_region(source_train_label.vols, 16, 1)
    source_train_img.crop(train_mask_region)
    source_train_label.crop(train_mask_region)
    
    target_train_img.read(train_config.target.PATH, train_config.target.IDs_train)
    target_train_label.read(train_config.target.PATH, train_config.target.IDs_train)
    target_train_img.crop(train_mask_region)
    target_train_label.crop(train_mask_region)

    source_train_img.vols, target_train_img.vols = align_vols(source_train_img.vols, target_train_img.vols)
    source_train_label.vols, target_train_label.vols = align_vols(source_train_label.vols, target_train_label.vols)

    source_train_img.histeq()
    source_train_img.split()
    source_train_label.split()    

    target_train_img.histeq()
    target_train_img.split()
    target_train_label.split()     

    eval_img.read(train_config.target.PATH, train_config.target.IDs_eval)
    eval_label.read(train_config.target.PATH, train_config.target.IDs_eval)
    eval_img.crop(train_mask_region)
    eval_label.crop(train_mask_region)

    eval_img.histeq()
    eval_img.split()
    eval_label.split()
    
    test_img.read(train_config.target.PATH, train_config.target.IDs_test)
    test_label.read(train_config.target.PATH, train_config.target.IDs_test)
    test_img.crop(train_mask_region)
    test_label.crop(train_mask_region)

    test_img.histeq()
    test_img.split()
    test_label.split()

    train_dataset = Dataset_DA(source_train_img.vols, source_train_label.vols, target_train_img.vols, target_train_label.vols, transform)
    test_dataset  = Dataset(test_img.vols, test_label.vols)
    eval_dataset = Dataset(eval_img.vols, eval_label.vols)
    
    return train_dataset, eval_dataset, test_dataset


def build_dataset_DA_a(train_config, transform=None):

    i_source, l_source = train_config.source.dataset + '_Img', train_config.source.dataset + '_Label'
    i_target, l_target = train_config.target.dataset + '_Img', train_config.target.dataset + '_Label'

    source_train_img, source_train_label = globals()[i_source](), globals()[l_source]()
    target_train_img, target_train_label = globals()[i_target](), globals()[l_target]()
    test_img, test_label = globals()[i_target](), globals()[l_target]()
    eval_img, eval_label = globals()[i_target](), globals()[l_target]()
    
    source_train_img.read(train_config.source.PATH, train_config.source.IDs_train)
    source_train_label.read(train_config.source.PATH, train_config.source.IDs_train)
    train_mask_region = get_mask_region(source_train_label.vols, 16, 1)
    source_train_img.crop(train_mask_region)
    source_train_label.crop(train_mask_region)
    
    target_train_img.read(train_config.target.PATH, train_config.target.IDs_train)
    target_train_label.read(train_config.target.PATH, train_config.target.IDs_train)
    target_train_img.crop(train_mask_region)
    target_train_label.crop(train_mask_region)

    source_train_img.vols, target_train_img.vols = align_vols(source_train_img.vols, target_train_img.vols)
    source_train_label.vols, target_train_label.vols = align_vols(source_train_label.vols, target_train_label.vols)

    source_train_img.histeq()
    source_train_img.split()
    source_train_label.split()    

    target_train_img.histeq()
    target_train_img.split()
    target_train_label.split()     

    # target_train_img.vols, target_train_label.vols = align_vols_corrcoef(source_train_label.vols, target_train_img.vols, target_train_label.vols)
    target_train_img.vols, target_train_label.vols = align_vols_corrcoef_img(source_train_img.vols, target_train_img.vols, target_train_label.vols)
    eval_img.read(train_config.target.PATH, train_config.target.IDs_eval)
    eval_label.read(train_config.target.PATH, train_config.target.IDs_eval)
    eval_img.crop(train_mask_region)
    eval_label.crop(train_mask_region)

    eval_img.histeq()
    eval_img.split()
    eval_label.split()
    
    test_img.read(train_config.target.PATH, train_config.target.IDs_test)
    test_label.read(train_config.target.PATH, train_config.target.IDs_test)

    test_img.crop(train_mask_region)
    test_label.crop(train_mask_region)

    test_img.histeq()
    test_img.split()
    test_label.split()

    train_dataset = Dataset_DA(source_train_img.vols, source_train_label.vols, target_train_img.vols, target_train_label.vols, transform)
    test_dataset  = Dataset(test_img.vols, test_label.vols)
    eval_dataset = Dataset(eval_img.vols, eval_label.vols)
    
    return train_dataset, eval_dataset, test_dataset


def build_dataset(train_config, transform=None):

    i_source, l_source = train_config.source.dataset + '_Img', train_config.source.dataset + '_Label'
    i_target, l_target = train_config.target.dataset + '_Img', train_config.target.dataset + '_Label'

    source_train_img, source_train_label = globals()[i_source](), globals()[l_source]()
    test_img, test_label = globals()[i_target](), globals()[l_target]()
    eval_img, eval_label = globals()[i_target](), globals()[l_target]()
    
    source_train_img.read(train_config.source.PATH, train_config.source.IDs_train)
    source_train_label.read(train_config.source.PATH, train_config.source.IDs_train)
    train_mask_region = get_mask_region(source_train_label.vols, 16, 1)

    source_train_img.crop(train_mask_region)
    source_train_label.crop(train_mask_region)

    source_train_img.histeq()
    source_train_img.split()
    source_train_label.split()     


    eval_img.read(train_config.target.PATH, train_config.target.IDs_eval)
    eval_label.read(train_config.target.PATH, train_config.target.IDs_eval)

    eval_img.crop(train_mask_region)
    eval_label.crop(train_mask_region)

    eval_img.histeq()
    eval_img.split()
    eval_label.split()
    
    test_img.read(train_config.target.PATH, train_config.target.IDs_test)
    test_label.read(train_config.target.PATH, train_config.target.IDs_test)

    test_img.crop(train_mask_region)
    test_label.crop(train_mask_region)

    test_img.histeq()
    test_img.split()
    test_label.split()

    source_train_dataset = Dataset(source_train_img.vols, source_train_label.vols, transform)
    test_dataset  = Dataset(test_img.vols, test_label.vols)
    eval_dataset = Dataset(eval_img.vols, eval_label.vols)
    
    return source_train_dataset, eval_dataset, test_dataset


def get_pseudo_label(target_train_loader, model):
    model.eval()
    pred_all  = []
    for inputs in target_train_loader:
        img, _ = inputs
        img = img.cuda()        
        with torch.no_grad():
             _, outputs = model(img)
        pred = outputs.data.max(1).indices.cpu()
        pred_all.append(pred)
    pred_all = torch.cat(pred_all, dim=0)
    pred_all = torch.split(pred_all, 1, dim=0)
    
    return [pred.numpy() for pred in pred_all]


def build_dataset_DA_ca(train_config, model, transform=None):

    i_source, l_source = train_config.source.dataset + '_Img', train_config.source.dataset + '_Label'
    i_target, l_target = train_config.target.dataset + '_Img', train_config.target.dataset + '_Label'

    source_train_img, source_train_label = globals()[i_source](), globals()[l_source]()
    target_train_img, target_train_label = globals()[i_target](), globals()[l_target]()
    test_img, test_label = globals()[i_target](), globals()[l_target]()
    eval_img, eval_label = globals()[i_target](), globals()[l_target]()

    source_train_img.read(train_config.source.PATH, train_config.source.IDs_train)
    source_train_label.read(train_config.source.PATH, train_config.source.IDs_train)
    train_mask_region = get_mask_region(source_train_label.vols, 16, 1)
    source_train_img.crop(train_mask_region)
    source_train_label.crop(train_mask_region)

    target_train_img.read(train_config.target.PATH, train_config.target.IDs_train)
    target_train_label.read(train_config.target.PATH, train_config.target.IDs_train)
    target_train_img.crop(train_mask_region)
    target_train_label.crop(train_mask_region)

    source_train_img.vols, target_train_img.vols = align_vols(source_train_img.vols, target_train_img.vols)
    source_train_label.vols, target_train_label.vols = align_vols(source_train_label.vols, target_train_label.vols)

    source_train_img.histeq()
    source_train_img.split()
    source_train_label.split()    

    target_train_img.histeq()
    target_train_img.split()
    target_train_label.split()     

    target_train_dataset = Dataset(target_train_img.vols, target_train_label.vols)
    target_train_loader = data.DataLoader(target_train_dataset, batch_size=32, num_workers=1, shuffle=False)

    target_train_pseudo_label = get_pseudo_label(target_train_loader, model)

    # target_train_img.vols, target_train_label.vols = align_vols_corrcoef(
    #     source_train_label.vols, target_train_img.vols, target_train_pseudo_label)

    source_train_img.vols, source_train_label.vols = align_vols_corrcoef(
        target_train_pseudo_label, source_train_img.vols, source_train_label.vols)

    eval_img.read(train_config.target.PATH, train_config.target.IDs_eval) #(3, 218, 182, 182)
    eval_label.read(train_config.target.PATH, train_config.target.IDs_eval)
    eval_img.crop(train_mask_region) #(3, 75, 96, 128)
    eval_label.crop(train_mask_region)
    eval_img.histeq() #(3, 75, 96, 128)
    eval_img.split() #(225, 1, 96, 128)
    eval_label.split()

    test_img.read(train_config.target.PATH, train_config.target.IDs_test)
    test_label.read(train_config.target.PATH, train_config.target.IDs_test)

    test_img.crop(train_mask_region)
    test_label.crop(train_mask_region)

    test_img.histeq()
    test_img.split()
    test_label.split()

    train_dataset = Dataset_DA(source_train_img.vols, source_train_label.vols, target_train_img.vols, target_train_label.vols, transform)
    test_dataset  = Dataset(test_img.vols, test_label.vols)
    eval_dataset = Dataset(eval_img.vols, eval_label.vols)
    

    return train_dataset, eval_dataset, test_dataset


def build_dataset_cycada(train_config, transform=None):

    def align_vols_corrcoef(src_img_list, tgt_img_list, tgt_label_list):
        new_tgt_img_list = []
        new_tgt_label_list = []
        for i in range(len(src_img_list)):
            src_img = src_img_list[i]
            index_range = range(i-2, i+3)
            sim = []
            for index in index_range:
                if index >= len(src_img_list):
                    continue
                sim.append(np.corrcoef(src_img.flat, tgt_img_list[index].flat)[0, 1])
            ind = np.argsort(sim)[-1]
            new_tgt_img_list.append(tgt_img_list[index_range[ind]])
            new_tgt_label_list.append(tgt_label_list[index_range[ind]])
        
        return new_tgt_img_list, new_tgt_label_list

    i_source, l_source = train_config.source.dataset + '_Img', train_config.source.dataset + '_Label'
    i_target, l_target = train_config.target.dataset + '_Img', train_config.target.dataset + '_Label'

    source_train_img, source_train_label = globals()[i_source](), globals()[l_source]()
    target_train_img, target_train_label = globals()[i_target](), globals()[l_target]()
    test_img, test_label = globals()[i_target](), globals()[l_target]()
    eval_img, eval_label = globals()[i_target](), globals()[l_target]()
    
    source_train_img.read(train_config.source.PATH, train_config.source.IDs_train)
    source_train_label.read(train_config.source.PATH, train_config.source.IDs_train)
    train_mask_region = get_mask_region(source_train_label.vols, 16, 1)
    source_train_img.crop(train_mask_region)
    source_train_label.crop(train_mask_region)
    
    target_train_img.read(train_config.target.PATH, train_config.target.IDs_train)
    target_train_label.read(train_config.target.PATH, train_config.target.IDs_train)
    target_train_img.crop(train_mask_region)
    target_train_label.crop(train_mask_region)

    source_train_img.vols, target_train_img.vols = align_vols(source_train_img.vols, target_train_img.vols)
    source_train_label.vols, target_train_label.vols = align_vols(source_train_label.vols, target_train_label.vols)

    source_train_img.histeq()
    source_train_img.split()
    source_train_label.split()    

    target_train_img.histeq()
    target_train_img.split()
    target_train_label.split()     

    target_train_img.vols, target_train_label.vols = align_vols_corrcoef(source_train_img.vols, target_train_img.vols, target_train_label.vols)

    eval_img.read(train_config.target.PATH, train_config.target.IDs_eval)
    eval_label.read(train_config.target.PATH, train_config.target.IDs_eval)
    eval_img.crop(train_mask_region)
    eval_label.crop(train_mask_region)

    eval_img.histeq()
    eval_img.split()
    eval_label.split()
    
    test_img.read(train_config.target.PATH, train_config.target.IDs_test)
    test_label.read(train_config.target.PATH, train_config.target.IDs_test)

    test_img.crop(train_mask_region)
    test_label.crop(train_mask_region)

    test_img.histeq()
    test_img.split()
    test_label.split()

    train_dataset = Dataset_DA(source_train_img.vols, source_train_label.vols, target_train_img.vols, target_train_label.vols, transform)
    test_dataset  = Dataset(test_img.vols, test_label.vols)
    eval_dataset = Dataset(eval_img.vols, eval_label.vols)
    
    return train_dataset, eval_dataset, test_dataset



if __name__ == '__main__':
    setup_seed(20)

    TRAIN_CONFIG = EXPERIMENTS[0]
    # src_d, tgt_d, eval_d, test_d, num_iters = build_dataset_DA(TRAIN_CONFIG, random_flip_rotate)
    # print(src_d.__len__())
    # print(tgt_d.__len__())    
    # print(num_iters)
    # src_loader = data.DataLoader(src_d, batch_size=4, num_workers=1, shuffle=True)
    # src_loader_iter = enumerate(src_loader)
    # for i in range(10):
    #     _, inputs = src_loader_iter.__next__()
    #     src_i, src_l = inputs
    #     print(src_i.mean())

    train_d, eval_d, test_d = build_dataset_DA(TRAIN_CONFIG, random_flip_rotate)
    train_loader = data.DataLoader(train_d, batch_size=1, num_workers=1, shuffle=False)
    print(train_loader.__len__())

    train_iter = enumerate(train_loader)
    for i in range(10):
        _, inputs = train_iter.__next__()
        src_i, src_l, tgt_i, tgt_l = inputs
        # print(tgt_i.mean())

