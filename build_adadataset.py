from torch.utils import data
#from einops import rearrange
from yacs.config import CfgNode as CN

from utils.brain_data import *
from utils.transforms import to_tensor, random_flip_rotate
      

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

    # def __getitem__(self, index):
    #     src_img, src_label = to_tensor(self.src_img[index], self.src_label[index])
    #     tgt_img, tgt_label = to_tensor(self.tgt_img[index], self.tgt_label[index])
    #     if self.transform:
    #         src_img, src_label, tgt_img, tgt_label = self.transform(src_img, src_label, tgt_img, tgt_label)
    #     src_label = rearrange(src_label, '1 h w -> h w')
    #     tgt_label = rearrange(tgt_label, '1 h w -> h w')
    #     return src_img, src_label, tgt_img, tgt_label

    def __getitem__(self, index):
        src_img, src_label = to_tensor(self.src_img[index], self.src_label[index])
        tgt_img, tgt_label = to_tensor(self.tgt_img[index], self.tgt_label[index])
        if self.transform:
            src_img, src_label = self.transform(src_img, src_label)
            tgt_img, tgt_label = self.transform(tgt_img, tgt_label)
        src_label = rearrange(src_label, '1 h w -> h w')
        tgt_label = rearrange(tgt_label, '1 h w -> h w')
        return src_img, src_label, tgt_img, tgt_label


def build_dataset_DA(source, target, transform=None):

    i_source, l_source = source.dataset + '_Img', source.dataset + '_Label'
    i_target, l_target = target.dataset + '_Img', target.dataset + '_Label'

    source_train_img, source_train_label = globals()[i_source](), globals()[l_source]()
    target_train_img, target_train_label = globals()[i_target](), globals()[l_target]()
    test_img, test_label = globals()[i_target](), globals()[l_target]()
    eval_img, eval_label = globals()[i_target](), globals()[l_target]()
    
    source_train_img.read(source.PATH, source.IDs_train)
    source_train_label.read(source.PATH, source.IDs_train)
    source_train_label.trans_vol_label(source.label_s, source.label_t)
    train_mask_region = get_mask_region(source_train_label.vols, 16, 1)

    source_train_img.crop(train_mask_region)
    source_train_label.crop(train_mask_region)

    source_train_img.histeq()
    source_train_img.split()
    source_train_label.split()    
    
    source_train_img.expand(2)
    source_train_label.expand(2)

    target_train_img.read(target.PATH, target.IDs_train)
    target_train_label.read(target.PATH, target.IDs_train)
    train_target_mask_region = get_mask_region(target_train_label.vols, 16, 1)  # 不应该有目标域label
    target_train_img.crop(train_target_mask_region)
    target_train_label.crop(train_target_mask_region)

    target_train_img.histeq()
    target_train_img.split()
    target_train_label.split()     

    eval_img.read(target.PATH, target.IDs_eval)
    eval_label.read(target.PATH, target.IDs_eval)
    
    eval_img.crop(train_target_mask_region)
    eval_label.crop(train_target_mask_region)
    eval_label.trans_vol_label(target.label_s, target.label_t)

    eval_img.histeq()
    eval_img.split()
    eval_label.split()
    
    test_img.read(target.PATH, target.IDs_test)
    test_label.read(target.PATH, target.IDs_test)

    test_img.crop(train_target_mask_region)
    test_label.crop(train_target_mask_region)
    test_label.trans_vol_label(target.label_s, target.label_t)

    test_img.histeq()
    test_img.split()
    test_label.split()

    train_dataset = Dataset_DA(source_train_img.vols, source_train_label.vols, target_train_img.vols, target_train_label.vols, transform)
    test_dataset  = Dataset(test_img.vols, test_label.vols)
    eval_dataset = Dataset(eval_img.vols, eval_label.vols)
    
    return train_dataset, test_dataset, eval_dataset


def build_dataset(source, target, transform=None):
    i_source, l_source = source.dataset + '_Img', source.dataset + '_Label'
    i_target, l_target = target.dataset + '_Img', target.dataset + '_Label'

    source_train_img, source_train_label = globals()[i_source](), globals()[l_source]()
    test_img, test_label = globals()[i_target](), globals()[l_target]()
    eval_img, eval_label = globals()[i_target](), globals()[l_target]()
    
    source_train_img.read(source.PATH, source.IDs_train)
    source_train_label.read(source.PATH, source.IDs_train)
    source_train_label.trans_vol_label(source.label_s, source.label_t)
    train_mask_region = get_mask_region(source_train_label.vols, 16, 1)

    source_train_img.crop(train_mask_region)
    source_train_label.crop(train_mask_region)

    source_train_img.histeq()
    source_train_img.split()
    source_train_label.split()     

    eval_img.read(target.PATH, target.IDs_eval)
    eval_label.read(target.PATH, target.IDs_eval)

    eval_img.crop(train_mask_region)
    eval_label.crop(train_mask_region)
    eval_label.trans_vol_label(target.label_s, target.label_t)

    eval_img.histeq()
    eval_img.split()
    eval_label.split()
    
    test_img.read(target.PATH, target.IDs_test)
    test_label.read(target.PATH, target.IDs_test)

    test_img.crop(train_mask_region)
    test_label.crop(train_mask_region)
    test_label.trans_vol_label(target.label_s, target.label_t)

    test_img.histeq()
    test_img.split()
    test_label.split()

    train_src_dataset = Dataset(source_train_img.vols, source_train_label.vols, transform)
    test_dataset  = Dataset(test_img.vols, test_label.vols)
    eval_dataset = Dataset(eval_img.vols, eval_label.vols)
    
    return train_src_dataset, test_dataset, eval_dataset


 ###############
if __name__ == '__main__':

    source = CN()
    source.dataset = 'IBSR'
    source.PATH = '/home/huqian/baby/DA/IBSR_18/register2'
    source.label_s = (9, 10, 11, 12, 13, 17, 18, 48, 49, 50, 51, 52, 53, 54)
    source.label_t = (1, 1, 2, 3, 4, 5, 6, 1, 1, 2, 3, 4, 5, 6)
    source.IDs_train = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']

    target = CN()
    target.dataset = 'MALC'
    target.PATH = '/home/huqian/baby/DA/MICCAI/register2'
    target.label_s = (59, 60, 36, 37, 57, 58, 55, 56, 47, 48, 31, 32)
    target.label_t = (1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6)
    target.IDs_train = ['15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27',
    '28', '29', '30', '31', '32', '33', '34']
    target.IDs_eval = ['08', '09', '10', '11', '12', '13', '14']
    target.IDs_test = ['01', '02', '03', '04', '05', '06', '07']

    train_d, e_d, t_d = build_dataset_DA(source, target, random_flip_rotate)
    train_loader = data.DataLoader(train_d, batch_size=4, num_workers=1, shuffle=True)
    e_loader = data.DataLoader(e_d, batch_size=1, num_workers=1, shuffle=False)
    t_loader = data.DataLoader(t_d, batch_size=1, num_workers=1, shuffle=False)

    train_loader_iter = enumerate(train_loader)
    e_loader_iter = enumerate(e_loader)
    t_loader_iter = enumerate(t_loader)

    _, inputs = train_loader_iter.__next__()
    src_i, src_l, tgt_i, tgt_l = inputs
    print(src_i.shape)
    print(src_l.shape)    
