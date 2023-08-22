import sys
sys.path.append('/home/data/hq/DA')

import time
import argparse
import os
import shutil
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, LambdaLR
from torch.utils import data
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import scipy.misc
import random
from build_dataset import build_dataset_DA, build_dataset_DA_ca
from models.layers import conv_block, up_conv
from utils.metrics import MHDValue, DiceScore
from utils.loss import dice_loss, DiceLoss, DeepInfoMaxLoss
from utils.utils import set_requires_grad, load_pretrained, setup_seed
from utils.transforms import random_flip_rotate
from utils.fft import FDA_source_to_target
from experiment_config import EXPERIMENTS,EXPERIMENTS_m

from runx.logx import logx
from einops import rearrange,reduce

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(comment='MI')

class U_Net_4(nn.Module):

    def __init__(self, in_ch=3, num_classes=7):
        super(U_Net_4, self).__init__()
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up4 = up_conv(filters[4], filters[3])
        self.Up_conv4 = conv_block(filters[4], filters[3])

        self.Up3 = up_conv(filters[3], filters[2])
        self.Up_conv3 = conv_block(filters[3], filters[2])

        self.Up2 = up_conv(filters[2], filters[1])
        self.Up_conv2 = conv_block(filters[2], filters[1])

        self.Up1 = up_conv(filters[1], filters[0])
        self.Up_conv1 = conv_block(filters[1], filters[0])

        self.cls = nn.Conv2d(filters[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool(e4)
        e5 = self.Conv5(e5)

        d4 = self.Up4(e5)
        d4 = torch.cat((e4, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e3, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e2, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Up1(d2)
        d1 = torch.cat((e1, d1), dim=1)
        d1 = self.Up_conv1(d1)

        out = self.cls(d1)

        #if self.training:
        return d1, out
        #else:
            #return out


class PosNeg(nn.Module):
    def __init__(self, input_nc, ndf=64, num_classes=7):
        super(PosNeg, self).__init__()
        self.proto_projection = nn.Sequential(
            nn.Conv2d(input_nc, ndf, kernel_size=1),
            nn.BatchNorm2d(ndf),
            nn.ReLU(inplace=True))
        self.proto_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten())        
        self.proto_D = nn.Sequential(
            nn.Conv2d(input_nc, ndf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.cls = nn.Conv2d(ndf * 2, num_classes, kernel_size=1, stride=1)

    def forward(self, fea, label):            
        mask_pos = label.cuda()        
        mask_neg = 1. - mask_pos 
        # fea_pos = self.proto_pool(self.proto_projection(fea.cuda()) * mask_pos)                      
        # fea_neg = self.proto_pool(self.proto_projection(fea.cuda()) * mask_neg)
        #print(fea_pos.shape)
        out_pos = self.cls(self.proto_D(fea.cuda())) * mask_pos
        out_neg = self.cls(self.proto_D(fea.cuda())) * mask_neg
        # norm_factor = np.prod(np.array(mask_pos.shape[-2:]))# 取mask_pos的h w, 创建新数组 ,元素相乘
        # out_pos = fea_pos * ((norm_factor) / (mask_pos.sum(list(range(1, mask_pos.ndim))).reshape(
        #    (mask_pos.size(0), 1)) + 1e-6))  # 

        # out_neg = fea_neg * ((norm_factor) / (mask_neg.sum(list(range(1, mask_pos.ndim))).reshape(
        #    (mask_pos.size(0), 1)) + 1e-6))
        #print(out_pos.shape)

        return out_pos, out_neg

class PixelDiscriminator_(nn.Module):
    def __init__(self, input_nc, ndf=128, num_classes=7):
        super(PixelDiscriminator_, self).__init__()

        self.D = nn.Sequential(
            nn.Conv2d(input_nc, ndf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(ndf, ndf//2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.cls1 = nn.Conv2d(ndf//2, num_classes, kernel_size=1, stride=1)
        self.cls2 = nn.Conv2d(ndf//2, num_classes, kernel_size=1, stride=1)

    def forward(self, x):
        out = self.D(x)
        src_out = self.cls1(out)
        tgt_out = self.cls2(out)
        out = torch.cat((src_out, tgt_out), dim=1)
        return out
class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, num_classes=7):
        super(PixelDiscriminator, self).__init__()

        self.D = nn.Sequential(
            nn.Conv2d(input_nc, ndf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.cls = nn.Conv2d(ndf * 2, num_classes, kernel_size=1, stride=1)

    def forward(self, x):
        out = self.cls(self.D(x))
        return out

def pseudo_to_binarymask(pseudo_label):
    pseudo_label = pseudo_label.cpu()
    print(pseudo_label)
    N, C, H, W = pseudo_label.shape
    binary_mask = np.zeros((N, 1, H, W), dtype = int) 
    for i in range(C):
        a = np.expand_dims(pseudo_label[:, i], axis=1)
        
        binary_mask = binary_mask | a #.cpu().data.numpy()
    #print(binary_mask.shape)
    return torch.from_numpy(binary_mask).float()


def entropy_confidence_mask(logits, th=0.1):
    prob = torch.softmax(logits, dim=1)
    entropy = torch.sum(-prob * torch.log(prob + 1e-10), dim=1).detach()
    mask = entropy.ge(th)
    return mask


def to_one_hot(label, num_classes=7):
    b, h, w = label.shape
    label = rearrange(label, 'b h w -> b 1 h w')
    label = torch.zeros(b, num_classes, h, w, dtype=torch.int64).cuda().scatter_(1, label, 1)
    return label

def to_one(label, num_classes=7):
    b, h, w = label.shape   
    label = rearrange(label, 'b h w -> b 1 h w')
    label = torch.where(label!=0,1,0)
    return label.float()

def soft_label_cross_entropy(pred, soft_label, pixel_weights=None):
    N, C, H, W = pred.shape
    loss = -soft_label.float()*F.log_softmax(pred, dim=1)
    if pixel_weights is None:
        return torch.mean(torch.sum(loss, dim=1))
    return torch.mean(pixel_weights*torch.sum(loss, dim=1))

class Criterion(nn.Module):
    def __init__(self, num_classes, ignore_index=255):
        super(Criterion, self).__init__()
        self.sce_loss = SymmetricCrossEntropyLoss(ignore_index=ignore_index)
        self.dice_loss = DiceLoss(num_classes, ignore_index=ignore_index)

    def forward(self, logits, label, alpha, beta):
        return self.sce_loss(logits, label, alpha, beta) + self.dice_loss(logits, label)


def symmetric_cross_entropy(logits, label, alpha=1, beta=1):
    ce_loss = F.cross_entropy(logits, label)
    pred = torch.softmax(logits, dim=1)
    label_one_hot = to_one_hot(label, 7)
    label_one_hot = torch.clamp(label_one_hot.float(), min=1e-4, max=1.0)
    rce_loss = torch.mean(torch.sum(-pred * torch.log(label_one_hot), dim=1))
    return alpha * ce_loss + beta * rce_loss


class SymmetricCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super(SymmetricCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, logits, label, alpha=1, beta=1):
        label = rearrange(label, 'b h w -> 1 (b h w)')
        ind = torch.where(label != self.ignore_index)[1]
        label = torch.index_select(label, 1, ind)

        logits = rearrange(logits, 'b c h w -> c (b h w)')
        logits = torch.index_select(logits, 1, ind)

        label = rearrange(label, '1 l -> 1 l 1')
        logits = rearrange(logits, 'c l -> 1 c l 1')

        loss = symmetric_cross_entropy(logits, label, alpha, beta)

        return loss

def validation(model, eval_loader):
    model.eval()
    pred_all = []
    label_all = []
    for inputs in eval_loader:
        img, label = inputs    
        img = img.cuda()
        with torch.no_grad():
            _, outputs = model(img)
            outputs = outputs[0, :, :, :]
        pred = outputs.data.max(0)[1].cpu()
        pred_all.append(pred)
        label_all.append(label)
    pred_all = torch.stack(pred_all, dim=0)
    label_all = torch.cat(label_all, dim=0)
    score = DiceScore(pred_all, label_all, 7)

    logx.msg('eval:')
    logx.msg('Mean Dice: {}'.format(score['Mean Dice']))
    logx.msg('Thalamus: {}'.format(score['Dice'][0]))
    logx.msg('Caudate: {}'.format(score['Dice'][1]))
    logx.msg('Putamen: {}'.format(score['Dice'][2]))
    logx.msg('Pallidum: {}'.format(score['Dice'][3]))
    logx.msg('Hippocampus: {}'.format(score['Dice'][4]))
    logx.msg('Amygdala: {}'.format(score['Dice'][5]))

    return score

def eval(model, best_checkpoint, test_loader):
    checkpoint = torch.load(best_checkpoint)
    model_state_dict = checkpoint['st_model_state_dict']
    load_pretrained(model, model_state_dict)
    model.eval()
    pred_all = []
    label_all = []
    for inputs in test_loader:
        img, label = inputs
        img = img.cuda()
        with torch.no_grad():
            _, outputs = model(img)
            outputs = outputs[0, :, :, :]
        pred = outputs.data.max(0)[1].cpu()
        pred_all.append(pred)
        label_all.append(label)
    pred_all = torch.stack(pred_all, dim=0)
    label_all = torch.cat(label_all, dim=0)
    score = DiceScore(pred_all, label_all, 7)
    score_mhd = MHDValue(pred_all, label_all, 7)

    logx.msg('test:')
    logx.msg('Mean Dice: {}'.format(score['Mean Dice']))
    logx.msg('Thalamus: {}'.format(score['Dice'][0]))
    logx.msg('Caudate: {}'.format(score['Dice'][1]))
    logx.msg('Putamen: {}'.format(score['Dice'][2]))
    logx.msg('Pallidum: {}'.format(score['Dice'][3]))
    logx.msg('Hippocampus: {}'.format(score['Dice'][4]))
    logx.msg('Amygdala: {}'.format(score['Dice'][5]))
    logx.msg('MHD Thalamus: {}'.format(score_mhd['MHD'][0]))
    logx.msg('MHD Caudate: {}'.format(score_mhd['MHD'][1]))
    logx.msg('MHD Putamen: {}'.format(score_mhd['MHD'][2]))
    logx.msg('MHD Pallidum: {}'.format(score_mhd['MHD'][3]))
    logx.msg('MHD Hippocampus: {}'.format(score_mhd['MHD'][4]))
    logx.msg('MHD Amygdala: {}'.format(score_mhd['MHD'][5]))
    return score, score_mhd

def test(model, best_checkpoint, test_loader):
    checkpoint = torch.load(best_checkpoint)
    model_state_dict = checkpoint['st_model_state_dict']
    load_pretrained(model, model_state_dict)
    model.eval()
    pred_all = []
    label_all = []
    for inputs in test_loader:
        img, label = inputs
        img = img.cuda()
        with torch.no_grad():
            _, outputs = model(img)
            outputs = outputs[0, :, :, :]
        pred = outputs.data.max(0)[1].cpu()
        pred_all.append(pred)
        label_all.append(label)
    pred_all = torch.stack(pred_all, dim=0)
    label_all = torch.cat(label_all, dim=0)
    score = DiceScore(pred_all, label_all, 7)
    score_mhd = MHDValue(pred_all, label_all, 7)

    logx.msg('test:')
    logx.msg('Mean Dice: {}'.format(score['Mean Dice']))
    logx.msg('Thalamus: {}'.format(score['Dice'][0]))
    logx.msg('Caudate: {}'.format(score['Dice'][1]))
    logx.msg('Putamen: {}'.format(score['Dice'][2]))
    logx.msg('Pallidum: {}'.format(score['Dice'][3]))
    logx.msg('Hippocampus: {}'.format(score['Dice'][4]))
    logx.msg('Amygdala: {}'.format(score['Dice'][5]))
    logx.msg('MHD Thalamus: {}'.format(score_mhd['MHD'][0]))
    logx.msg('MHD Caudate: {}'.format(score_mhd['MHD'][1]))
    logx.msg('MHD Putamen: {}'.format(score_mhd['MHD'][2]))
    logx.msg('MHD Pallidum: {}'.format(score_mhd['MHD'][3]))
    logx.msg('MHD Hippocampus: {}'.format(score_mhd['MHD'][4]))
    logx.msg('MHD Amygdala: {}'.format(score_mhd['MHD'][5]))
    logx.msg('----------------------------------------------------------------')

    log_(score, score_mhd, 'val')


def log_(score, score_mhd, phase = 'val', epoch=None):
    log = {
         'Mean Dice'      : score['Mean Dice'],
         'Thalamus'   : score['Dice'][0],
         'Caudate'    : score['Dice'][1],
         'Putamen'    : score['Dice'][2],
         'Pallidum'   : score['Dice'][3],
         'Hippocampus': score['Dice'][4],
         'Amygdala'   : score['Dice'][5],
         'MHD Thalamus'   : score_mhd['MHD'][0],
         'MHD Caudate'    : score_mhd['MHD'][1],
         'MHD Putamen'    : score_mhd['MHD'][2],
         'MHD Pallidum'   : score_mhd['MHD'][3],
         'MHD Hippocampus': score_mhd['MHD'][4],
         'MHD Amygdala'   : score_mhd['MHD'][5]
         }
    logx.metric(phase=phase, metrics=log, epoch=epoch)


def train(model, D, MI, train_loader, num_iters, optimizer, optimizer_D, st_model,optimizer_st, criterion, config, epoch):#
    model.train()
    st_model.train()
    D.train()
    loss_MI = DeepInfoMaxLoss(type="conv") #conv concat dot
    train_iter = enumerate(train_loader)
    for i in range(num_iters):
    # train segmentation network
        set_requires_grad(D, requires_grad=False)
        set_requires_grad(model, requires_grad=True)
        set_requires_grad(st_model, requires_grad=False)  
        model.zero_grad()
        _, inputs = train_iter.__next__()
        src_img, src_label, tgt_img = inputs[0].cuda(), inputs[1].cuda(), inputs[2].cuda()
        src_fea, src_logits = model(src_img)
        src_label_ = to_one(src_label, 7) # 4,7,96,128 to 4,1,96,128      
        src_pos, src_neg = MI(src_fea,src_label_) #正负特征筛选
        src_logits = src_logits.div(config.t)
        loss_seg_src = F.cross_entropy(src_logits, src_label) + dice_loss(src_logits, src_label) #源域分割损失 

        with torch.no_grad(): #目标域数据风格变换
            #tgt_img_aug = random_colorjitter(tgt_img, 0.5, 0.5, 0.5)
            tgt_img_aug = FDA_source_to_target(tgt_img, src_img)
            img_aug_min = reduce(tgt_img_aug, 'b c h w -> b c 1 1', 'min')
            img_aug_max = reduce(tgt_img_aug, 'b c h w -> b c 1 1', 'max')
            tgt_img_aug = (tgt_img_aug - img_aug_min).div(img_aug_max - img_aug_min) 
        #学生模型对目标域输出
            _, tgt_logits_st = st_model(tgt_img)
            mask = entropy_confidence_mask(tgt_logits_st, 0.1) #对输出结果取置信区间
            tgt_pseudo_label = tgt_logits_st.max(1).indices
            tgt_pseudo_label_ = tgt_pseudo_label.clone()               
            tgt_pseudo_label_[torch.where(mask)] = 255 
            tgt_pseudo_label = to_one_hot(tgt_pseudo_label, 7) #0 1 编码
            src_label = to_one_hot(src_label, 7)     
       #教师模型对目标域输出
        tgt_fea_aug, tgt_logits_aug = model(tgt_img_aug) 
        tgt_pseudo_aug = tgt_logits_aug.max(1).indices  # 4,96,128
        tgt_pseudo_ = to_one(tgt_pseudo_aug,7)      
        tgt_pos, tgt_neg = MI(tgt_fea_aug, tgt_pseudo_) #正负特征筛选
        loss_seg_tgt = criterion(tgt_logits_aug, tgt_pseudo_label_, 0.5, 0.5) #学生模型结果对教师模型输出监督目标域分割损失
        loss_seg = loss_seg_src + 0.1*loss_seg_tgt  #教师模型一致性分割损失
             
        loss_mu = 0.5*loss_MI(src_pos, src_neg, tgt_pos) + 0.5*loss_MI(src_neg, src_pos, tgt_neg) #教师模型互信息损失
        tgt_D_pred = D(tgt_fea_aug) #目标域域判别损失
        loss_adv_aug = config.lambda_adv * soft_label_cross_entropy(tgt_D_pred, torch.cat((tgt_pseudo_label, torch.zeros_like(tgt_pseudo_label)), dim=1))
        loss = loss_seg + loss_adv_aug+ loss_mu #教师模型总损失
        loss.backward()
        optimizer.step()

    # train Discriminator
        set_requires_grad(D, requires_grad=True)
        set_requires_grad(model, requires_grad=False) #
        # train with source
        D.zero_grad()
        loss_D_src = config.lambda_D*soft_label_cross_entropy(D(src_fea.detach()), torch.cat((src_label, torch.zeros_like(src_label)), dim=1)) #D对源域判别
        loss_D_src.backward()
        # 目标域软特征伪标签分割损失
        loss_D_tgt = config.lambda_D*soft_label_cross_entropy(D(tgt_fea_aug.detach()), torch.cat((torch.zeros_like(tgt_pseudo_label), tgt_pseudo_label), dim=1))
        loss_D_tgt.backward()
        optimizer_D.step()

        set_requires_grad(D, requires_grad=False) 
        set_requires_grad(model, requires_grad=False)
        set_requires_grad(st_model, requires_grad=True)       
        st_model.zero_grad()  
        tgt_fea_st, tgt_logits_st = st_model(tgt_img)
        tgt_pseudo_label = to_one(tgt_logits_st.max(1).indices, 7) 
        tgt_aug_fea, tgt_aug_logits = model(tgt_img_aug)  
        with torch.no_grad():                      
             tgt_pseudo_logits = tgt_aug_logits.max(1).indices #原模型生成伪标签
             tgt_pseudo_src = to_one(tgt_pseudo_logits,7)           
             mask = entropy_confidence_mask(tgt_aug_logits, 0.1)
             tgt_pseudo_logits[torch.where(mask)] = 255 
        loss_seg_st = criterion(tgt_logits_st, tgt_pseudo_logits, 0.5, 0.5) #一致性蒸馏损失
        tgt_st_pos, tgt_st_neg = MI(tgt_fea_st, tgt_pseudo_label)
        src_pos, src_neg = MI(tgt_aug_fea, tgt_pseudo_src)

        loss_mu_st = 0.5*loss_MI(src_pos, src_neg, tgt_st_pos) + 0.5*loss_MI(src_neg, src_pos, tgt_st_neg) 
        loss_st = loss_seg_st + 0.5*loss_mu_st  #学生模型优化包括一致性分割损失和互信息损失
        loss_st.backward()        
        optimizer_st.step()

        #writer.add_scalar('loss/loss',loss , epoch)        
        #writer.add_scalar('loss/loss_seg',loss_seg , epoch)       
        #writer.add_scalar('loss/loss_mu',loss_mu , epoch) 
        #writer.add_scalar('loss/loss_st', loss_st , epoch) 


def main(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)

    setup_seed(config.seed)
    TRAIN_CONFIG = EXPERIMENTS[config.experiment] #EXPERIMENTS_m

    model = U_Net_4().cuda()
    D = PixelDiscriminator_(64).cuda()
    #D = PixelDiscriminator(7).cuda()
    st_model = U_Net_4().cuda()
    MI = PosNeg(64).cuda()

    checkpoint = torch.load(config.checkpoint) #预训练模型
    model.load_state_dict(checkpoint['model_state_dict'])
    st_model.load_state_dict(checkpoint['model_state_dict'])

    train_dataset, eval_dataset, test_dataset = build_dataset_DA_ca(TRAIN_CONFIG, model, random_flip_rotate)#build_dataset_DA_ca
    train_loader = data.DataLoader(train_dataset, batch_size=4, num_workers=1, shuffle=True)
    #test_loader = data.DataLoader(test_dataset, batch_size=1, num_workers=1, shuffle=False)
    eval_loader  = data.DataLoader(eval_dataset, batch_size=1, num_workers=1, shuffle=False)
    
    #eval 分割可视化
    #seg_checkpoint = torch.load(config.seg_checkpoint)
    #segout_IBSR(st_model, checkpoint, eval_loader)

    optimizer = torch.optim.SGD(model.parameters(), lr=config.init_lr, momentum=0.9, weight_decay=5e-4)
    optimizer_D = torch.optim.Adam(D.parameters(), lr=config.init_lr_D, betas=(0.5, 0.999)) #betas=(0.9, 0.99)
    optimizer_st = torch.optim.SGD(st_model.parameters(), lr=config.init_lr, momentum=0.9, weight_decay=5e-4)

    lr_decay_function = lambda epoch: 1.0 - max(0, epoch - config.milestone) / float(config.n_epochs - config.milestone)  # 衰减率
    #lr_decay_function = lambda epoch: 1.0 - max(0, epoch - 20) / float(40)
    scheduler = StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
    scheduler_st = StepLR(optimizer_st, step_size=config.step_size, gamma=config.gamma)
    scheduler_D = LambdaLR(optimizer_D, lr_lambda=lr_decay_function)

    criterion = Criterion(7)
    t_start = time.time()

    for epoch in range(config.n_epochs):
        logx.msg('epoch: {}'.format(epoch))        
        t_epoch = time.time()
        train(model, D, MI, train_loader, config.num_iters, optimizer, optimizer_D, st_model,optimizer_st, criterion, config, epoch) #
        scheduler.step()
        scheduler_st.step()
        scheduler_D.step()
        t_train = time.time()
        logx.msg('cost {:.2f} seconds in this train epoch'.format(t_train - t_epoch))
        score_eval = validation(st_model, eval_loader)
        validation(model, eval_loader)
        save_dict = {
             'st_model_state_dict':st_model.state_dict(),
             'model_state_dict':model.state_dict(),
             'D_state_dict':D.state_dict(),
             'MI_state_dict':MI.state_dict()
        }
        logx.save_model(save_dict, metric=score_eval['Mean Dice'], epoch=epoch, higher_better=True)

    best_checkpoint = logx.get_best_checkpoint()

     #test_loader
    test(st_model, best_checkpoint, eval_loader)

    t_end = time.time()
    logx.msg('cost {:.2f} minutes in this train epoch'.format((t_end - t_start) / 60))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--gpu_id', nargs='?', type=int, default=3)
    parser.add_argument('--seed', nargs='?', type=int, default=200)
    parser.add_argument('--batch_size', nargs='?', type=int, default=4)
    parser.add_argument('--n_epochs', nargs='?', type=int, default=80)
    parser.add_argument('--num_iters', nargs='?', type=int, default=250)
    parser.add_argument('--init_lr', nargs='?', type=float, default=1e-2)
    parser.add_argument('--init_lr_D', nargs='?', type=float, default=1e-4)
    parser.add_argument('--init_lr_MI', nargs='?', type=float, default=1e-4)
    parser.add_argument('--step_size', nargs='?', type=int, default=20) #10
    parser.add_argument('--gamma', nargs='?', type=float, default=0.1)
    parser.add_argument('--milestone', nargs='?', type=int, default=20)
    parser.add_argument('--logdir', nargs='?', type=str, default='/home/data/hq/DA/train/test_MALC')#test_MALC IBSR
    parser.add_argument('--checkpoint', nargs='?', type=str, default='/home/data/hq/DA/ada_pretrained/3.pth')
    #parser.add_argument('--seg_checkpoint', nargs='?', type=str, default='/home/data/hq/DA/train/test_IBSR/8.pth')

    parser.add_argument('--experiment', nargs='?', type=int, default=0)#7 0
    parser.add_argument('--lambda_mi', nargs='?', type=float, default=1)
    parser.add_argument('--lambda_adv', nargs='?', type=float, default=1)
    parser.add_argument('--lambda_D', nargs='?', type=float, default=0.5)
    parser.add_argument('--t', nargs='?', type=float, default=2)
    parser.add_argument('--th', nargs='?', type=float, default=0.1)
    parser.add_argument('--lambda_c', nargs='?', type=float, default=1)

    config = parser.parse_args()
    if os.path.exists(config.logdir):
        shutil.rmtree(config.logdir)
    logx.initialize(logdir=config.logdir, coolname=False, tensorboard=False, hparams=vars(config), no_timestamp=True)

    main(config)
