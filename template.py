import numpy as np
import nibabel as nib
import ants
import os
import warnings
warnings.filterwarnings('ignore')

def resample(img_path, sa_path):
    agent = nib.load(img_path)
    affine = agent.affine
    img = agent.get_fdata().squeeze().transpose(0, 2, 1)
    img = np.flip(img, axis=2)
    t = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    affine = t.dot(agent.affine).dot(t)
    agent_new = nib.Nifti1Image(img, affine)
    nib.save(agent_new, sa_path)


def ants_register(MNI_path, img_path, label_path, re_img_path, re_label_path):
    f_img = ants.image_read(MNI_path)  # fixed
    m_img = ants.image_read(img_path)  # moved
    m_label =  ants.image_read(label_path)
    mytx = ants.registration(fixed=f_img, moving=m_img, type_of_transform='Translation')
    warped_img = ants.apply_transforms(fixed=f_img, moving=m_img, transformlist=mytx['fwdtransforms'],
                                       interpolator="linear")
    warped_label = ants.apply_transforms(fixed=f_img, moving=m_label, transformlist=mytx['fwdtransforms'],
                                       interpolator="nearestNeighbor")
    ants.image_write(warped_img, re_img_path)
    ants.image_write(warped_label, re_label_path)


img_path = '/home/huqian/baby/DA/IBSR_18/img' #MICCAI  IBSR_18
ra_path = '/home/huqian/baby/DA/IBSR_18/resample'
re_path = '/home/huqian/baby/DA/IBSR_18/register' #2
template_path = '/home/huqian/baby/DA/train/MNI152_T1_1mm_Brain.nii.gz'

for i in range(1, 19):
    if i<10:
        i = '0' + str(i)
    else:
        i = str(i)
    img_i_path = os.path.join(img_path, i+'_img.nii')
    label_i_path = os.path.join(img_path, i+'_seg_6.nii')
    
    ra_img_i_path = os.path.join(ra_path, i+'_img_ra.nii')
    ra_label_i_path = os.path.join(ra_path, i+'_seg_ra.nii')
    
    re_img_i_path = os.path.join(re_path, i+'_img_re.nii')
    re_label_i_path = os.path.join(re_path, i+'_seg_6_re.nii')    

    resample(img_i_path, ra_img_i_path)
    resample(label_i_path, ra_label_i_path)
    
    
    ants_register(template_path, ra_img_i_path, ra_label_i_path,
                  re_img_i_path, re_label_i_path)

for i in range(1, 36):
    if i<10:
        i = '0' + str(i)
    else:
        i = str(i)
    img_i_path = os.path.join(img_path, i+'_img.nii')
    label_i_path = os.path.join(img_path, i+'_seg_6.nii')
    
#     ra_img_i_path = os.path.join(ra_path, 'IBSR_'+i+'_img_ra.nii')
#     ra_label_i_path = os.path.join(ra_path, 'IBSR_'+i+'_seg_ra.nii')
    
    re_img_i_path = os.path.join(re_path, i+'_img_re.nii')
    re_label_i_path = os.path.join(re_path, i+'_seg_6_re.nii')    

#     resample(img_i_path, ra_img_i_path)
#     resample(label_i_path, ra_label_i_path)
    
    
    ants_register(template_path, img_i_path, label_i_path,
                  re_img_i_path, re_label_i_path)