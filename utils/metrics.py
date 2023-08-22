import numpy as np
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion, \
    generate_binary_structure
import torch


class runningScore(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred):
        n_classes = self.n_classes
        mask = (label_true >= 0) & (label_true < n_classes)

        hist = np.bincount(n_classes * label_true[mask].astype(int) + label_pred[mask],
                           minlength=n_classes ** 2).reshape(n_classes, n_classes)
        return hist

    def update(self, label_trues, label_preds):
        self.confusion_matrix += self._fast_hist(label_trues.flatten(), label_preds.flatten())

    def _calc_iu(self):
        iu = np.zeros(self.n_classes)
        hist = self.confusion_matrix
        temp1 = np.diag(hist)
        temp2 = (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        for ind in range(temp2.size):
            if temp2[ind] == 0:
                iu[ind] = 0
            else:
                iu[ind] = temp1[ind] / temp2[ind]
        return iu

    def get_scores(self):
        #         hist = self.confusion_matrix
        iu = self._calc_iu()
        dice = np.divide(np.multiply(iu, 2), np.add(iu, 1))
        mean_iu = np.nanmean(iu)
        mean_dice = (mean_iu * 2) / (mean_iu + 1)
        return {'Dice': dice, 'Mean Dice': mean_dice, }

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


def dc(result, reference):
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))

    intersection = np.count_nonzero(result & reference)

    size_i1 = np.count_nonzero(result)
    size_i2 = np.count_nonzero(reference)

    try:
        dc = 2. * intersection / float(size_i1 + size_i2)
    except ZeroDivisionError:
        dc = 0.0

    return np.round(dc, decimals=4)
#     return (intersection, size_i1, size_i2)


def mhd(result, reference):
    hd1 = __surface_distances(result, reference).mean()
    hd2 = __surface_distances(reference, result).mean()
    hd = max(hd1, hd2)
    return np.round(hd, decimals=4)


def __surface_distances(result, reference):
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))

    # binary structure
    footprint = generate_binary_structure(result.ndim, 1)

    # test for emptiness
    if 0 == np.count_nonzero(result):
        sds = np.nan
    elif 0 == np.count_nonzero(reference):
        sds = np.nan
    else:
            # extract only 1-pixel border line of objects
        result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
        reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)

        # compute average surface distance
        # Note: scipys distance transform is calculated only inside the borders of the
        #       foreground objects, therefore the input has to be reversed
        dt = distance_transform_edt(~reference_border, sampling=None)
        sds = dt[result_border]

    return sds


def iu(result, reference):
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))    
    intersection = np.count_nonzero(result & reference)
    
    union = np.count_nonzero(result | reference)
    
    try:
        iou = 1.0 * intersection / (1.0 * union)
    except ZeroDivisionError:
        iou = 0.0    
    return np.round(iou, decimals=4)


def metrics(pred, label, num_class):
    
    l, r, c = pred.shape
    pred_n = torch.zeros([num_class, l, r, c]).scatter_(0, pred.unsqueeze(0), 1)
    label_n = torch.zeros([num_class, l, r, c]).scatter_(0, label.unsqueeze(0), 1)
    pred_n = pred_n.numpy()
    label_n = label_n.numpy()
    iou = []
    dice = []
    # hausdorff_distance = []
    for i in range(1, num_class):
        dice.append(dc(pred_n[i, :, :, :], label_n[i, :, :, :]))
        iou.append(iu(pred_n[i, :, :, :], label_n[i, :, :, :]))
        # hausdorff_distance.append(mhd(pred_n[i, :, :, :], label_n[i, :, :, :]))
    mean_iou = sum(iou)/(num_class - 1)
    mean_dice = np.round(2 * mean_iou / (1 + mean_iou), decimals=4)    # score = {'Mean Dice': mean_dice, 'Dice': dice, 'MHD': hausdorff_distance}
    score = {'Mean Dice': mean_dice, 'Dice': dice}
    return score

def DiceScore(pred, label, num_class):
    
    l, r, c = pred.shape
    pred_n = torch.zeros([num_class, l, r, c]).scatter_(0, pred.unsqueeze(0), 1)
    label_n = torch.zeros([num_class, l, r, c]).scatter_(0, label.unsqueeze(0), 1)
    pred_n = pred_n.numpy()
    label_n = label_n.numpy()
    iou = []
    dice = []
    for i in range(1, num_class):
        dice.append(dc(pred_n[i, :, :, :], label_n[i, :, :, :]))
        iou.append(iu(pred_n[i, :, :, :], label_n[i, :, :, :]))
    mean_iou = sum(iou)/(num_class - 1)
    mean_dice = np.round(2 * mean_iou / (1 + mean_iou), decimals=4)
    score = {'Mean Dice': mean_dice, 'Dice': dice}
    return score


def MHDValue(pred, label, num_class):
    l, r, c = pred.shape
    pred_n = torch.zeros([num_class, l, r, c]).scatter_(0, pred.unsqueeze(0), 1)
    label_n = torch.zeros([num_class, l, r, c]).scatter_(0, label.unsqueeze(0), 1)
    pred_n = pred_n.numpy()
    label_n = label_n.numpy()
    hausdorff_distance = []
    for i in range(1, num_class):
        hausdorff_distance.append(mhd(pred_n[i, :, :, :], label_n[i, :, :, :]))
    score = {'MHD': hausdorff_distance}
    return score
