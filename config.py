from yacs.config import CfgNode as CN

_C = CN()
#IBSR----MALC
# _C.SOURCE = CN()
# _C.SOURCE.dataset = 'IBSR'
# _C.SOURCE.PATH = '/home/huqian/baby/DA_code/IBSR_18/IBSR_18_re'
# _C.SOURCE.label_s = (9, 10, 11, 12, 13, 17, 18, 48, 49, 50, 51, 52, 53, 54)
# _C.SOURCE.label_t = (1, 1, 2, 3, 4, 5, 6, 1, 1, 2, 3, 4, 5, 6)
# _C.SOURCE.IDs_train = ['08', '09', '02', '07', '04', '05', '16', '03', '06']

# _C.TARGET = CN()
# _C.TARGET.dataset = 'MALC'
# _C.TARGET.PATH = '/home/huqian/baby/DA_code/MICCAI/MALC_re'
# _C.TARGET.label_s = (59, 60, 36, 37, 57, 58, 55, 56, 47, 48, 31, 32)
# _C.TARGET.label_t = (1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6)
# _C.TARGET.IDs_train = ['20', '28', '08', '31', '06', '35', '34', '25', '13', '05', '01', '21',
#                     '17', '27', '33', '11', '12', '16', '10', '32', '18', '04', '14', '02',
#                     '22', '09', '19']
# _C.TARGET.IDs_eval = ['29', '03', '26', '23']
# _C.TARGET.IDs_test = ['07', '15', '24', '30']  

#MALC----IBSR
_C.SOURCE = CN()
_C.SOURCE.dataset = 'MALC'
_C.SOURCE.PATH = '/home/data/hq/DA//MICCAI/MALC_re'
_C.SOURCE.label_s = (59, 60, 36, 37, 57, 58, 55, 56, 47, 48, 31, 32)
_C.SOURCE.label_t = (1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6)
_C.SOURCE.IDs_train = ['20', '28', '08', '31', '06', '35', '34', '25', '13', '05', '01', '21',
                     '17', '27', '33', '11', '12', '16', '10', '32', '18', '04', '14', '02',
                     '22', '09', '19', '23', '30']

_C.TARGET = CN()
_C.TARGET.dataset = 'IBSR'
_C.TARGET.PATH = '/home/data/hq/DA/IBSR_18/IBSR_18_re'
_C.TARGET.label_s = (9, 10, 11, 12, 13, 17, 18, 48, 49, 50, 51, 52, 53, 54)
_C.TARGET.label_t = (1, 1, 2, 3, 4, 5, 6, 1, 1, 2, 3, 4, 5, 6)
_C.TARGET.IDs_train = ['08', '09', '02', '07', '04', '05', '16', '03', '06', '01', '17', '15']
_C.TARGET.IDs_eval = ['10', '11', '12','13', '14', '18']
_C.TARGET.IDs_test = ['13', '14', '18']  
def get_cfg_defaults():
    return _C.clone()
