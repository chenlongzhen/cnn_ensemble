# coding: utf-8

import h5py
import sys,os
import functools
from PIL import Image as pil_image
import keras.backend as K

import logging
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import keras
import numpy as np
from keras.models import save_model
from sklearn.utils import shuffle
from keras.callbacks import TensorBoard,EarlyStopping,CSVLogger,ModelCheckpoint
import yaml


def myLog(logPath):
    '''
    logging
    :param logPath:  where to save log
    :return: logging handle
    '''

    logging.basicConfig(level=logging.INFO,
                        format='[ %(asctime)s %(filename)s line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename=logPath,
                        filemode='w')

    #定义一个StreamHandler，将INFO级别或更高的日志信息打印到标准错误，并将其添加到当前的日志处理对象#
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    return logging

def array_to_img(x, data_format='channels_last', scale=True):
    """Converts a 3D Numpy array to a PIL Image instance.
    # Arguments
        x: Input Numpy array.
        data_format: Image data format.
        scale: Whether to rescale image values
            to be within [0, 255].
    # Returns
        A PIL Image instance.
    # Raises
        ImportError: if PIL is not available.
        ValueError: if invalid `x` or `data_format` is passed.
    """
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    x = np.asarray(x, dtype=K.floatx())
    if x.ndim != 3:
        raise ValueError('Expected image array to have rank 3 (single image). '
                         'Got array with shape:', x.shape)

    if data_format is None:
        data_format = K.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Invalid data_format:', data_format)

    # Original Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but target PIL image has format (width, height, channel)
    if data_format == 'channels_first':
        x = x.transpose(1, 2, 0)
    if scale:
        x = x + max(-np.min(x), 0)
        x_max = np.max(x)
        if x_max != 0:
            x /= x_max
        x *= 255
    if x.shape[2] == 3:
        # RGB
        return pil_image.fromarray(x.astype('uint8'), 'RGB')
    elif x.shape[2] == 1:
        # grayscale
        return pil_image.fromarray(x[:, :, 0].astype('uint8'), 'L')
    else:
        raise ValueError('Unsupported channel number: ', x.shape[2])


#################################
# cofig
#################################
CNF = yaml.load(open('../conf/setting.yaml'))

version = CNF['version']

prefix = CNF['prefix']


genPath = prefix +CNF['gen_path']
if genPath is 'None':
    train_path = prefix + CNF['train_path']
    test_path = prefix + CNF['test_path']
else:
    train_path = test_path = 'None'

use_model = CNF['use_model']

gen_layer = CNF['gen_layer']

# board
board_path = prefix + '/'+CNF['board_path']
if not os.path.isdir(board_path): os.mkdir(board_path)

# train log
LOG_PATH = prefix + '/'+CNF['train_log']
if not os.path.isdir(LOG_PATH): os.mkdir(LOG_PATH)

# weight path
BEST_WEIGHT = prefix + '/'+CNF['weight_path']
if not os.path.isdir(BEST_WEIGHT): os.mkdir(BEST_WEIGHT)

# logger path
log_path = prefix + '/' + CNF['log_path']
if not os.path.isdir(log_path): os.mkdir(log_path)

# data shuffle seed
np.random.seed(int(CNF['seed']))

# logger
logger = myLog(log_path+"/log_{}".format(version))
logger.info(CNF)

# train settings
batch_size = int(CNF['batch_size'])
nb_epoch = int(CNF['nb_epoch'])
validation_split=float(CNF['validation_split'])

# fixed path
submissionPath =  "../data/output/submission.csv"
END_MODEL =  "../data/endModel/endModel_{}.h5".format(version)
model_path = '../data/model'
end_model_path = '../data/endModel'
if not os.path.isdir(model_path): os.mkdir(model_path)
if not os.path.isdir(end_model_path): os.mkdir(end_model_path)


#################################
# read features
#################################
gen_data = [] # generation from model , shape like train or test (,,,)

#for m in use_model:
#    filename =  "../data/model/gap_{}_{}.h5".format(m,version)
#    logger.info("[INFO] begin to read {}".format(os.path.basename(filename)))
#    with h5py.File(filename, 'r') as h:
#        X_train.append(np.array(h['train']))
#        X_test.append(np.array(h['test']))
#        y_train = np.array(h['label'])
#
#X_train = np.concatenate(X_train, axis=1)
#X_test = np.concatenate(X_test, axis=1)
#
#logger.info("train shape\r")
#logger.info(X_train.shape)
#logger.info("test shape\r")
#logger.info(X_test.shape)
#
#logger.info("[INFO] shuffle")
#X_train, y_train = shuffle(X_train, y_train)
#

# test

for m in use_model:
    filename =  "../data/model/gap_{}_{}.h5".format(m,version)
    with h5py.File(filename,'r') as h:
        data = np.array(h['gen'])
        print(data.shape)
        for i in range(data.shape[0]):
            for j in range(data[i].shape[2]):
                p2d = data[i][:,:,j]
                p3d = np.expand_dims(p2d,axis=2)
                print(p3d.shape)
                img = array_to_img(p3d)
                img = img.resize((224,224))
                img.save('../data/image/{}_{}_{}.jpg'.format(m,version,j))
