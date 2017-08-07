
# coding: utf-8

# In[1]:

import h5py
import sys,os
import functools

import logging
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import keras
import numpy as np
from keras.models import save_model
from sklearn.utils import shuffle
from keras.callbacks import TensorBoard,EarlyStopping,CSVLogger,ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation,Conv1D
from keras.layers.normalization import BatchNormalization
from keras import layers

import yaml

def get_session(allow_growth=True):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    num_threads = os.environ.get('OMP_NUM_THREADS')
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    gpu_options = tf.GPUOptions(allow_growth=True)


    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


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


#################################
# cofig
#################################
CNF = yaml.load(open('../conf/setting.yaml'))

version = CNF['version']

prefix = CNF['prefix']

test_path = CNF['test_path']

use_model = CNF['use_model']

gen_layer = CNF['gen_layer']

num_classes = CNF['num_classes']

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

# gpu setting
allow_growth = bool(CNF['allow_growth'])
KTF.set_session(get_session(allow_growth=allow_growth))

# topk_acc
topK_acc = CNF['topK_acc']

#################################
# read features
#################################

X_train = []
X_test = []

for m in use_model:
    filename =  "../data/model/gap_{}_{}.h5".format(m,version)
    logger.info("[INFO] begin to read {}".format(os.path.basename(filename)))
    with h5py.File(filename, 'r') as h:
        X_train.append(np.array(h['train']))
        X_test.append(np.array(h['test']))
        y_train = np.array(h['label'])

X_train = np.concatenate(X_train, axis=1)
X_test = np.concatenate(X_test, axis=1)

logger.info("train shape\r")
logger.info(X_train.shape)
logger.info("test shape\r")
logger.info(X_test.shape)

logger.info("[INFO] shuffle")
X_train, y_train = shuffle(X_train, y_train)

y_train = keras.utils.to_categorical(y_train, num_classes=num_classes)
print y_train
#################################
# train pipe
#################################

from keras.models import *
from keras.layers import *

logger.info("[INFO] train")
print(X_train.shape)

#model = Sequential()
#model.add(Dropout(0.5),input_dim=X_train.shape[1])
#model.add(Dense(num_classes,activation='sigmoid'))
#model.add(Dense(num_classes,activation='sigmoid',input_dim=X_train.shape[1]))
input_tensor = Input(shape=(X_train.shape[1],))
x = input_tensor

x = Dropout(0.50)(x)
x = BatchNormalization()(x)
#x = Dense(1024)(x)
#x = Activation('relu')(x)
#x = Dropout(0.50)(x)
#x = BatchNormalization()(x)
x = Dense(num_classes)(x)
x = Activation('softmax')(x)

model = Model(input_tensor, x)
model.summary()

topK_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=int(topK_acc))

topK_acc.__name__ = 'topK_acc'

adam=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
adadelta=keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
model.compile(optimizer=adadelta,
              loss='categorical_crossentropy',
              metrics=['accuracy',topK_acc])


##############################
# Call Back
##############################

# tensor board
tb = TensorBoard(log_dir=board_path+'/{}/'.format(version), histogram_freq=1,  
                  write_graph=True, write_images=False)
#* tensorboard --logdir path_to_current_dir/Graph --port 8080 
print("tensorboard --logdir {} --port 8080".format(board_path))


# earlystoping
# ES = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')

# csv log
csvlog = CSVLogger(LOG_PATH+"/{}.log".format(version),separator=',', append=True)

# saves the model weights after each epoch if the validation loss decreased
checkpointer = ModelCheckpoint(filepath=BEST_WEIGHT+"/{}.h5".format(version),verbose=1, save_best_only=True)

##############################
# fit
##############################
model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epoch, validation_split=validation_split,
        callbacks = [tb,csvlog,checkpointer])

logger.info("[INFO] save model")
save_model(model,END_MODEL)


##yaml test
#with open('../data/model.yaml','w') as ifile:
#    ifile.write(model.to_yaml())

