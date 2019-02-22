from __future__ import print_function
import argparse
import keras
from keras import backend as K
from keras.datasets import mnist
from keras.models import Sequential, save_mxnet_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_channels_first
import mxnet as mx
import numpy as np
import os
from pathlib import Path
import shutil


##############################
##### INFERENCE FUNCTIONS
##############################

def model_fn(model_dir):
    net = mx.gluon.SymbolBlock.imports(
        str(Path(model_dir, 'mnist_cnn-symbol.json')),
        ['/conv2d_1_input1'],
        str(Path(model_dir, 'mnist_cnn-0000.params'))
    )
    return net

# def input_fn()

def predict_fn(input_object, model):
    data_batch = input_object.expand_dims(0)
    return model(data_batch)

# def output_fn()


##############################
##### TRAINING FUNCTIONS
##############################


def load_data():
    # load data
    (x_train_nhw, y_train), (x_test_nhw, y_test) = mnist.load_data()
    # preprocess x
    x_train_nhwc = x_train_nhw[..., np.newaxis]
    x_train = to_channels_first(x_train_nhwc)
    x_test_nhwc = x_test_nhw[..., np.newaxis]
    x_test = to_channels_first(x_test_nhwc)
    x_train = x_train.astype('float32')/255
    x_test = x_test.astype('float32')/255
    # preprocess y
    num_classes = 10
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return (x_train, y_train), (x_test, y_test)
    
    
def create_model(input_shape=(1,28,28), num_classes=10):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def save_model(model, model_dir):
    prefix = 'mnist_cnn'
    params_filename = '{}-0000.params'.format(prefix)
    symbol_filename = '{}-symbol.json'.format(prefix)
    data_names, data_shapes = save_mxnet_model(model=model, prefix=prefix, epoch=0)
    shutil.copy(src=str(Path('.', params_filename)), dst=str(Path(model_dir, params_filename)))
    shutil.copy(src=str(Path('.', symbol_filename)), dst=str(Path(model_dir, symbol_filename)))
    
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    args, _ = parser.parse_known_args()
    return args
    
    
def train_model():
    args = parse_args()
    # set to channels_last
    K.set_image_data_format('channels_first')
    # data
    (x_train, y_train), (x_test, y_test) = load_data()
    # model
    model = create_model()
    # training
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    model.fit(x_train, y_train,
              batch_size=args.batch_size,
              epochs=args.epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    # saving
    save_model(model, args.model_dir)
    
    
if __name__ == "__main__":
    train_model()
    
