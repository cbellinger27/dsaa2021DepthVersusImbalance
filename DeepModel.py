# %%
import tensorflow as tf
import pandas as pd
import numpy as np 
from functools import reduce
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import time

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

METRICS = [
      tf.keras.metrics.TruePositives(name='tp'),
      tf.keras.metrics.FalsePositives(name='fp'),
      tf.keras.metrics.TrueNegatives(name='tn'),
      tf.keras.metrics.FalseNegatives(name='fn'), 
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),
      tf.keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]

#The VGG-16-like model
def get_modelCnnSmall(inputDim, outputDim, depth=1, hidden=10, reg=[], dropout=[], modelName='best_model'):
    # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=1e-5)
    # mc = tf.keras.callbacks.ModelCheckpoint(modelName+".h5", monitor='val_loss', mode='min', save_best_only=True)
    # tb = TensorBoard(log_dir="log_"+modelName+".log")
    # kernel_regularizer='l1','l1_l2', 'l2'
    inp = tf.keras.Input(inputDim)
    x = tf.keras.layers.Conv2D(filters=8,kernel_size=(3,3),padding="same", activation='relu')(inp)
    x = tf.keras.layers.Conv2D(filters=8,kernel_size=(3,3),padding="same", kernel_initializer='he_uniform', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)
    if depth > 1:
        if 2 in reg:
            x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), padding="same", kernel_regularizer =tf.keras.regularizers.l2( l=0.01), activation='relu')(x)
            x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), padding="same", kernel_regularizer =tf.keras.regularizers.l2( l=0.01), activation='relu')(x)
            x = tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)
        else:
            x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), padding="same", activation='relu')(x)
            x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), padding="same", activation='relu')(x)
            x = tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)
        if 2 in dropout:
            x = tf.keras.layers.Dropout(rate=0.2)(x)
    if depth > 2:
        if 3 in reg:
            x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding="same", kernel_regularizer =tf.keras.regularizers.l2( l=0.01), activation='relu')(x)
            x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding="same", kernel_regularizer =tf.keras.regularizers.l2( l=0.01), activation='relu')(x)
            x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding="same", kernel_regularizer =tf.keras.regularizers.l2( l=0.01), activation='relu')(x)
            x = tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)
        else: 
            x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding="same", activation='relu')(x)
            x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding="same", activation='relu')(x)
            x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding="same", activation='relu')(x)
            x = tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)
        if 4 in dropout:
            x = tf.keras.layers.Dropout(rate=0.2)(x)
    if depth > 3:
        if 4 in reg:
            x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", kernel_regularizer =tf.keras.regularizers.l2( l=0.01), activation='relu')(x)
            x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", kernel_regularizer =tf.keras.regularizers.l2( l=0.01), activation='relu')(x)
            x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", kernel_regularizer =tf.keras.regularizers.l2( l=0.01), activation='relu')(x)
            x = tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)
        else:
            x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", activation='relu')(x)
            x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", activation='relu')(x)
            x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", activation='relu')(x)
            x = tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)
        if 4 in dropout:
            x = tf.keras.layers.Dropout(rate=0.2)(x)
    if depth > 4:
        if 5 in reg:
            x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", kernel_regularizer =tf.keras.regularizers.l2( l=0.01), activation='relu')(x)
            x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", kernel_regularizer =tf.keras.regularizers.l2( l=0.01), activation='relu')(x)
            x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", kernel_regularizer =tf.keras.regularizers.l2( l=0.01), activation='relu')(x)
            x = tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)
        else:
            x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", activation='relu')(x)
            x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", activation='relu')(x)
            x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", activation='relu')(x)
            x = tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)
        if 5 in dropout:
            x = tf.keras.layers.Dropout(rate=0.2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(hidden, activation='relu')(x)
    if 6 in dropout:
        x = tf.keras.layers.Dropout(rate=0.2)(x)
    x = tf.keras.layers.Dense(hidden, activation='relu')(x)
    if 7 in dropout:
        x = tf.keras.layers.Dropout(rate=0.2)(x)
    out = tf.keras.layers.Dense(outputDim, activation='softmax')(x)
    model = tf.keras.Model(inputs=inp, outputs=out)
    model.compile(loss='binary_crossentropy', metrics=METRICS, optimizer=tf.keras.optimizers.Adam())
    return model


def get_modelFullyCon(inputDim, outputDim, depth=1, hidden=4960, modelName='best_model'):
    # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=1e-5)
    # mc = tf.keras.callbacks.ModelCheckpoint(modelName+".h5", monitor='val_loss', mode='min', save_best_only=True)
    # tb = TensorBoard(log_dir="log_"+modelName+".log")
    inp = tf.keras.Input(inputDim,name='input')
    x = tf.keras.layers.Flatten(name='flatten')(inp)
    x = tf.keras.layers.Dense(hidden, activation='relu', name='layer1')(x)
    # x = tf.keras.layers.Dropout(rate=0.5)(x)
    for i in range(depth-1):
        x = tf.keras.layers.Dense(hidden, activation='relu', name='layer'+str(i+2))(x)
        # x = tf.keras.layers.Dropout(rate=0.5)(x)
    out = tf.keras.layers.Dense(outputDim, activation='softmax', name='output')(x)
    model = tf.keras.Model(inputs=inp, outputs=out)
    model.compile(loss='binary_crossentropy', metrics=METRICS, optimizer=tf.keras.optimizers.Adam())
    return model

