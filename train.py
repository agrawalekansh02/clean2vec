from ast import In
import numpy as np
import pickle as pkl
import tensorflow as tf
from data import *
from model import *


VOCAB_SIZE = 5500
INPUT_SHAPE = (27,VOCAB_SIZE)
OUTPUT_SHAPE = 50
HIDDEN_SIZE = 256
DROPOUT_RATE = 0.5

def train():
    metadata, idx_q, idx_a = load_data()
    (trainX, trainY), (testX, testY), (validX, validY) = split_dataset(idx_q, idx_a)

    _model = Seq2Seq(INPUT_SHAPE, OUTPUT_SHAPE, VOCAB_SIZE, HIDDEN_SIZE, DROPOUT_RATE)
    model = _model.build()
    model.compile(loss='categorical_crossentropy', 
        optimizer='adam', 
        metrics=['accuracy'])

    trainX1 = trainX.copy()
    trainX2 = trainY.copy()
    for i in range(trainY.shape[1]):
        np.roll(trainY[i,:], -1)
        trainY[i,-1] = 0

    trainX1 = tf.keras.utils.to_categorical(trainX1, VOCAB_SIZE)
    trainX2 = tf.keras.utils.to_categorical(trainX2, VOCAB_SIZE)
    trainY = tf.keras.utils.to_categorical(trainY, VOCAB_SIZE)

    model.fit([trainX1, trainX2], trainY, batch_size=64, epochs=100, validation_split=0.2)


if __name__ == '__main__':
    train()