import tensorflow as tf
from model import *
from train import *
from data import *

seeds = ["stain on blanket"]

def str_to_tokens(sentence, w2idx):
    words = sentence.lower().split()
    tokens = [w2idx.get(w, UNK) for w in words]
    seq = tf.keras.utils.pad_sequences([tokens], 
        maxlen=INPUT_LEN, 
        padding='post')
    return seq

def inference():
    _model = Seq2Seq(OUTPUT_SHAPE, VOCAB_SIZE, HIDDEN_SIZE, DROPOUT_RATE)
    model = _model.model
    model.load_weights('data/model.h5')
    metadata, idx_q, idx_a = load_data()
    w2idx = metadata['w2idx']

    for seed in seeds:
        seq = str_to_tokens(seed, w2idx)


if __name__ == "__main__":
    inference()