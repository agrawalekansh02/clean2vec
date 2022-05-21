import tensorflow as tf
import numpy as np
from model import *
from train import *
from data import *

seeds = ["blanket is dirty and is not clean need help",
        "I want to clean fabric that has grass and mud stains"]

def str_to_tokens(sentence, w2idx, idx2w):
    words = sentence.lower().split()
    unk_idx = w2idx["<UNK>"]
    tokens = [w2idx.get(w, unk_idx) for w in words]
    seq = tf.keras.preprocessing.sequence.pad_sequences([tokens], 
        maxlen=INPUT_LEN, 
        padding='post')
    return seq

def inference_runner():
    _model = Seq2Seq(OUTPUT_SHAPE, VOCAB_SIZE, HIDDEN_SIZE, DROPOUT_RATE)
    model = _model.model
    model.load_weights('data/model.h5')
    decoder, encoder = _model.inference()
    metadata, idx_q, idx_a = load_data()
    w2idx = metadata['w2idx']
    idx2w = metadata['idx2w']

    for seed in seeds:
        seq = inference(seed, w2idx, idx2w, decoder, encoder)
        print(seq)

def inference(seed, w2idx, idx2w, decoder, encoder):
    seq = str_to_tokens(seed, w2idx, idx2w)
    states_values = encoder.predict(seq)
    empty_target = np.zeros((1, 1))
    empty_target[0,0] = w2idx['<SOS>']
    stop = False
    response = ''
    while not stop:
        decoder_output, h, c = decoder.predict([empty_target] + states_values)
        argmax = np.argmax(decoder_output[0, -1, :])
        word = idx2w[argmax]
        response += f" {word}"
        if word == "<EOS>" or len(response.split()) > 27:
            stop = True

        # prepare next iteration
        empty_target = np.zeros((1, 1))
        empty_target[0, 0] = argmax
        states_values = [h, c]

    return response



if __name__ == "__main__":
    inference_runner()
