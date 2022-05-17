import numpy as np
import pickle as pkl
from random import sample
import nltk
import itertools

EN_WHITELIST = '0123456789abcdefghijklmnopqrstuvwxyz ' # space is included in whitelist
EN_BLACKLIST = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\''
UNK = 'unk'
VOCAB_SIZE = 8000

limit = {
        'maxq' : 25,
        'minq' : 2,
        'maxa' : 25,
        'mina' : 2
        }

# create a dictionary with (key = line_id, value = text)
def get_id2line():
    lines = open('data/data_clean.txt', encoding='utf-8', errors='ignore').read().split('\n')
    id2line = {}
    uid = 0
    for line in lines:
        id2line[uid] = line
        uid += 1
    return id2line

# create conversaton sequences with list of [list of line ids]
def get_conversations():
    conv_lines = open('data/data_clean.txt', encoding='utf-8', errors='ignore').read().split('\n')
    convs = []
    lines = [line for line in conv_lines]
    for line_num in range(len(lines)-1):
        convs.append([line_num, line_num+1])
    return convs 

# get each line from conversation and save it t
def gather_dataset(convs, id2line):
    questions, answers = [], []
    for q, a in convs:
        questions.append(id2line[q])
        answers.append(id2line[a])
    return questions, answers

# filter out unecessary characters
def filter_line(line, whitelist):
    return ''.join([ch for ch in line if ch in whitelist])

# remove too long or too short sequences
def filter_data(qseq, aseq):
    filtered_q, filtered_a = [], []
    raw_data_len = len(qseq)

    assert len(qseq) == len(aseq)

    for i in range(raw_data_len):
        qlen, alen = len(qseq[i].split(' ')), len(aseq[i].split(' '))
        if qlen >= limit['minq'] and qlen <= limit['maxq']:
            if alen >= limit['mina'] and alen <= limit['maxa']:
                filtered_q.append(qseq[i])
                filtered_a.append(aseq[i])

    # print the fraction of the original data, filtered
    filt_data_len = len(filtered_q)
    filtered = int((raw_data_len - filt_data_len)*100/raw_data_len)
    print(str(filtered) + '% filtered from original data')

    return filtered_q, filtered_a

# create id to word, word to id, frequency dictionaries
def index_(tokenized_sentences, vocab_size):
    # get frequency distribution
    freq_dist = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    # get vocabulary of 'vocab_size' most used words
    vocab = freq_dist.most_common(vocab_size)
    # index2word
    index2word = ['_'] + [UNK] + [ x[0] for x in vocab ]
    # word2index
    word2index = dict([(w,i) for i,w in enumerate(index2word)] )
    return index2word, word2index, freq_dist

# remove unkown words
def filter_unk(qtokenized, atokenized, w2idx):
    data_len = len(qtokenized)

    filtered_q, filtered_a = [], []

    for qline, aline in zip(qtokenized, atokenized):
        unk_count_q = len([ w for w in qline if w not in w2idx ])
        unk_count_a = len([ w for w in aline if w not in w2idx ])
        if unk_count_a <= 2:
            if unk_count_q > 0:
                if unk_count_q/len(qline) > 0.2:
                    pass
            filtered_q.append(qline)
            filtered_a.append(aline)

    # print the fraction of the original data, filtered
    filt_data_len = len(filtered_q)
    filtered = int((data_len - filt_data_len)*100/data_len)
    print(str(filtered) + '% filtered from original data')

    return filtered_q, filtered_a

# sequences to array of indeces and zero pad
def zero_pad(qtokenized, atokenized, w2idx):
    # num of rows
    data_len = len(qtokenized)

    # numpy arrays to store indices
    idx_q = np.zeros([data_len, limit['maxq']], dtype=np.int32)
    idx_a = np.zeros([data_len, limit['maxa']], dtype=np.int32)

    for i in range(data_len):
        q_indices = pad_seq(qtokenized[i], w2idx, limit['maxq'])
        a_indices = pad_seq(atokenized[i], w2idx, limit['maxa'])

        idx_q[i] = np.array(q_indices)
        idx_a[i] = np.array(a_indices)

    return idx_q, idx_a

# replace with index or zero if unknown
def pad_seq(seq, lookup, maxlen):
    indices = []
    for word in seq:
        if word in lookup:
            indices.append(lookup[word])
        else:
            indices.append(lookup[UNK])
    return indices + [0]*(maxlen - len(seq))


def main():
    id2line = get_id2line()
    print('>> gathered id2line dictionary.\n')

    convs = get_conversations()
    print(convs[121:125])
    print('>> gathered conversations.\n')
    questions, answers = gather_dataset(convs, id2line)
    print(questions[0])

    # change to lower case (just for en)
    questions = [line.lower() for line in questions]
    answers = [line.lower() for line in answers]

    # filter out unnecessary characters
    print('\n>> filter unecessary characters')
    questions = [filter_line(line, EN_WHITELIST) for line in questions]
    answers = [filter_line(line, EN_WHITELIST) for line in answers]

    # filter out too long or too short sequences
    print('\n>> 2nd layer of filtering')
    qlines, alines = filter_data(questions, answers)

    for q,a in zip(qlines[141:145], alines[141:145]):
        print('q : [{0}]; a : [{1}]'.format(q,a))

    # convert list of [lines of text] into list of [list of words]
    print('\n>> segment lines into words')
    qtokenized = [[w.strip() for w in wordlist.split(' ') if w] for wordlist in qlines]
    atokenized = [[w.strip() for w in wordlist.split(' ') if w] for wordlist in alines]
    print('\n:: sample from segmented list of words')

    for q,a in zip(qtokenized[141:145], atokenized[141:145]):
        print('q : [{0}]; a : [{1}]'.format(q,a))

    # indexing -> idx2w, w2idx
    print('\n >> index words')
    idx2w, w2idx, freq_dist = index_( qtokenized + atokenized, vocab_size=VOCAB_SIZE)

    # filter out sentences with too many unknowns
    print('\n >> filter Unknowns')
    qtokenized, atokenized = filter_unk(qtokenized, atokenized, w2idx)
    print('\n Final dataset len : ' + str(len(qtokenized)))


    print('\n >> zero padding')
    idx_q, idx_a = zero_pad(qtokenized, atokenized, w2idx)

    print('\n >> save numpy arrays to disk')
    # save them
    np.save('data/idx_q.npy', idx_q)
    np.save('data/idx_a.npy', idx_a)

    # let us now save the necessary dictionaries
    metadata = {
            'w2idx' : w2idx,
            'idx2w' : idx2w,
            'limit' : limit,
            'freq_dist' : freq_dist
                }

    # write to disk : data control dictionaries
    with open('data/metadata.pkl', 'wb') as f:
        pkl.dump(metadata, f)

    # count of unknowns
    unk_count = (idx_q == 1).sum() + (idx_a == 1).sum()
    # count of words
    word_count = (idx_q > 1).sum() + (idx_a > 1).sum()

    print('% unknown : {0}'.format(100 * (unk_count/word_count)))
    print('Dataset count : ' + str(idx_q.shape[0]))


def split_dataset(x, y, ratio = [0.7, 0.15, 0.15] ):
    # number of examples
    data_len = len(x)
    lens = [ int(data_len*item) for item in ratio ]

    trainX, trainY = x[:lens[0]], y[:lens[0]]
    testX, testY = x[lens[0]:lens[0]+lens[1]], y[lens[0]:lens[0]+lens[1]]
    validX, validY = x[-lens[-1]:], y[-lens[-1]:]

    return (trainX,trainY), (testX,testY), (validX,validY)



def batch_gen(x, y, batch_size):
    # infinite while
    while True:
        for i in range(0, len(x), batch_size):
            if (i+1)*batch_size < len(x):
                yield x[i : (i+1)*batch_size ].T, y[i : (i+1)*batch_size ].T

def rand_batch_gen(x, y, batch_size):
    while True:
        sample_idx = sample(list(np.arange(len(x))), batch_size)
        yield x[sample_idx].T, y[sample_idx].T

def decode(sequence, lookup, separator=''): # 0 used for padding, is ignored
    return separator.join([ lookup[element] for element in sequence if element ])

def load_data(PATH=''):
    # read data control dictionaries
    with open(PATH + 'data/metadata.pkl', 'rb') as f:
        metadata = pkl.load(f)
    # read numpy arrays
    idx_q = np.load(PATH + 'data/idx_q.npy')
    idx_a = np.load(PATH + 'data/idx_a.npy')
    return metadata, idx_q, idx_a




if __name__ == "__main__":
    main()