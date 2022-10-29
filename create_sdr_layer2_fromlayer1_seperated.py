from audioop import add
import myutils_htm
import time
import numpy as np
import tqdm
import copy
# from multiprocessing import Pool, Manager, freeze_support
import math
import os
import bitarray
import re

print("PYTHONHASHSEED : ", os.environ['PYTHONHASHSEED'])
MAX_BUCKET = 4096
SENTENCE_SIZE = 2
timestr = time.strftime("%Y%m%d-%H%M%S")
padding = "#####"

def readlines(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return f.readlines()
def createSentenseSDR_concat(sentense, max_bucket):
    text_representation = None
    for word in sentense.split(' '):
        if text_representation is None:
            text_representation = bitarray.bitarray(myutils_htm.text2Representation1(word, max_bucket).tolist())
            continue
        text_representation = text_representation + bitarray.bitarray(myutils_htm.text2Representation1(word, max_bucket).tolist())
    return text_representation
def createSentenseSDR_concat_fromlayer1(train_sentences, max_bucket):
    text_representation = None
    for train_sentence in train_sentences:
        if text_representation is None:
            text_representation = createSentenseSDR_concat(train_sentence, max_bucket)
            continue
        text_representation = text_representation | createSentenseSDR_concat(train_sentence, max_bucket)
    return text_representation


def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def create_sdr_layer2(full_degrade5_ground_a_train80, max_bucket, sentence_sizes):
    
    eval_sdr_layer1_merge = myutils_htm.loadpkl('../sdr_layer1_' + str(max_bucket) + '_0.2_0.2_20000/eval_sdr_layer1_remerge_' + str(max_bucket) + '_0.2_0.2_20000.pkl')
    # eval_sdr_layer1_merge = myutils_htm.loadpkl('eval_sdr_layer1_remerge_4096_0.2_0.2_20000.pkl')
    training_words_layer1 = eval_sdr_layer1_merge['training_words_layer1']
    training_words_dict = eval_sdr_layer1_merge['training_words_dict']
    del eval_sdr_layer1_merge
    training_words_dict[str(len(training_words_layer1))] = set([padding,])
    training_words_layer1.append(myutils_htm.text2Representation1(padding, max_bucket))
    training_words_layer1 = np.array(training_words_layer1)
    training_words_layer1_length = training_words_layer1.sum(axis=1)
    training_words_layer1_dict = {word : stri  for stri in training_words_dict.keys() for word in training_words_dict[stri]}
    training_words = sorted([word for stri in training_words_dict.keys() for word in training_words_dict[stri]])
    print("Training Layer1 size:", len(training_words_layer1))
    print("Training Vocab Size:", len(training_words))

    for sentence_size in sentence_sizes:
        print(max_bucket, sentence_size)
        sentence_layer_dict = {}
        for i, train_sentence in tqdm.tqdm(enumerate(full_degrade5_ground_a_train80)):
            train_sentence = train_sentence.strip()
            if train_sentence == "" : continue
            train_words = train_sentence.split(' ')
            if len(train_words)<sentence_size:
                [train_words.append(padding) for k in range(sentence_size - len(train_words))]
            
            for j in range(len(train_words) - sentence_size + 1):
                words = train_words[j:j + sentence_size]
                ssentence = ' '.join(words)
                layer1sentence = '#'.join([training_words_layer1_dict[w] for w in words])
                if layer1sentence not in sentence_layer_dict:
                    sentence_layer_dict[layer1sentence] = set()
                sentence_layer_dict[layer1sentence].add(ssentence)

        print(sentence_size, "training sentences layer 2 size : ", len(sentence_layer_dict.keys()))

        train_sentences = sorted(list(sentence_layer_dict.keys()))
        layers = [createSentenseSDR_concat_fromlayer1(sentence_layer_dict[index], MAX_BUCKET) for i, index in tqdm.tqdm(enumerate(train_sentences))]
        layer_dict = {str(i) : sentence_layer_dict[index] for i, index in tqdm.tqdm(enumerate(train_sentences))}

        print(sentence_size, max_bucket, len(layers))
        myutils_htm.savepkl('training_fromsentence_layer2_seperated_' + str(sentence_size) + '_'  + str(max_bucket) + '_' + str(MAX_BUCKET) + '.pkl', {
            'training_sentence_layer2': layers,
            'training_sentence_layer2_dict' : layer_dict
        })

# full_degrade5_ground_a_test80 = myutils_htm.readlines('../full_degrade5_ground_a_test80.txt')
full_degrade5_ground_a_train80 = myutils_htm.readlines('../full_degrade5_ground_a_train80.txt')
# full_degrade5_input_a_test80 = myutils_htm.readlines('../full_degrade5_input_a_test80.txt')
# full_degrade5_input_a_train80 = myutils_htm.readlines('../full_degrade5_input_a_train80.txt')
print("Training sentence line :", len(full_degrade5_ground_a_train80))

max_buckets = [4096]
sentence_sizes = [2, 3]
for max_bucket in max_buckets:
    create_sdr_layer2(full_degrade5_ground_a_train80, max_bucket, sentence_sizes)