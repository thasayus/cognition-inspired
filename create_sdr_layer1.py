
import myutils_htm
import time
import numpy as np
import tqdm
import copy
from multiprocessing import Pool, Manager, freeze_support
import math
import bitarray
MAX_BUCKET = 4096
timestr = time.strftime("%Y%m%d-%H%M%S")

import re
pattern_word = re.compile("^([a-zA-Z]+)$")
def removespchar(x, r = '?'):
  return re.sub(r'[^a-zA-Z]', r, x)
def searchWord(x):
    if len(x)>3 and pattern_word.search(x):
        return True
    else:
        return False
def readlines(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return f.readlines()


def merge(layers, layer_dict):

    replace = 1
    count = 0
    while replace > 0:
        replace = 0
        i = 0
        while i < len(layers):
            replaced = -1
            stri = str(i)
            layer = layers[i]
            layers_length = layers.sum(axis=1)
            allR = np.logical_and(layer, layers)
            allsum = allR.sum(axis=1)
            allsum = (allsum * 2)/((layers_length + sum(layer)))
            rlen = 30
            if len(allsum)<rlen:
                rlen = len(allsum)
            indexes = np.argpartition(-allsum, range(rlen))
            indexes = [id for id in indexes[:rlen] if id != i and layers_length[id]<MAX_BUCKET*0.05 and allsum[id]>0.5]
            
            if len(indexes)>0:
                index = indexes[0]
                strindex = str(index)
                # index_value = allsum[index[0]]
                layers[i] = np.logical_or(layer, layers[index])
                # print(sum(layer), sum(layers[index]))
                layer_dict[stri].update(layer_dict[strindex])

                layers = np.delete(layers, index, axis=0)
                
                templayer_dict = {'none' if int(strii) == index else (strii if int(strii) < index else str(int(strii) - 1)) :layer_dict[strii] for strii in sorted(layer_dict.keys())}
                if 'none' in templayer_dict.keys(): templayer_dict.pop('none')
                layer_dict = templayer_dict

                # newlayer_dict[str(index[0] + replace)].append(layer_dict[stri])
                # del newlayer_dict[str(i - replace)]
                # newlayer_dict.pop(stri)
                replaced = index
                replace = replace + 1
            
            if replaced < 0:
                i = i + 1
            else:
                if replaced > i :
                    i = i + 1

            if i%1000==0:
                print("run", i, replace, time.strftime("%Y%m%d-%H%M%S"))

        # myutils_htm.savepkl('eval_sdr_layer1_sample_' + str(count) + '.pkl',  {
        #     "training_words_layer1": new_layers,
        #     "training_words_dict" : newlayer_dict
        # })
        count = count + 1

    print("lenght merge layers : ", len(layers))
    myutils_htm.savepkl('eval_sdr_layer1_sample_' + str(MAX_BUCKET) + '.pkl',  {
        "training_words_layer1": layers,
        "training_words_dict" : layer_dict
    })

    return layers, layer_dict

sum_layer1_dict = {}
sum_layer1 = []
sum_layer1_length = 0
if __name__ == '__main__':

    train_test_set = myutils_htm.loadpkl('train_test_a.pkl')
    training_words = train_test_set['train_ground_words']
    del train_test_set

    training_words = training_words
    training_vocabs = sorted(list(set(training_words)))
    print("Training Words : ", len(training_words))
    print("Training Vocabs : ", len(training_vocabs))

    training_words_dict = {str(i) : set([word,]) for i, word in enumerate(training_vocabs)}
    training_words_layer1 = [bitarray.bitarray(myutils_htm.text2Representation1(word, MAX_BUCKET).tolist()) for i, word in tqdm.tqdm(enumerate(training_vocabs))]
    print("lenght layers : ", len(training_words_layer1))

    myutils_htm.savepkl('eval_sdr_layer1_' + str(MAX_BUCKET) + '.pkl',  {
        "training_words_layer1": training_words_layer1,
        "training_words_dict" : training_words_dict
    })

    # merge(training_words_layer1, training_words_dict)

