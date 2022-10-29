import enum
import networkx as nx
import pylab
import os
import myutils
import myutils_htm
import time
from operator import itemgetter
from time import perf_counter_ns, perf_counter
from neuspell import available_checkers, ElmosclstmChecker
from tensorflow.keras import backend as K
import pickle
import torch

print("cuda" if torch.cuda.is_available() else "cpu")
# print(K.tensorflow_backend._get_available_gpus())

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

timestr = time.strftime("%Y%m%d-%H%M%S")

def readlines(filename):
    with open(filename, 'r', encoding="utf-8") as f:
        return f.readlines()

def savelines(filename, lines):
    with open(filename, 'w', encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")

def countCorrectWords(input_sentense, ground_sentense):
    input_words = input_sentense.split(' ')
    ground_word = ground_sentense.split(' ') 
    return sum([w == input_words[i] for i, w in enumerate(ground_word)])

def findCorrectWordFromDistance(input_sentense, candidates):
    distances = {}
    for c in candidates:
        distances[c] = myutils.levenshteinDistance(input_sentense, c)
    cadidates = []
    for n in sorted(distances.items(), key=itemgetter(1)):
        cadidates.append(n[0])
    return cadidates


# train_test_set = myutils_htm.loadpkl('train_test_a.pkl')
# ground_words = train_test_set['train_ground_words']
# input_words = train_test_set['train_input_words']
# savelines('train_ground_words_neuspell_layer1.txt', ground_words)
# savelines('train_input_words_neuspell_layer1.txt', input_words)

# print("training size : " , len(ground_words))

checker = ElmosclstmChecker() # BertChecker()
checker.from_pretrained()
checker.finetune(clean_file="train_ground_words_neuspell_layer1.txt", corrupt_file="train_input_words_neuspell_layer1.txt", data_dir="default", n_epochs=2)
pickle.dump(checker, open('ElmosclstmChecker_layer1_train_a80.pkl', 'wb'))
