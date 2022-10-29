import enum
import networkx as nx
import pylab
import os
import myutils
import myutils_htm
import pandas as pd
import time
import numpy as np
import glob
import re
import tqdm
from operator import itemgetter
from time import perf_counter_ns, perf_counter
import neuspell
from neuspell import available_checkers, BertChecker, SclstmelmoChecker, BertsclstmChecker, NestedlstmChecker, CnnlstmChecker, SclstmChecker, SclstmbertChecker, ElmosclstmChecker
from tensorflow.keras import backend as K
import pickle
import torch

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

print("cuda" if torch.cuda.is_available() else "cpu")
# print(K.tensorflow_backend._get_available_gpus())



timestr = time.strftime("%Y%m%d-%H%M%S")

def readlines(filename):
    with open(filename, 'r') as f:
        return f.readlines()

def savelines(filename, lines):
    with open(filename, 'w') as f:
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


checker = pickle.load(open('NestedlstmChecker_layer1_train_a80.pkl', 'rb'))

ground_lines_saved = []
input_lines_saved = []
def evalSentence(input_sentences, target_sentences):

    count, error, countdiff, errordiff, countlimit, errormax, errorrange = 0, 0, 0, 0, 0, 0, 0

    errors = []
    timeperfsum = 0
    for index, sentence in tqdm.tqdm(enumerate(input_sentences)):
        
        if not isinstance(sentence, str):
          print('error not string : ', str(sentence))
          continue

        sentence = sentence.strip()
        input_sentense = sentence
        target_sentense = target_sentences[index]

        ground_lines_saved.append(target_sentense)
        input_lines_saved.append(input_sentense)

        target_words = target_sentense.split(' ')
        input_words = input_sentense.split(' ')
        if '' in input_words:
            continue

        if len(target_words) != len(input_words):
            continue
        
        t1_start = perf_counter()
        corrected_output = checker.correct(input_sentense)
        t1_stop = perf_counter()
        
        corrected_sentense = corrected_output
        timeperfsum += (t1_stop-t1_start)
        

        if (not target_sentense == corrected_sentense):
            # check each word
            countCorrect = countCorrectWords(corrected_sentense, target_sentense)
            error += (len(target_words) - countCorrect)

            
            errors.append([input_sentense, corrected_sentense, target_sentense])
            if count>0:
                perf = float(error)/count
            

        # if not input_sentense == target_sentense:
        #     countdiff += len(target_words)
        #     if not target_sentense == corrected_sentense:
        #         errordiff += len(target_words)
        
        if index%100 == 0 and count>0:
            print('error = ' + str(error) + ' errormax = ' + str(errormax) + ' errordh = ' + str(errordiff) + ' errorrange = ' + str(errorrange) +' count = ' + str(count) + ' countlimit = ' + str(countlimit) + ' time = ' + str(timeperfsum/count))
            # for i,c in enumerate(candidates_indexs):
            #     print(c)
            # print(candidates_indexs)
            candidates_indexs = []
            candidates_index_values = []
        
        count += len(target_words)
    return [count, error, countlimit, countdiff, errordiff, timeperfsum]


results = []
errors = []
train_test_set = myutils_htm.loadpkl('train_test_a.pkl')
input_test_words = train_test_set['input_test_words']
ground_test_words = train_test_set['ground_test_words']
# input_train_words = train_test_set['train_input_words']
# ground_train_words = train_test_set['train_ground_words']

result = evalSentence(input_test_words, ground_test_words)
print(result)
# results.append(result)
# note = 'hashsdr layer 2 only or with words in sentense'
# myutils.savejson('result_hashsdr_' + timestr + '_training100000.json', { 'file': __file__, 'note' : note, 'results' : results, 'errors' :  errors})


