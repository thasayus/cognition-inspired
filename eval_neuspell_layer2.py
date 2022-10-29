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

def merge_sentences(corrected_sentence, ground_sentence):
    ground_words = ground_sentence.split(' ')
    corrected_words = corrected_sentence.split(' ')
    correct_ground_words = []
    j = 0
    prev_i = -1
    CHECK_LENGTH = 3
    for i, gword in enumerate(ground_words):

        if i + j >=len(corrected_words):
            correct_ground_words.append('')
            continue
        min_value = 9999
        min_a = -1
        for a in range(CHECK_LENGTH):
            if i + j + a >=len(corrected_words):
                break
            distance = myutils.levenshteinDistance(corrected_words[i + j + a], gword)
            if min_value > distance:
                min_value = distance
                min_a = a
        if min_value != 9999:
            if prev_i != -1:
                correct_ground_words.append(''.join(corrected_words[prev_i : i+j+min_a]))
            prev_i = i+j+min_a
            j = j + min_a

        if i == len(ground_words) - 1 and prev_i < len(corrected_words):
            correct_ground_words.append(''.join(corrected_words[prev_i : len(corrected_words)]))

    return correct_ground_words


def countCorrectWords(input_sentense, ground_sentense):
    input_words = input_sentense.split(' ')
    ground_word = ground_sentense.split(' ') 
    return sum([w == input_words[i] for i, w in enumerate(ground_word) if i < len(input_words)])

def findCorrectWordFromDistance(input_sentense, candidates):
    distances = {}
    for c in candidates:
        distances[c] = myutils.levenshteinDistance(input_sentense, c)
    cadidates = []
    for n in sorted(distances.items(), key=itemgetter(1)):
        cadidates.append(n[0])
    return cadidates

checker = ElmosclstmChecker() # BertChecker()
checker.from_pretrained()
# checker = pickle.load(open('ElmosclstmChecker_layer2_full_degrade5_ground_a_train80.pkl', 'rb'))
# checker.tokenize = False

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
        target_sentense = target_sentense.strip()

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

        corrected_output = ' '.join(merge_sentences(corrected_output, target_sentense))
        corrected_sentense = corrected_output
        timeperfsum += (t1_stop-t1_start)
        

        if (not target_sentense == corrected_sentense):
            # check each word
            # print(target_sentense, input_sentense, corrected_sentense)
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
full_degrade5_ground_a_test80 = myutils_htm.readlines('../full_degrade5_ground_a_test80.txt')
# full_degrade5_ground_a_train80 = myutils_htm.readlines('../full_degrade5_ground_a_train80.txt')
full_degrade5_input_a_test80 = myutils_htm.readlines('../full_degrade5_input_a_test80.txt')
# full_degrade5_input_a_train80 = myutils_htm.readlines('../full_degrade5_input_a_train80.txt')

result = evalSentence(full_degrade5_input_a_test80, full_degrade5_ground_a_test80)
print(result)
# results.append(result)
# note = 'hashsdr layer 2 only or with words in sentense'
# myutils.savejson('result_hashsdr_' + timestr + '_training100000.json', { 'file': __file__, 'note' : note, 'results' : results, 'errors' :  errors})


