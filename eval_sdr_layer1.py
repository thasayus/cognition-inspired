import myutils_htm
import time
import numpy as np
import tqdm
import bitarray
import os

MAX_BUCKET = 4096
timestr = time.strftime("%Y%m%d-%H%M%S")
filename_layer1 = 'eval_sdr_layer1_' + str(MAX_BUCKET) + '.pkl'
filename_meanlength = 'training_words_layer1_length_' + str(MAX_BUCKET) + '.pkl'

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



eval_sdr_layer1_merge = myutils_htm.loadpkl(filename_layer1)
training_words_layer1 = eval_sdr_layer1_merge['training_words_layer1']
# training_words_layer1 = [bitarray.bitarray(l.tolist()) for l in training_words_layer1]
training_words_dict = eval_sdr_layer1_merge['training_words_dict']
training_words_layer1_length = [l.count() for l in training_words_layer1]
training_words_layer1_dict = {word : stri  for stri in training_words_dict.keys() for word in training_words_dict[stri]}
training_words = sorted([word for stri in training_words_dict.keys() for word in training_words_dict[stri]])

if not os.path.isfile(filename_meanlength):
    training_words_dict_avglength = {}
    training_words_dict_maxlength = {}
    training_words_dict_minlength = {}
    training_words_dict_count = {}
    for stri in tqdm.tqdm(training_words_dict.keys()):
        words = training_words_dict[stri]
        training_words_dict[stri] = list(set(words))
        sumlen = 0
        lens = []
        for word in words:
            word_presentation = myutils_htm.text2Representation1(word)
            len1 = sum(word_presentation)
            sumlen = sumlen + len1
            lens.append(len1)
        
        training_words_dict_avglength[stri] = sumlen/len(words)
        training_words_dict_maxlength[stri] = max(lens)
        training_words_dict_minlength[stri] = min(lens)
        training_words_dict_count[stri] = len(words)

    training_words_layer1_avglength = [training_words_dict_avglength[str(i)] for i in range(len(training_words_dict_avglength.keys()))]
    training_words_layer1_maxlength = [training_words_dict_maxlength[str(i)] for i in range(len(training_words_dict_maxlength.keys()))]
    training_words_layer1_minlength = [training_words_dict_minlength[str(i)] for i in range(len(training_words_dict_minlength.keys()))]
    training_words_layer1_count = [training_words_dict_count[str(i)] for i in range(len(training_words_dict_count.keys()))]

    myutils_htm.savepkl(filename_meanlength, {
        "training_words_layer1_avglength" : training_words_layer1_avglength,
        "training_words_dict_maxlength" : training_words_layer1_maxlength,
        "training_words_dict_minlength" : training_words_layer1_minlength,
        "training_words_dict_count" : training_words_layer1_count,
        })
    training_words_layer1_avglength = np.array(training_words_layer1_avglength)
else:
    training_words_layer1_length = myutils_htm.loadpkl(filename_meanlength)
    training_words_layer1_avglength = np.array(training_words_layer1_length['training_words_layer1_avglength'])
    training_words_layer1_maxlength = training_words_layer1_length['training_words_dict_maxlength']
    training_words_layer1_minlength = training_words_layer1_length['training_words_dict_minlength']
    training_words_layer1_count = training_words_layer1_length['training_words_dict_count']


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

def predictword_LD(word, keys):
    # if word in keys:
    #     return [word, 0]
    testint_data = {}
    for dict in keys:
        testint_data[dict] = levenshteinDistance(dict, word)
    cadidates = []
    for n in sorted(testint_data.items(), key=itemgetter(1)):
        cadidates.append([n[0], n[1]])
    return cadidates
def predictword_sdr_notmatch(word, ground_word):
    if word in training_words:
        return [[word, 1]]
    wordR = bitarray.bitarray(myutils_htm.text2Representation1(word, MAX_BUCKET).tolist())
    # t1_start = perf_counter()
    allsum = np.array([(wordR & l).count() for l in training_words_layer1])
    # t1_stop = perf_counter()
    # print("logical_and : ", t1_stop - t1_start)
    # allsum = allR.sum(axis=1)
    # length = sum([len(word) for word in test_sentense.split(' ')])
    # allsum = allsum - (training_sentenses_layer1_length_representation + sum(wordR))/2 + 1
    # allsum = abs(allsum)
    # allsum = (allsum + 1) / ((training_sentenses_layer1_length_representation + sum(wordR))/2)
    allsum = (allsum*2)/((training_words_layer1_avglength + (wordR.count())))
    # all_indexs = (-allsum).argsort()
    rlen = 100
    if len(allsum)<rlen:
        rlen = len(allsum)
    all_indexs = np.argpartition(-allsum, range(rlen))[:rlen]
    # print(allsum[all_indexs[:5]], [training_words_dict[str(index)]  for index in all_indexs[:5]])
    # all_indexs_values = np.sort(allsum)[::-1]
    # index_values = [[training_words_dict[str(index)], allsum[index]]]
    cadicates = []
    for index in all_indexs:
        words = training_words_dict[str(index)]
        cadicates.extend(words)
    # print(word, cadicates)
    return predictword_LD(word, cadicates)

import tqdm
from time import perf_counter_ns, perf_counter
from operator import itemgetter
from collections import OrderedDict
max_cadidate = 0

def test(input_texts, target_texts):
    count, error, countdiff, errordiff, countlimit, errormax, errorrange, errorrange30 = 0, 0, 0, 0, 0, 0, 0, 0
    candidates_indexs = []
    timeperfsum = 0
    perf = 0.0
    
    for index, word in tqdm.tqdm(enumerate(input_texts)):
        if not isinstance(word, str):
          print('error not string : ', str(word))
          continue
        word = word.strip()
        ground_text = target_texts[index].strip()
        input_text = word

        t1_start = perf_counter()
        candidates_and_values = predictword_sdr_notmatch(word, ground_text)       
        # candidates_and_values = predictword_sdr_notmatch(word, ground_text)
        t1_stop = perf_counter()

        candidates = [c[0] for c in candidates_and_values]
        
        corrected_text = candidates[0]
        # print(word, corrected_text)

        timeperfsum += (t1_stop-t1_start)
        
        max_value = candidates_and_values[0][1]

        candicates_max = [ ]
        for i, c in enumerate(candidates_and_values):
            if c[1] < max_value:
                break
            else:
                candicates_max.append(c[0])
        # print(max_value)
        candidates_range = candidates
        candidates_range30 = candidates[:30]

        if (not ground_text == corrected_text):
            if ground_text in candidates:
                countlimit += 1
            # else:
                # print('error!! ' + word)
            error += 1
            errors.append([input_text, corrected_text, ground_text])
            if count>0:
                perf = float(error)/count
            # print(candicates_max)
            if not ground_text in candicates_max:
                errormax += 1
                # print('errormax!! ', word, ground_text, candicates_max)

            if not ground_text in candidates_range:
                errorrange += 1
                # print('errorrange!! ', word, ground_text)
                for i, c in enumerate(candidates):
                    if i >= 30:
                        if c == ground_text:
                            candidates_indexs.append([word, ground_text])

            if not ground_text in candidates_range30:
                errorrange30 += 1

        if not input_text == ground_text:
            countdiff += 1
            if not ground_text == corrected_text:
                errordiff += 1
        
        if index%100 == 0 and count>0:
            print('error = ' + str(error) + ' errormax = ' + str(errormax)  + ' errorrange = ' + str(errorrange) + ' errorrange30 = ' + str(errorrange30) +' count = ' + str(count) + ' countlimit = ' + str(countlimit) + ' time = ' + str(timeperfsum/count))
            # print(candidates_indexs)
            candidates_indexs = []
        count += 1

    print('error = ' + str(error) + ' errormax = ' + str(errormax)  + ' errorrange = ' + str(errorrange) + ' errorrange30 = ' + str(errorrange30) + ' count = ' + str(count) + ' countlimit = ' + str(countlimit) + ' time = ' + str(timeperfsum/count))
    return [count, error, countlimit, countdiff, errordiff, timeperfsum]

errors = []
print("Training Layer size:", len(training_words_layer1))
print("Training Vocab Size:", len(training_words))

train_test_set = myutils_htm.loadpkl('train_test_a.pkl')
input_train_words = train_test_set['train_input_words']
ground_train_words = train_test_set['train_ground_words']
input_test_words = train_test_set['input_test_words']
ground_test_words = train_test_set['ground_test_words']

print("test size : ", len(ground_train_words))
result = test(input_train_words, ground_train_words)
print(result)
