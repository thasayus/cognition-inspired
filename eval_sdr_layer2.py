import enum
import os
import myutils_htm
import pandas as pd
import time
import numpy as np
import glob
import re
import tqdm
from operator import itemgetter
from time import perf_counter_ns, perf_counter
import json
import bitarray
import collections
import myutils
import math

SENTENCE_SIZE = 2
MAX_BUCKET = 2048
padding = "#####"
timestr = time.strftime("%Y%m%d-%H%M%S")
print("PYTHONHASHSEED : ", os.environ['PYTHONHASHSEED'])
filename_layer2 = 'training_fromsentence_layer2_seperated_2_4096_2048.pkl'
filename_length = 'training_sentence_meanlength_seperated_2_4096_2048.pkl'

def nested_dict():
    return collections.defaultdict(nested_dict)
def checkWord(words):
    return all([searchWord(w) for w in words])
def readjson(filename):
    with open(filename) as json_file:
        return json.load(json_file)
    return None
pattern_word = re.compile("^([\w]+)$")
# pattern_word = re.compile("^([a-zA-Z]+)$")
def searchWord(x):
    if pattern_word.search(x):
        return True
    else:
        return False
def searchFileWithLastDateModified(search, reverse=True):
    files = glob.glob(search)
    files = sorted(files, key=os.path.getmtime, reverse=reverse)
    return files
def createSentenseSDR_union(sentense, max_bucket):
    text_representation = None
    for i, word in enumerate(sentense.split(' ')):
        if i == 0:
            text_representation = bitarray.bitarray(myutils_htm.text2Representation1(word, max_bucket).tolist())
            continue
        text_representation = text_representation | bitarray.bitarray(myutils_htm.text2Representation1(word, max_bucket).tolist())
    return text_representation
def createSentenseSDR_concat(sentense, max_bucket):
    text_representation = None
    for i, word in enumerate(sentense.split(' ')):
        if i == 0:
            text_representation = bitarray.bitarray(myutils_htm.text2Representation1(word, max_bucket).tolist())
            continue
        text_representation = text_representation + bitarray.bitarray(myutils_htm.text2Representation1(word, max_bucket).tolist())
    return text_representation


# # process layer 1
# eval_sdr_layer1_merge = myutils_htm.loadpkl('../sdr_layer1_4096_0.2_0.2_20000/eval_sdr_layer1_remerge_4096_0.2_0.2_20000.pkl')
# eval_sdr_layer1_merge = myutils_htm.loadpkl('eval_sdr_layer1_remerge_4096_0.2_0.2_20000.pkl')
# training_words_layer1 = [bitarray.bitarray(l.tolist()) for l in eval_sdr_layer1_merge['training_words_layer1']]
# training_words_dict = eval_sdr_layer1_merge['training_words_dict']
# padding = "     "
# training_words_dict[str(len(training_words_layer1))] = set([padding,])
# training_words_layer1.append(bitarray.bitarray(myutils_htm.text2Representation1(padding, MAX_BUCKET).tolist()))
# training_words_layer1_length = [l.count() for l in training_words_layer1]
# training_words_layer1_dict = {word : stri  for stri in training_words_dict.keys() for word in training_words_dict[stri]}
# training_words = sorted([word for stri in training_words_dict.keys() for word in training_words_dict[stri]])
# del eval_sdr_layer1_merge

eval_sdr_layer2_merge = myutils_htm.loadpkl(filename_layer2)
training_sentence_layer2 = eval_sdr_layer2_merge['training_sentence_layer2']
training_sentence_dict = eval_sdr_layer2_merge['training_sentence_layer2_dict']
training_sentence_words_dict = {sentence : stri  for stri in training_sentence_dict.keys() for sentence in training_sentence_dict[stri]}
training_sentences = list(training_sentence_dict.keys())
training_sentences_count = [0] * len(training_sentence_layer2)
for stri in training_sentence_dict.keys():
    training_sentences_count[int(stri)] = len(training_sentence_dict[stri])
print("layer size : ", len(training_sentence_layer2))

if not os.path.isfile(filename_length):
    training_sentence_meanlength = [0] * len(training_sentence_layer2)
    for stri in tqdm.tqdm(training_sentence_dict.keys()):
        sentences = training_sentence_dict[stri]
        sentences = list(set(sentences))
        sumlen = 0
        lens = []
        for sentence in sentences:
            word_presentation = createSentenseSDR_concat(sentence, MAX_BUCKET)
            len1 = word_presentation.count()
            sumlen = sumlen + len1
            lens.append(len1)
        
        training_sentence_dict[stri] = sentences
        training_sentence_meanlength[int(stri)] = np.mean(lens)

    myutils_htm.savepkl(filename_length, training_sentence_meanlength)
else:
    training_sentence_meanlength = np.array(myutils_htm.loadpkl(filename_length))


training_sentence_meanlength = np.array(training_sentence_meanlength)

training_sentence_length = np.array([l.count()/len(training_sentence_dict[str(i)]) for i, l in enumerate(training_sentence_layer2)])

def distance_sentence(sentence1, sentence2):
    sentence1s = sentence1.split(' ')
    sentence2s = sentence2.split(' ')
    # if len(sentence1s)<len(sentence2s):
    #     temp = sentence2s
    #     sentence2s = sentence1s
    #     sentence1s = temp
    # result = [myutils.levenshteinDistance(s, sentence2s[i]) if i>=len(sentence2s) else myutils.levenshteinDistance(s, "") for i, s in enumerate(sentence1s)]
    result = [myutils.levenshteinDistance(s, sentence2s[i]) for i, s in enumerate(sentence1s)]

    return sum(result)

def predictsentence_LD(sentence, candidates):
    testint_data = {}
    for candidate in candidates:
        testint_data[candidate] = distance_sentence(sentence, candidate)
    sentence_cadidates = []
    for n in sorted(testint_data.items(), key=itemgetter(1)):
        sentence_cadidates.append([n[0], n[1]])
    return sentence_cadidates

def predict_sentense_notmatch(input_sentence, ground_sentence):
    if input_sentence in training_sentences:
        return [[input_sentence, 1]]
    # t1 = perf_counter()
    input_sentenseR = createSentenseSDR_concat(input_sentence, MAX_BUCKET)
    # print(input_sentence)
    # t2 = perf_counter()
    # print("time createSentenseSDR_concat", t2-t1)

    # allR = np.logical_and(np.logical_not(input_sentenseR), training_sentenses_layer2)
    # allsum = allR.sum(axis=1)
    # length = sum([len(word) for word in test_sentense.split(' ')])
    
    # allsum = np.array([((~input_sentenseR) & l).count() for l in training_sentence_layer2])
    # allsum = allsum - training_sentence_meanlength + 1
    
    # t1 = perf_counter()

    # allsum_c = np.array([(input_sentenseR & l).count() for l in training_sentence_layer2])
    # allsum = (allsum_c*2)/((training_sentence_length + (input_sentenseR.count())))

    allsum_c = np.array([((input_sentenseR) & l).count() for l in training_sentence_layer2])
    allsum = (allsum_c*2)/((training_sentence_meanlength + (input_sentenseR.count())))
    
    # t2 = perf_counter()
    # print("time | ", t2-t1)
    # t1_stop = perf_counter()
    # print("logical_and : ", t1_stop - t1_start)
    # allsum = allR.sum(axis=1)
    # length = sum([len(word) for word in test_sentense.split(' ')])
    # allsum = allsum - (training_sentenses_layer1_length_representation + sum(wordR))/2 + 1
    # allsum = abs(allsum)
    # allsum = (allsum + 1) / ((training_sentenses_layer1_length_representation + sum(wordR))/2)
    # t1 = perf_counter()
    # all_indexs = (-allsum).argsort()
    rlen = 300
    if len(allsum)<rlen:
        rlen = len(allsum)
    # all_indexs = np.argpartition(-allsum, range(rlen))[:rlen]
    all_indexs = np.argpartition(-allsum, range(rlen))[:rlen]

    # print(allsum[all_indexs[:5]], [training_words_dict[str(index)]  for index in all_indexs[:5]])
    # all_indexs_values = np.sort(allsum)[::-1]
    # index_values = [[training_words_dict[str(index)], allsum[index]]]
    cadicates = []
    for index in all_indexs:
        sentences = training_sentence_dict[str(index)]
        cadicates.extend(sentences)
    # print(word, cadicates)
    # gindexstr = training_sentence_words_dict[ground_sentence]
    # t2 = perf_counter()
    # print("time argpartition ", t2-t1)
    # print(ground_sentence in cadicates, input_sentenseR.count(), allsum_c[int(gindexstr)], allsum[int(gindexstr)], allsum_c[all_indexs[0]], allsum[all_indexs[0]])
    # t1 = perf_counter()
    result = predictsentence_LD(input_sentence, cadicates)
    # t2 = perf_counter()
    # print("time predictsentence_LD ", t2-t1)

    # stri = training_words_layer1_dict[ground_word]
    # print(ground_word, allsum[int(stri)], word, allsum[all_indexs[0]], training_words_dict[str(all_indexs[0])])

    return result

def countCorrectWords(input_sentense, ground_sentense):
    input_words = input_sentense.split(' ')
    ground_word = ground_sentense.split(' ') 
    return sum([w == ground_word[i] for i, w in enumerate(input_words)])

def findCorrectWordFromDistance(input_sentense, candidates):
    distances = {}
    for c in candidates:
        distances[c] = myutils.levenshteinDistance(input_sentense, c)
    cadidates = []
    for n in sorted(distances.items(), key=itemgetter(1)):
        cadidates.append(n[0])
    return cadidates



def evalSentence(input_sentences, target_sentences):

    count, error, countdiff, errordiff, countlimit, errormax, errorrange = 0, 0, 0, 0, 0, 0, 0
    candidates_indexs = []
    candidates_index_values = []

    errors = []
    timeperfsum = 0
    for index, sentence in tqdm.tqdm(enumerate(input_sentences)):

        if not isinstance(sentence, str):
            print('error not string : ', str(sentence))
            continue
        
        input_sentense1 = sentence.strip()
        if input_sentense1 == "": continue

        target_sentense1 = target_sentences[index]
        target_sentense1 = target_sentense1.strip()


        target_words = target_sentense1.split(' ')
        input_words = input_sentense1.split(' ')

        if len(target_words) != len(input_words):
            continue

        if len(input_words)<SENTENCE_SIZE:
            [input_words.append(padding) for k in range(SENTENCE_SIZE - len(input_words))]

        if len(target_words)<SENTENCE_SIZE:
            [target_words.append(padding) for k in range(SENTENCE_SIZE - len(target_words))]

        sentence_input_resizes = [' '.join(input_words[j:j + SENTENCE_SIZE]) for j in range(0, len(input_words) - SENTENCE_SIZE + 1, SENTENCE_SIZE)]
        sentence_ground_resizes = [' '.join(target_words[j:j + SENTENCE_SIZE]) for j in range(0, len(target_words) - SENTENCE_SIZE + 1, SENTENCE_SIZE)]
        
        if len(input_words)%SENTENCE_SIZE != 0:
            sentence_input_resizes.append(' '.join(input_words[len(input_words)-SENTENCE_SIZE:len(input_words)]))

        if len(target_words)%SENTENCE_SIZE != 0:
            sentence_ground_resizes.append(' '.join(target_words[len(target_words)-SENTENCE_SIZE:len(target_words)]))

        for a, input_sentense in enumerate(sentence_input_resizes):
            target_sentense = sentence_ground_resizes[a]
            target_sentense_words = target_sentense.split(' ')
            t1_start = perf_counter()
            candidates_and_values = predict_sentense_notmatch(input_sentense, target_sentense)
            t1_stop = perf_counter()
            # print("time : ", (t1_stop - t1_start)/SENTENCE_SIZE)
            candidates = [c[0] for c in candidates_and_values]
            
            corrected_sentense = candidates[0]
            timeperfsum += (t1_stop-t1_start)
            
            max_candidate_value = candidates_and_values[0][1]
            candicates_max = []
            for i, c in enumerate(candidates_and_values):
                if c[1] > max_candidate_value:
                    break
                else:
                    candicates_max.append(c[0])

            candidates_range = candidates

            if (not target_sentense == corrected_sentense):
                # check each word
                countCorrect = countCorrectWords(corrected_sentense, target_sentense)
                error += (len(target_sentense_words) - countCorrect)

                if target_sentense in candidates:
                    # every word should be in candidates
                    # countCorrect = countCorrectWords(input_sentense, target_sentense)
                    countlimit += len(target_sentense_words)
                # else:
                    # print('error!! ' + input_sentense)
                
                errors.append([input_sentense, corrected_sentense, target_sentense])
                if count>0:
                    perf = float(error)/count
                
                if not target_sentense in candicates_max:
                    max_sentense = ''
                    count_max_sentense = 0
                    for c in candicates_max:
                        countCorrect = countCorrectWords(c, target_sentense)
                        if (count_max_sentense<countCorrect):
                            max_sentense = c
                            count_max_sentense = countCorrect
                    errormax += (len(target_sentense_words) - count_max_sentense)

                    # print('errormax!! ', input_sentense, '|' ,corrected_sentense, '|' , str(myutils.levenshteinDistance(input_sentense, corrected_sentense)) + '|' ,target_sentense , '|' , str(myutils.levenshteinDistance(input_sentense, target_sentense)))

                if not target_sentense in candidates_range:
                    count_max_sentense = 0
                    for c in candidates_range:
                        countCorrect = countCorrectWords(c, target_sentense)
                        if (count_max_sentense<countCorrect):
                            max_sentense = c
                            count_max_sentense = countCorrect
                    errorrange += (len(target_sentense_words) - count_max_sentense)

                    # print('errorrange!! ', word, ground_text)
                    # for i, c in enumerate(candidates):
                    #     if i >= 30:
                    #         if c == target_sentense:
                    #             candidates_indexs.append([input_sentense, corrected_sentense, target_sentense])
                    #             candidates_index_values.append([max_candidate_value, candidates_and_values[i]])
                
            # candidates_range_dh = findCorrectWordFromDistance(input_sentense, candidates_range)
            # if not target_sentense == candidates_range_dh[0]:
            #     countCorrect = countCorrectWords(input_sentense, candidates_range_dh[0])
            #     errordiff += (len(target_words) - countCorrect)
            #     candidates_indexs.append([input_sentense, candidates_range_dh[0], target_sentense])


            # if not input_sentense == target_sentense:
            #     countdiff += len(target_words)
            #     if not target_sentense == corrected_sentense:
            #         errordiff += len(target_words)
            
            if count%100 == 0 and count>0:
                print('error = ' + str(error) + ' errormax = ' + str(errormax) + ' errordh = ' + str(errordiff) + ' errorrange = ' + str(errorrange) +' count = ' + str(count) + ' countlimit = ' + str(countlimit) + ' time = ' + str(timeperfsum/count))
                # for i,c in enumerate(candidates_indexs):
                #     print(c)
                # print(candidates_indexs)
                candidates_indexs = []
                candidates_index_values = []
            
            count += len(target_sentense_words)
    
    print('error = ' + str(error) + ' errormax = ' + str(errormax) + ' errordh = ' + str(errordiff) + ' errorrange = ' + str(errorrange) +' count = ' + str(count) + ' countlimit = ' + str(countlimit) + ' time = ' + str(timeperfsum/count))

    return [count, error, countlimit, countdiff, errordiff, timeperfsum]

full_degrade5_ground_a_test80 = myutils_htm.readlines('../full_degrade5_ground_a_test80.txt')
# full_degrade5_ground_a_train80 = myutils_htm.readlines('../full_degrade5_ground_a_train80.txt')
# full_degrade5_ground_a_train80 = myutils_htm.readlines('full_degrade5_ground_a_train80.txt')
full_degrade5_input_a_test80 = myutils_htm.readlines('../full_degrade5_input_a_test80.txt')
# full_degrade5_input_a_train80 = myutils_htm.readlines('../full_degrade5_input_a_train80.txt')

print("max bucket : ", MAX_BUCKET)
print("sentence size : ", SENTENCE_SIZE)
print("data set : ", "train")
print("testing data line size : ", len(full_degrade5_input_a_test80))

result = evalSentence(full_degrade5_ground_a_test80, full_degrade5_ground_a_test80)
