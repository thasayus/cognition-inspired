import tqdm
# from multiprocessing import Pool, Manager, freeze_support
import sys
import os
import inspect
import myutils_htm


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


def predictword_LD(word):
    # if word in keys:
    #     return [word, 0]
    testint_data = {}
    for dict in keys:
        testint_data[dict] = levenshteinDistance(dict, word)
    cadidates = []
    for n in sorted(testint_data.items(), key=itemgetter(1)):
        cadidates.append([n[0], n[1]])
    return cadidates

import tqdm
from time import perf_counter_ns, perf_counter
from operator import itemgetter
from collections import OrderedDict
max_cadidate = 0

def test(input_texts, target_texts):
    count, error, countdiff, errordiff, countlimit, errormax, errorrange = 0, 0, 0, 0, 0, 0, 0
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
        candidates_and_values = predictword_LD(word)       
        # candidates_and_values = predictword_sdr_notmatch(word, ground_text)
        t1_stop = perf_counter()
        # print(candidates_and_values[:5])
        if len(candidates_and_values)<=0:
            print("error candidates not found ", word)
            continue
        
        candidates = [c[0] for c in candidates_and_values]
        
        corrected_text = candidates[0]
        # print(word, corrected_text)

        timeperfsum += (t1_stop-t1_start)
        if len(candidates_and_values[0])<1:
            print("error value not found ", word)
            continue

        max_value = candidates_and_values[0][1]

        candicates_max = [ ]
        for i, c in enumerate(candidates_and_values):
            if c[1] > max_value:
                break
            else:
                candicates_max.append(c[0])
        # print(candicates_max)
        candidates_range = candidates[:30]
        
        if (not ground_text == corrected_text):
            if ground_text in candidates:
                countlimit += 1
            # else:
                # print('error!! ' + word)
            error += 1
            # errors.append([input_text, corrected_text, ground_text])
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


        if not input_text == ground_text:
            countdiff += 1
            if not ground_text == corrected_text:
                errordiff += 1
        
        if index%100 == 0 and count>0:
            print('error = ' + str(error) + ' errormax = ' + str(errormax)  + ' errorrange = ' + str(errorrange) +' count = ' + str(count) + ' countlimit = ' + str(countlimit) + ' time = ' + str(timeperfsum/count))
            # print(candidates_indexs)
            candidates_indexs = []
        count += 1
    return [count, error, countlimit, countdiff, errordiff, timeperfsum]

# trainingtemp = trainingtemp[trainingtemp['oword'].apply(searchWord)]

# training_set = myutils_htm.loadpkl('eval_sdr_layer1_train_dict.pkl')






# ground_file_test = 'full_sentences_ground_a40000_test80.txt'
# input_file_test = 'full_sentences_input_a40000_test80.txt'
# ground_test_lines = myutils_htm.readlines(ground_file_test)
# input_test_lines = myutils_htm.readlines(input_file_test)

# input_words = []
# ground_words = []
# for i, ground_line in tqdm.tqdm(enumerate(ground_test_lines)):
#     input_line = input_test_lines[i]
#     gwords  =ground_line.strip().split(' ')
#     iwords = input_line.strip().split(' ')
#     if len(gwords) == len(iwords):
#         input_words.extend(iwords)
#         ground_words.extend(gwords)
if __name__ == '__main__':
    args = sys.argv

    # train_test_set = myutils_htm.loadpkl('train_test_a.pkl')
    train_test_set = myutils_htm.loadpkl('train_test_a.pkl')
    ground_train_words = train_test_set['train_ground_words']
    input_train_words = train_test_set['train_input_words']
    input_test_words = train_test_set['input_test_words']
    ground_test_words = train_test_set['ground_test_words']

    if len(args)>=2:
        sample = int(args[1])
        input_test_words = input_test_words[:sample]
        ground_train_words = ground_train_words[:sample]
        ground_test_words = ground_test_words[:sample]
        input_train_words = input_train_words[:sample]

    keys = list(set(ground_train_words)) #training_set.keys()
    training_words = sorted(keys)

    print("Training Size:", len(training_words))

    # ground_train_words = ground_train_words[:100000]

    print("Testing size : ", len(input_test_words))
    result = test(input_test_words, ground_test_words)
    print(result)

