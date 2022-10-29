import myutils_htm
import time
import numpy as np
import tqdm
import bitarray
import os
from multiprocessing import Process, Queue
import math
import tqdm
from time import perf_counter
from operator import itemgetter

THREAD = 10
MAX_BUCKET = 2048
timestr = time.strftime("%Y%m%d-%H%M%S")
filename_layer1 = 'eval_sdr_layer1_' + str(MAX_BUCKET) + '.pkl'
filename_meanlength = 'training_words_layer1_length_' + str(MAX_BUCKET) + '_{0}.pkl'

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
    # cadidates = []
    # for n in sorted(testint_data.items(), key=itemgetter(1)):
    #     cadidates.append([n[0], n[1]])
    # return cadidates
    return testint_data

def predictword_sdr_notmatch(word, ground_word, layer1_words, layer1, layer1_avglength, layer1_dict, max_bucket = MAX_BUCKET):
    if word in layer1_words:
        return { word : 1}
    wordR = bitarray.bitarray(myutils_htm.text2Representation1(word, max_bucket).tolist())
    allsum = np.array([(wordR & l).count() for l in layer1])
    allsum = (allsum*2)/((layer1_avglength + (wordR.count())))
    # all_indexs = (-allsum).argsort()
    rlen = 100
    if len(allsum)<rlen:
        rlen = len(allsum)
    all_indexs = np.argpartition(-allsum, range(rlen))[:rlen]
    cadicates = []
    for index in all_indexs:
        words = layer1_dict[str(index)]
        cadicates.extend(words)
    # print(word, cadicates)
    return predictword_LD(word, cadicates)

def predict_word_notmatch_callprocess(queues, out_queue, max_process, count, input_word, target_word):
    for k in range(len(queues)):
        queues[k].put([k, "run", [count, input_word, target_word]])
    return get_results(out_queue, max_process)

def test(input_texts, target_texts, queues, out_queue, max_process):
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
        candidates_and_values = predict_word_notmatch_callprocess(queues, out_queue, max_process, count, input_text, ground_text)       
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

            if not ground_text in candidates_range30:
                errorrange30 += 1

        if not input_text == ground_text:
            countdiff += 1
            if not ground_text == corrected_text:
                errordiff += 1
        count += 1
        if count%1000 == 0 and count>0:
            print('error = ' + str(error) + ' errormax = ' + str(errormax)  + ' errorrange = ' + str(errorrange) + ' errorrange30 = ' + str(errorrange30) +' count = ' + str(count) + ' countlimit = ' + str(countlimit) + ' time = ' + str(timeperfsum/count))
            # print(candidates_indexs)
            candidates_indexs = []
        

    print('error = ' + str(error) + ' errormax = ' + str(errormax)  + ' errorrange = ' + str(errorrange) + ' errorrange30 = ' + str(errorrange30) + ' count = ' + str(count) + ' countlimit = ' + str(countlimit) + ' time = ' + str(timeperfsum/count))
    return [count, error, countlimit, countdiff, errordiff, timeperfsum]

def get_masterdata_layer(k, max_bucket):
    
    global filename_meanlength
    global filename_layer1

    eval_sdr_layer1_merge = myutils_htm.loadpkl(filename_layer1)
    training_words_layer1 = eval_sdr_layer1_merge['training_words_layer1']
    # training_words_layer1 = [bitarray.bitarray(l.tolist()) for l in training_words_layer1]
    training_words_dict = eval_sdr_layer1_merge['training_words_dict']
    training_words_layer1_length = [l.count() for l in training_words_layer1]
    training_words_layer1_dict = {word : stri  for stri in training_words_dict.keys() for word in training_words_dict[stri]}
    training_words = sorted([word for stri in training_words_dict.keys() for word in training_words_dict[stri]])

    
    thread_layer_size = math.ceil(len(training_words_layer1)/THREAD)
    startk = k * thread_layer_size
    endk = (k + 1) * thread_layer_size

    if endk>len(training_words_layer1):
        endk = len(training_words_layer1)


    layers = [layer for layer in training_words_layer1[startk:endk]]
    # layer_meanlength = [meanlength for meanlength in training_sentence_meanlength[startk:endk]]
    print(k, "layers size : ", len(layers))
    layer_dict = {str(i) : training_words_dict[str(i + startk)] for i in range(endk-startk)  if str(i + startk) in training_words_dict}
    print(k, " layer_dict size : ", len(layer_dict.keys()))
    layer_word_dict = {word : stri  for stri in layer_dict.keys() for word in layer_dict[stri]}
    thread_training_words_1 = list(layer_word_dict.keys())

    del eval_sdr_layer1_merge

    filename_meanlength = filename_meanlength.format(str(k))
    if not os.path.isfile(filename_meanlength):
        layer_meanlength = [0] * len(layers)
        for stri in tqdm.tqdm(layer_dict.keys()):
            words = layer_dict[stri]
            words = list(set(words))
            sumlen = 0
            lens = []
            for word in words:
                word_presentation = bitarray.bitarray(myutils_htm.text2Representation1(word, max_bucket).tolist())
                len1 = word_presentation.count()
                sumlen = sumlen + len1
                lens.append(len1)

            layer_dict[stri] = words
            layer_meanlength[int(stri)] = np.mean(lens)

        myutils_htm.savepkl(filename_meanlength, layer_meanlength)
        layer_meanlength = np.array(layer_meanlength)
    else:
        layer_meanlength = np.array(myutils_htm.loadpkl(filename_meanlength))
        layer_meanlength = np.array(layer_meanlength)


    
    return thread_training_words_1, layers, layer_meanlength, layer_dict, layer_word_dict


def process_predict_sentense(k ,queue, queueout):
    layer1_trainingwords, layers1, layer1_meanlength, layer1_dict, layer_word_dict = get_masterdata_layer(k, MAX_BUCKET)
    queueout.put([k, "loaded"])

    print(k, "process has been started")
    while True:
        msg = queue.get()
        if msg[0] != k: 
            continue
        if msg[1] == "terminate":
            break
        inputid, input_sentence, target_sentence = msg[2]
        result = predictword_sdr_notmatch(input_sentence, target_sentence, layer1_trainingwords, layers1, layer1_meanlength, layer1_dict)
        queueout.put([k, msg[1], inputid, result])

def get_results(out_queue, max_process):
    l = 0
    result_dict = {}
    while l < max_process:
        result = out_queue.get()
        result_dict[str(result[0])] = result[3]
        l = l + 1
    
    result = {}
    for k in range(max_process):
        result = {**result, **result_dict[str(k)]}
    
    sentence_cadidates = []
    for n in sorted(result.items(), key=itemgetter(1)):
        sentence_cadidates.append([n[0], n[1]])
    return sentence_cadidates

def create_process(out_queue, max_process):
    queues = []
    processes =[]
    for k in range(max_process):
        child_queue = Queue()
        process_p  = Process(target=process_predict_sentense, args=(k, child_queue, out_queue,))
        process_p.daemon = True
        process_p.start()
        processes.append(process_p)
        queues.append(child_queue)
        out_queue.get()

    return processes, queues 


if __name__ == '__main__':
    out_queue = Queue()
    processes, child_queues  = create_process(out_queue, THREAD)
    print("processes have been loaded")

    train_test_set = myutils_htm.loadpkl('train_test_a.pkl')
    input_train_words = train_test_set['train_input_words']
    ground_train_words = train_test_set['train_ground_words']
    input_test_words = train_test_set['input_test_words']
    ground_test_words = train_test_set['ground_test_words']

    print("test size : ", len(ground_train_words))
    result = test(input_train_words, ground_train_words, child_queues, out_queue, THREAD)
    print(result)


    for k, process in enumerate(processes):
        queue = child_queues[k]
        queue.put([k, "terminate", []])
        process.terminate()
        time.sleep(0.1)
        if not process.is_alive():
            process.join(timeout=1.0)
            queue.close()
    out_queue.close()
