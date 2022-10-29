import os
import myutils_htm
import time
import numpy as np
import tqdm
from operator import itemgetter
from time import perf_counter
import bitarray
import myutils
import math
from multiprocessing import Process, Queue
import os.path


SENTENCE_SIZE = 2
MAX_BUCKET = 2048
THREAD = 10
padding = "#####"
timestr = time.strftime("%Y%m%d-%H%M%S")
print("PYTHONHASHSEED-0: ", os.environ['PYTHONHASHSEED'])
layer2_filename = 'training_frommergelayer1_layer2_seperated_remerge_2_4096_2048_0.1_1.975.pkl'
meanlength_filename = 'training_frommergelayer1_layer2_seperated_remerge_meanlength_2_4096_2048_0.1_1.975_{0}.pkl'

# layer2_filename = 'training_fromsentence_layer2_seperated_' + str(SENTENCE_SIZE)+ '_'  + str(MAX_BUCKET) + '.pkl'
# meanlength_filename = 'training_sentence_meanlength_seperated_process_0.2_0.2_' + str(SENTENCE_SIZE)+ '_'  + str(MAX_BUCKET) + '_{0}.pkl'
datapath = "../"

def distance_sentence(sentence1, sentence2):
    sentence1s = sentence1.split(' ')
    sentence2s = sentence2.split(' ')
    result = [myutils.levenshteinDistance(s, sentence2s[i]) for i, s in enumerate(sentence1s)]

    return sum(result)

def predictsentence_LD(sentence, candidates):
    testint_data = {}
    for candidate in candidates:
        testint_data[candidate] = distance_sentence(sentence, candidate)
    # sentence_cadidates = []
    # for n in sorted(testint_data.items(), key=itemgetter(1)):
    #     sentence_cadidates.append([n[0], n[1]])
    return testint_data

def predictsentence_LD_main(sentence, candidates):
    testint_data = {}
    for candidate in candidates:
        testint_data[candidate] = distance_sentence(sentence, candidate)
    sentence_cadidates = []
    for n in sorted(testint_data.items(), key=itemgetter(1)):
        sentence_cadidates.append([n[0], n[1]])
    return testint_data

def predict_sentense_notmatch_main(input_sentence, ground_sentence, layer2_sentences, layers2, layer2_meanlength, layer2_dict, max_bucket = MAX_BUCKET):
    if input_sentence in layer2_sentences:
        return [ [input_sentence , 1] ]
    input_sentenseR = createSentenseSDR_concat(input_sentence, max_bucket)
    allsum_c = np.array([(input_sentenseR & l).count() for l in layers2])
    allsum = (allsum_c*2)/((layer2_meanlength + (input_sentenseR.count())))
    rlen = 30
    if len(allsum)<rlen:
        rlen = len(allsum)
    all_indexs = np.argpartition(-allsum, range(rlen))[:rlen]
    cadicates = []
    for index in all_indexs:
        sentences = layer2_dict[str(index)]
        cadicates.extend(sentences)
    result = predictsentence_LD_main(input_sentence, cadicates)

    return result

def createSentenseSDR_concat(sentense, max_bucket):
    text_representation = None
    for i, word in enumerate(sentense.split(' ')):
        if i == 0:
            text_representation = bitarray.bitarray(myutils_htm.text2Representation1(word, max_bucket).tolist())
            continue
        text_representation = text_representation + bitarray.bitarray(myutils_htm.text2Representation1(word, max_bucket).tolist())
    return text_representation

def predict_sentense_notmatch(input_sentence, ground_sentence, layer2_sentences, layers2, layer2_meanlength, layer2_dict, max_bucket = MAX_BUCKET):
    if input_sentence in layer2_sentences:
        return { input_sentence : 1 }
    input_sentenseR = createSentenseSDR_concat(input_sentence, max_bucket)
    allsum_c = np.array([(input_sentenseR & l).count() for l in layers2])
    allsum = (allsum_c*2)/((layer2_meanlength + (input_sentenseR.count())))
    rlen = 30
    if len(allsum)<rlen:
        rlen = len(allsum)
    all_indexs = np.argpartition(-allsum, range(rlen))[:rlen]
    cadicates = []
    for index in all_indexs:
        sentences = layer2_dict[str(index)]
        cadicates.extend(sentences)
    result = predictsentence_LD(input_sentence, cadicates)

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

def predict_sentense_notmatch_callprocess(queues, out_queue, max_process, count, input_sentence, target_sentence):
    if input_sentence in main_training_sentences:
        return [ [input_sentence, 1] ]
    for k in range(len(queues)):
        queues[k].put([k, "run", [count, input_sentence, target_sentence]])
    return get_results(out_queue, max_process)

main_training_sentence_layer2 = None
main_training_sentence_dict = None
training_sentence_words_dict = None
main_training_sentences = None
def get_main_masterdata_layer():
    eval_sdr_layer2_merge = myutils_htm.loadpkl(layer2_filename)
    training_sentence_layer2 = eval_sdr_layer2_merge['training_sentence_layer2']
    training_sentence_dict = eval_sdr_layer2_merge['training_sentence_layer2_dict']
    training_sentence_words_dict = {sentence : stri  for stri in training_sentence_dict.keys() for sentence in training_sentence_dict[stri]}
    del eval_sdr_layer2_merge
    return training_sentence_layer2, training_sentence_dict, training_sentence_words_dict, 


def get_masterdata_layer(k, sentence_size, max_bucket):
    
    global meanlength_filename

    eval_sdr_layer2_merge = myutils_htm.loadpkl(layer2_filename)
    training_sentence_layer2 = eval_sdr_layer2_merge['training_sentence_layer2']
    training_sentence_dict = eval_sdr_layer2_merge['training_sentence_layer2_dict']
    training_sentence_words_dict = {sentence : stri  for stri in training_sentence_dict.keys() for sentence in training_sentence_dict[stri]}
    training_sentences = list(training_sentence_words_dict.keys())

    
    thread_layer_size = math.ceil(len(training_sentence_layer2)/THREAD)
    startk = k * thread_layer_size
    endk = (k + 1) * thread_layer_size

    if endk>len(training_sentence_layer2):
        endk = len(training_sentence_layer2)


    layers = [layer for layer in training_sentence_layer2[startk:endk]]
    # layer_meanlength = [meanlength for meanlength in training_sentence_meanlength[startk:endk]]
    print(k, "layers size : ", len(layers))
    layer_dict = {str(i) : training_sentence_dict[str(i + startk)] for i in range(endk-startk)  if str(i + startk) in training_sentence_dict}
    print(k, " layer_dict size : ", len(layer_dict.keys()))
    layer_words_dict = {sentence : stri  for stri in layer_dict.keys() for sentence in layer_dict[stri]}
    thread_training_sentences_1 = list(layer_words_dict.keys())

    del eval_sdr_layer2_merge

    meanlength_filename = meanlength_filename.format(str(k))
    if not os.path.isfile(meanlength_filename):
        layer_meanlength = [0] * len(layers)
        for stri in tqdm.tqdm(layer_dict.keys()):
            sentences = layer_dict[stri]
            sentences = list(set(sentences))
            sumlen = 0
            lens = []
            for sentence in sentences:
                word_presentation = createSentenseSDR_concat(sentence, max_bucket)
                len1 = word_presentation.count()
                sumlen = sumlen + len1
                lens.append(len1)

            layer_dict[stri] = sentences
            layer_meanlength[int(stri)] = np.mean(lens)

        myutils_htm.savepkl(meanlength_filename, layer_meanlength)
        layer_meanlength = np.array(layer_meanlength)
    else:
        layer_meanlength = np.array(myutils_htm.loadpkl(meanlength_filename))
        layer_meanlength = np.array(layer_meanlength)


    
    return thread_training_sentences_1, layers, layer_meanlength, layer_dict, layer_words_dict


def process_predict_sentense(k ,queue, queueout):
    layer2_sentences, layers2, layer2_meanlength, layer2_dict, layer_words_dict = get_masterdata_layer(k, SENTENCE_SIZE, MAX_BUCKET)
    queueout.put([k, "loaded"])

    print(k, "process has been started")
    while True:
        msg = queue.get()
        if msg[0] != k: 
            continue
        if msg[1] == "terminate":
            break
        inputid, input_sentence, target_sentence = msg[2]
        result = predict_sentense_notmatch(input_sentence, target_sentence, layer2_sentences, layers2, layer2_meanlength, layer2_dict)
        queueout.put([k, msg[1], inputid, result, input_sentence, target_sentence])

        

def evalSentence(input_sentences, target_sentences, queues, out_queue, max_process):
    
    print("evalSentence")

    count, error, countdiff, errordiff, countlimit, errormax, errorrange, errorrange30 = 0, 0, 0, 0, 0, 0, 0, 0
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
            target_sentense_words = target_sentense.split()

            t1_start = perf_counter()

            
            candidates_and_values = predict_sentense_notmatch_callprocess(queues, out_queue, max_process, count, input_sentense, target_sentense)
            t1_stop = perf_counter()
            
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
            candidates_range30 = candidates[:30]

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
                # if count>0:
                #     perf = float(error)/count
                
                if not target_sentense in candicates_max:
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

                if not target_sentense in candidates_range30:
                    count_max_sentense = 0
                    for c in candidates_range30:
                        countCorrect = countCorrectWords(c, target_sentense)
                        if (count_max_sentense<countCorrect):
                            count_max_sentense = countCorrect
                    errorrange30 += (len(target_sentense_words) - count_max_sentense)
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
            
            count += len(target_sentense_words)

            if count % (100 * SENTENCE_SIZE) == 0 and count>0:
                print('error = ' + str(error) + ' errormax = ' + str(errormax) + ' errorrange30 = ' + str(errorrange30) + ' errorrange = ' + str(errorrange) +' count = ' + str(count) + ' countlimit = ' + str(countlimit) + ' time = ' + str(timeperfsum/count))
                # for i,c in enumerate(candidates_indexs):
                #     print(c)
                # print(candidates_indexs)
                # candidates_indexs = []
                # candidates_index_values = []
            
            
            # print("count ", count)
    
    print('error = ' + str(error) + ' errormax = ' + str(errormax) + ' errordh = ' + str(errordiff) + ' errorrange = ' + str(errorrange) +' count = ' + str(count) + ' countlimit = ' + str(countlimit) + ' time = ' + str(timeperfsum/count))

    return [count, error, countlimit, countdiff, errordiff, timeperfsum]

def get_results(out_queue, max_process):
    l = 0
    result_dict = {}
    input_sentence = ""
    ground_sentence = ""
    while l < max_process:
        result = out_queue.get()
        result_dict[str(result[0])] = result[3]
        input_sentence = result[4]
        ground_sentence = result[5]
        l = l + 1
    
    result = {}
    for k in range(max_process):
        result = {**result, **result_dict[str(k)]}

    # main_layer2 = []
    # layer2_dict = {}
    # layer2_sentences = list(result.keys())
    # # print(layer2_sentences)
    # for i, sentence in enumerate(layer2_sentences):
    #     stri = training_sentence_words_dict[sentence]
    #     layer = main_training_sentence_layer2[int(stri)]
    #     main_layer2.append(layer)
    #     if sentence not in layer2_dict:
    #         layer2_dict[str(i)] = set()
    #     layer2_dict[str(i)].add(sentence)

    # # print(output)
    # layer2_meanlength = [0] * len(main_layer2)
    # for stri in tqdm.tqdm(layer2_dict.keys()):
    #     sentences = layer2_dict[stri]
    #     sentences = list(set(sentences))
    #     sumlen = 0
    #     lens = [createSentenseSDR_concat(sentence, MAX_BUCKET).count() for sentence in sentences]
    #     # layer2_dict[stri] = sentences
    #     layer2_meanlength[int(stri)] = np.mean(lens)
    # # print(layer2_meanlength)
    # output = predict_sentense_notmatch_main(input_sentence, ground_sentence, layer2_sentences, main_layer2, layer2_meanlength, layer2_dict, MAX_BUCKET)
    
    # print(output)
    # return output
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
    
    main_training_sentence_layer2, main_training_sentence_dict, training_sentence_words_dict = get_main_masterdata_layer()
    main_training_sentences = sorted(list(training_sentence_words_dict.keys()))
    # full_degrade5_ground_a_test80 = myutils_htm.readlines('../full_degrade5_ground_a_test80.txt')
    # full_degrade5_ground_a_train80 = myutils_htm.readlines(datapath + 'full_degrade5_ground_a_train80.txt')
    full_degrade5_ground_a_train80 = myutils_htm.readlines('../full_degrade5_ground_a_train80.txt')
    # full_degrade5_input_a_test80 = myutils_htm.readlines('../full_degrade5_input_a_test80.txt')
    full_degrade5_input_a_train80 = myutils_htm.readlines('../full_degrade5_input_a_train80.txt')

    result = evalSentence(full_degrade5_input_a_train80, full_degrade5_ground_a_train80, child_queues, out_queue, THREAD)

    for k, process in enumerate(processes):
        queue = child_queues[k]
        queue.put([k, "terminate", []])
        process.terminate()
        time.sleep(0.1)
        if not process.is_alive():
            process.join(timeout=1.0)
            queue.close()
    out_queue.close()
