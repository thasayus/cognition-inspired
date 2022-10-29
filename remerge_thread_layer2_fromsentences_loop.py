import time
import numpy as np
import tqdm
import copy
from multiprocessing import Pool, Manager, freeze_support
import math
import myutils_htm
import pickle

def savepkl(filename, obj):
    pickle.dump(obj, open(filename, 'wb'))

timestr = time.strftime("%Y%m%d-%H%M%S")

def merge(k, size, layers, layer_dict, args):
    print("merge :", k, args)
    sentence_size = args[0]
    max_bucket = args[1]

    # input_layers = input_dict['layers']
    # input_layers_dict = input_dict['layers_dict']

    # startk = k
    # endk = k + size
    # if endk>len(input_layers):
    #     endk = len(input_layers)

    # layers = [layer for layer in input_layers[startk:endk]]
    print(k, " layers size : ", len(layers))

    # layer_dict = {str(i) : input_layers_dict[str(i + startk)] for i in range(endk-startk)}
    
    print(k, " layer_dict size : ", len(layer_dict.keys()))

    replace = 11
    count = 0
    while replace > 10:
        replace = 0
        i = 0
        while i < len(layers):
            replaced = -1
            stri = str(i)
            layer = layers[i]
            layers_length = np.array([l.count() for l in layers])
            allsum = np.array([(layer | l).count() for l in layers])
            # allsum = allR.sum(axis=1)
            allsum = np.array(((allsum * 2)/((layers_length + (layer.count())))))
            rlen = 30
            if len(allsum)<rlen:
                rlen = len(allsum)
            indexes = np.argpartition(-allsum, range(rlen))
            indexes = [id for id in indexes[:rlen] if id != i and layers_length[id]<(max_bucket * sentence_size *0.1) and allsum[id]>1.975]
            if len(indexes)>0:
                index = indexes[0]
                strindex = str(index)
                # index_value = allsum[index[0]]
                layers[i] = (layer | layers[index])
                # print(sum(layer), sum(layers[index]))
                
                layer_dict[stri].update(layer_dict[strindex])
                # layers = np.delete(layers, index, axis=0)
                del layers[index]
                
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

            if count%1000==0:
                print("run", sentence_size, max_bucket, k, i, replace, len(layers), time.strftime("%Y%m%d-%H%M%S"))
            
            count = count + 1

        # myutils_htm.savepkl('eval_sdr_layer1_sample_' + str(count) + '.pkl',  {
        #     "training_words_layer1": new_layers,
        #     "training_words_dict" : newlayer_dict
        # })

        print(k, 'new layer : ', replace, len(layers))

        myutils_htm.savepkl('training_frommergelayer1_layer2_seperated_remerge_' + str(sentence_size) + '_4096_'+ str(max_bucket) + '_' + str(0.1) + '_'+ str(1.975) + '_' + str(k) + '.pkl',  {
            "training_sentence_layer2": layers,
            "training_sentence_layer2_dict" : layer_dict
        })

    return layers, layer_dict

def custom_error_callback(error):
    print(f'Got an Error: {error}', flush=True)

sum_layer1_dict = {}
sum_layer1 = []
sum_layer1_length = 0
THREAD = 60
MERGE_SIZE = 40000
TRAIN_SIZE = 20000
if __name__ == '__main__':
    # freeze_support()
    pool=Pool(THREAD)
    manager = Manager()
    # input_dict = manager.dict()
    # output_dict = manager.dict()
    
    # sum_layer_length = 0
    # sum_layer1_dict = {}
    # sum_layer1 = []
    # for i in range(71):
        
    #     output_dict = pickle.load(open('eval_sdr_layer1_merge_' + str(i * TRAIN_SIZE) + '.pkl', 'rb'))

    #     newlayer_dict = output_dict['training_words_dict']
    #     new_layers = output_dict['training_words_layer1']
    #     print('dict length : ', i,len(newlayer_dict.keys()))
    #     print('layer length : ', i, len(new_layers))

    #     for j in newlayer_dict.keys():
    #         extend_j = int(j) + sum_layer_length
    #         sum_layer1_dict[str(extend_j)] = newlayer_dict[j]

    #     sum_layer1.extend(new_layers)
    #     sum_layer_length = sum_layer_length + len(new_layers)

    # print("merged size : ", sum_layer_length)

    # savepkl('eval_sdr_layer1_merge.pkl',  {
    #     "training_words_layer1": sum_layer1,
    #     "training_words_dict" : sum_layer1_dict
    # })
    max_buckets = [2048]
    sentence_sizes = [2]
    for sentence_size in sentence_sizes:
        for i, max_bucket in enumerate(max_buckets):
            print("running : ", sentence_size, max_bucket)
            eval_sdr_layer1_merge = myutils_htm.loadpkl('training_fromsentence_layer2_seperated_' + str(sentence_size)+ '_4096_' + str(max_bucket) + '.pkl')
            training_words_layer1 = eval_sdr_layer1_merge['training_sentence_layer2']
            training_words_dict = eval_sdr_layer1_merge['training_sentence_layer2_dict']
            print('sum size', len(training_words_layer1))
            ite = math.ceil(len(training_words_layer1)/MERGE_SIZE)
            myutils_htm.runmerge_threads(merge, training_words_layer1, training_words_dict, [sentence_size, max_bucket], pool, manager, len(training_words_layer1), MERGE_SIZE, THREAD, call_back_error=custom_error_callback)

        