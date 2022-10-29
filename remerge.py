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

MAX_BUCKET = 1024
timestr = time.strftime("%Y%m%d-%H%M%S")

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
            indexes = [id for id in indexes[:rlen] if id != i and layers_length[id]<MAX_BUCKET*0.1 and allsum[id]>0.3]
            
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

            if count%1000==0:
                print("run", i, replace, len(layers), time.strftime("%Y%m%d-%H%M%S"))
            
            count = count + 1


    myutils_htm.savepkl('training_frommergelayer1_layer2_seperated_remerge_2_4096_2048_0.1_1.975.pkl',  {
        "training_sentence_layer2": new_layers,
        "training_sentence_layer2_dict" : newlayer_dict
    })

    return new_layers, newlayer_dict




SIZE = 40000
sum_layer_length = 0
sum_layer1_dict = {}
sum_layer1 = []
for i in range(64 + 1):
    
    output_dict = pickle.load(open('training_frommergelayer1_layer2_seperated_remerge_2_4096_2048_0.1_1.975_' + str(i * SIZE) + '.pkl', 'rb'))

    newlayer_dict = output_dict['training_sentence_layer2_dict']
    new_layers = output_dict['training_sentence_layer2']
    print('dict length : ', i,len(newlayer_dict.keys()))
    print('layer length : ', i, len(new_layers))

    for j in newlayer_dict.keys():
        extend_j = int(j) + sum_layer_length
        sum_layer1_dict[str(extend_j)] = newlayer_dict[j]

    sum_layer1.extend(new_layers)
    sum_layer_length = sum_layer_length + len(new_layers)

print("merged size : ", sum_layer_length)

savepkl('training_frommergelayer1_layer2_seperated_remerge_2_4096_2048_0.1_1.975.pkl',  {
    "training_sentence_layer2": sum_layer1,
    "training_sentence_layer2_dict" : sum_layer1_dict
})


print('sum size', sum_layer_length)


# eval_sdr_layer1_merge = myutils_htm.loadpkl('eval_sdr_layer1_remerge_2048_0.1_0.3.pkl')
# training_words_layer1 = eval_sdr_layer1_merge['training_words_layer1']
# training_words_dict = eval_sdr_layer1_merge['training_words_dict']
# merge(np.array(training_words_layer1), training_words_dict)