import numpy as np
import time
import pickle
import copy

def savepkl(filename, obj):
    pickle.dump(obj, open(filename, 'wb'))

def loadpkl(filename):
    return pickle.load(open(filename, 'rb'))
    
def readlines(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return f.readlines()
def savelines(filename, lines):
    with open(filename, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(f"{line}\n")

def string2int(text, start_index = ord('a')):
    text = text.lower()
    return [ord(x) - start_index for x in text]

def subtext(text, index, length):
    t = text[index:index+length]
    if len(t)<length:
        return None
    else:
        return t
    
def subtext_list(text, length):
    index = 0
    list = []
    while True:
        t = subtext(text, index, length)
        if t == None:
            break
        list.append(t)
        index = index + 1
    return list

def text2hashindex(text, max_bucket = 1024):
    return hash(text) % max_bucket

def text2hashindexs(text_list, max_bucket = 1024):
    return [hash(text) % max_bucket for text in text_list]

def text2Representation(text, max_bucket = 1024):
    textllist = []
    for i in range(1,len(text) + 1):
        textllist.extend(subtext_list(text, i))
    text_representation_indexs = text2hashindexs(textllist, max_bucket)
    text_representation = np.zeros(max_bucket, dtype=bool)
    text_representation[text_representation_indexs] = True
    return text_representation

def text2Representation1(text, max_bucket = 1024):
    textllist = []
    for i in range(1,len(text) + 1):
        textllist.extend(subtext_list1(text, i))
    # print(textllist)
    text_representation_indexs = text2hashindexs(textllist, max_bucket)
    text_representation = np.zeros(max_bucket, dtype=bool)
    text_representation[text_representation_indexs] = True
    return text_representation

def subtext_list1(text, length):
    index = 0
    if length>1:
        text = (" " * (length - 1)) + text + (" " * (length - 1))
    list = []
    while True:
        t = subtext(text, index, length)
        if t == None:
            break
        list.append(t)
        index = index + 1
    return list



def run_threads(callback_async, input_dict, output_dict, pool, manager, length, size, THREAD, call_back_error = None, call_back_finished = None, call_back_checkpoint = None):
    queues = []    
    i = 0
    while i < length:
        
        j = 0
        # results = []
        while len(queues) < THREAD:
            r = pool.apply_async(callback_async, [i + j, size, input_dict, output_dict], error_callback = call_back_error)
            queues.append(r)
            print("add queue", i + j, len(queues))
            j = j + size
        i = i + j

        removed_threads = []
        while len(removed_threads) <= 0:
            z = 0
            while z < len(queues):
                try:
                    queue = queues[z]
                    if (queue.ready() and queue.successful()):
                        queue.get()
                        removed_threads.append(queue)
                except Exception:
                    continue
                finally:
                    z = z + 1

            if (len(removed_threads)>0):
                for r in removed_threads:
                    queues.remove(r)
                    print("remove queue", len(queues))
            else:
                time.sleep(0.1)
        
        if call_back_checkpoint is not None:
            call_back_checkpoint(output_dict)

    print("i : ", i, length)
    for r in queues:
        r.wait()

    if call_back_finished is not None:
        call_back_finished(output_dict)




def runmerge_threads(callback_async, input_layers, input_layers_dict, args, pool, manager, length, size, THREAD, call_back_error = None, call_back_finished = None, call_back_checkpoint = None):
    queues = []    
    i = 0
    while i < length:
        
        j = 0
        # results = []
        while len(queues) < THREAD:
            
            k = i + j
            startk = k
            endk = k + size
            if endk>len(input_layers):
                endk = len(input_layers)

            layers = copy.deepcopy(input_layers[startk:endk])
            print(k, " layers size : ", len(layers))

            layer_dict = {str(a) : input_layers_dict[str(a + startk)] for a in range(endk-startk)}

            r = pool.apply_async(callback_async, [i + j, size, layers, layer_dict, args], error_callback = call_back_error)
            queues.append(r)
            print("add queue", i + j, len(queues))
            j = j + size
        i = i + j

        removed_threads = []
        while len(removed_threads) <= 0:
            z = 0
            while z < len(queues):
                try:
                    queue = queues[z]
                    if (queue.ready() and queue.successful()):
                        queue.get()
                        removed_threads.append(queue)
                except Exception:
                    continue
                finally:
                    z = z + 1

            if (len(removed_threads)>0):
                for r in removed_threads:
                    queues.remove(r)
                    print("remove queue", len(queues))
            else:
                time.sleep(0.1)
        
        if call_back_checkpoint is not None:
            call_back_checkpoint()

    print("i : ", i, length)
    for r in queues:
        r.wait()

    if call_back_finished is not None:
        call_back_finished()



def runwait_threads(callback_async, args, pool, THREAD, call_back_error = None, call_back_finished = None, call_back_checkpoint = None):
    queues = []
    for i in range(THREAD):
        queues.append(pool.apply_async(callback_async, args[i], error_callback = call_back_error))
    
    results = []
    for q in queues:
        q.wait()
        results.append(q.get())

    return results
