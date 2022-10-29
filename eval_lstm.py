import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from tensorflow.python.client import device_lib
import re
import inspect
import sys
import tqdm

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import myutils_htm

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

char_set = list(" abcdefghijklmnopqrstuvwxyz0123456789\r\n\t?")


char2int = { char_set[x]:x for x in range(len(char_set)) }
int2char = { char2int[x]:x for x in char_set }

def replacechar(words, replace_char):
    new_words = []
    for word in tqdm.tqdm(words):
        w = ""
        for c in word:
            if c in char_set:
                w = w + c
            else:
                w = w + replace_char
        new_words.append(w)
        # print(word, w)
    return new_words

train_test_set = myutils_htm.loadpkl('../train_test_a.pkl')
ground_train_words = train_test_set['train_ground_words']
input_train_words = train_test_set['train_input_words']
input_test_words = train_test_set['input_test_words']
ground_test_words = train_test_set['ground_test_words']

ground_words = ground_test_words[:100000]
input_words = input_test_words[:100000]

input_words = replacechar(input_words, "?")
ground_words = replacechar(ground_words, "?")

input_texts = input_words
target_texts = ground_words

max_enc_len = max([len(x) for x in input_texts])
max_dec_len = max([len(x) for x in target_texts])

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,LSTM,Dense
batch_size = 128
epochs = 100
latent_dim = 256

num_enc_tokens = len(char_set)
num_dec_tokens = len(char_set) + 2 # includes \n \t
encoder_inputs = Input(shape=(None,num_enc_tokens))
encoder = LSTM(latent_dim,return_state=True)
encoder_outputs , state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h,state_c]
decoder_inputs = Input(shape=(None,num_dec_tokens))
decoder_lstm = LSTM(latent_dim,return_sequences=True,return_state=True)
decoder_ouputs,_,_ = decoder_lstm(decoder_inputs,initial_state = encoder_states)

decoder_dense = Dense(num_dec_tokens, activation='softmax')
decoder_ouputs = decoder_dense(decoder_ouputs)

model = Model([encoder_inputs,decoder_inputs],decoder_ouputs)
model.compile(optimizer='rmsprop',loss='categorical_crossentropy')
model.summary()
# h=model.fit([encoder_input_data,decoder_input_data],decoder_target_data
#          ,epochs = epochs,
#           batch_size = batch_size,
#           validation_split = 0.2
#          )

import time
timestr = time.strftime("%Y%m%d-%H%M%S")
# model.save('s2s_trec5_' + timestr + '.h5')
from tensorflow import keras
# model = keras.models.load_model('s2s_trec5_20210526-154811.h5')
# plt.plot(h.history['loss'])
# plt.title('Model Loss')
# plt.show()


encoder_model = Model(encoder_inputs,encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h,decoder_state_input_c]
decoder_outputs,state_h,state_c = decoder_lstm(
        decoder_inputs,initial_state = decoder_states_inputs
)
decoder_states = [state_h,state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states
)

# encoder_model.save('encoder_trec5_' + timestr + '.h5')
# decoder_model.save('decoder_trec5_' + timestr + '.h5')
encoder_model = keras.models.load_model('encoder_trec5_20221028-121730.h5')
decoder_model = keras.models.load_model('decoder_trec5_20221028-121730.h5')

def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1, num_dec_tokens))
    target_seq[0, 0, char2int['\t']] = 1.
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = int2char[sampled_token_index]
        decoded_sentence += sampled_char
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_dec_len):
            stop_condition = True
        target_seq = np.zeros((1, 1, num_dec_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence


def encode(input_texts, target_texts):
    max_enc_len = max([len(x) for x in input_texts])
    num_samples = len(input_texts)
    encoder_input_data = np.zeros( (num_samples , max_enc_len , len(char_set)),dtype='float32' )
    for i,input_text in enumerate(input_texts):
        for t,char in enumerate(input_text):
            encoder_input_data[ i , t , char2int[char] ] = 1
    return [encoder_input_data, input_texts, target_texts]

import tqdm
from time import perf_counter_ns, perf_counter
errors = []
def test(input_texts, target_texts):
    count, error, countdiff, errordiff = 0, 0, 0, 0
    [testdata, input_texts, target_texts] = encode(input_texts, target_texts)
    timeperfsum = 0
    perf = 0.0
    for seq_index in tqdm.tqdm(range(len(testdata))):
        input_seq = testdata[seq_index: seq_index + 1]
        t1_start = perf_counter()
        decoded_sentence = decode_sequence(input_seq)
        t1_stop = perf_counter()
        timeperfsum += (t1_stop-t1_start)
        input_text = input_texts[seq_index].strip()
        corrected_text = decoded_sentence.strip()
        ground_text = target_texts[seq_index].strip()
        count += 1
        # print('Wrong sentence:', input_text)
        # print('Corrected sentence:', corrected_text)
        # print('Ground Truth:', ground_text)
        if not ground_text == corrected_text:
            error += 1
            errors.append([input_text, corrected_text, ground_text])
            perf = float(error)/count
        if not input_text == ground_text:
            countdiff += 1
            if not ground_text == corrected_text:
                errordiff += 1
        if seq_index%100 == 0:
            print('error = ' + str(error) + " count = " + str(count)  + " perf = " + str(timeperfsum/count))
    return [count, error, countdiff, errordiff, timeperfsum]
# input_texts = input_words
# target_texts = ground_words
result = test(input_texts, target_texts)

