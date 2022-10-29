import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from tensorflow.python.client import device_lib
import myutils
import re
import myutils_htm
import tqdm

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
ground_words = train_test_set['train_ground_words']
input_words = train_test_set['train_input_words']
ground_words = ground_words[:100000]
input_words = input_words[:100000]

pattern_word = re.compile("^([a-zA-Z0-9]+)$")
char_set = list(" abcdefghijklmnopqrstuvwxyz0123456789\r\n\t?")

input_texts = replacechar(input_words, "?")
target_texts = replacechar(ground_words, "?")

char2int = { char_set[x]:x for x in range(len(char_set)) }
int2char = { char2int[x]:x for x in char_set }

REPEAT_FACTOR = 1


def prefixword(x):
    return '\t' + x + '\n'
def searchWord(x):
    if len(x)>3 and pattern_word.search(x):
        return True
    else:
        return False

target_texts = [prefixword(w) for w in target_texts]

print("LEN OF SAMPLES:",len(input_texts))

max_enc_len = max([len(x) for x in input_texts])
max_dec_len = max([len(x) for x in target_texts])
print("Max Enc Len:",max_enc_len)
print("Max Dec Len:",max_dec_len)

num_samples = len(input_texts)
encoder_input_data = np.zeros( (num_samples , max_enc_len , len(char_set)),dtype='float32' )
decoder_input_data = np.zeros( (num_samples , max_dec_len , len(char_set)+2),dtype='float32' )
decoder_target_data = np.zeros( (num_samples , max_dec_len , len(char_set)+2),dtype='float32' )
print("CREATED ZERO VECTORS")

#filling in the enc,dec datas
for i,(input_text,target_text) in enumerate(zip(input_texts, target_texts)):
    for t,char in enumerate(input_text):
        encoder_input_data[ i , t , char2int[char] ] = 1
    for t,char in enumerate(target_text):
        decoder_input_data[ i, t , char2int[char] ] = 1
        if t > 0 :
            decoder_target_data[ i , t-1 , char2int[char] ] = 1
print("COMPLETED...")

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

import time
timestr = time.strftime("%Y%m%d-%H%M%S")

import tensorflow.keras
from tensorflow.keras.callbacks import ModelCheckpoint, Callback


epochfile = "epoch.best.json"
class CustomSaver(tensorflow.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (epoch%10 == 0):
            print('epoch = ' + str(epoch))
            myutils.savejson(epochfile, { 'epoch' : epoch})

modelfile = "weights.lastest.hdf5"
checkpoint = ModelCheckpoint(modelfile, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=10, save_freq='epoch')
callbacks_list = [CustomSaver(), checkpoint]


new = True
model = Model([encoder_inputs,decoder_inputs],decoder_ouputs)
import os.path
if not new and os.path.isfile(modelfile):
    model.load_weights(modelfile)


if os.path.isfile(epochfile):
    if not new:
        epoch = myutils.readjson(epochfile)['epoch']
        epochs -= epoch

model.compile(optimizer='rmsprop',loss='categorical_crossentropy')
model.summary()
h=model.fit([encoder_input_data,decoder_input_data],decoder_target_data
         ,epochs = epochs,
          batch_size = batch_size,
          validation_split = 0.2, 
          callbacks=callbacks_list,
          verbose=1
         )

# model.save('s2s_trec5_' + timestr + '.h5')
from tensorflow import keras
model.save('s2s_trec5_' + timestr + '.h5')
# model = keras.models.load_model('s2s_trec5.h5')
plt.plot(h.history['loss'])
plt.title('Model Loss')
# plt.show()
plt.savefig('loss_' + timestr + '.png')


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

encoder_model.save('encoder_trec5_' + timestr + '.h5')
decoder_model.save('decoder_trec5_' + timestr + '.h5')
# encoder_model = keras.models.load_model('encoder_trec5.h5')
# decoder_model = keras.models.load_model('decoder_trec5.h5')


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



# myutils.savejson('result_lstm_' + timestr + '.json', { 'results' : results, 'errors' :  errors})