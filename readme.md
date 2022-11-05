# File Description

**create_trec-5a.py**

processed trec-5 to trec5_degrade20_processed.txt and trec5_degrade5_processed (process from  TREC-5 raw data) download processed files from https://drive.google.com/drive/folders/1-ACRx8X94JGbHU0tmSKsFhmLYZdX6MzT?usp=share_link

**create_testtrain.py**

separated train and test data, full_degrade5_ground_a_train80.txt, full_degrade5_input_a_train80.txt, full_degrade5_ground_a_test80.txt, full_degrade5_input_a_test80.txt

**create_testdata.py**
saved train and test data to a pkl file, train_test_a.pkl

**eval_dh.py**
evaluation word level by using Levenshtein Distance, it read data from  train_test_a.pkl file

**eval_pyspell.py**
evaluation word level by using pyspellchecker, it read data from  train_test_a.pkl file

**train_lstm.py**
training word level by using  LSTM, it read data from  train_test_a.pkl file

**eval_lstm.py**
evaluation word level by using LSTM, it read data from  train_test_a.pkl file

**train_neuspell_layer1.py**
training word level by using  Neuspell (SC-LSTM plus ELMO (at input)), the data is generated to files to train neuspell in word level by using  full_degrade5_ground_a_train80.txt and  full_degrade5_input_a_train80.txt

**eval_neuspell_layer1.py**
evaluation word level by using  Neuspell (SC-LSTM plus ELMO (at input)), it read data from  train_test_a.pkl file

**create_sdr_layer1.py**
training correct words to the new method, cognition-inspired  , it read data from  train_test_a.pkl

**eval_sdr_layer1.py**
evaluation word level by using the new method, cognition-inspired, it read data from  train_test_a.pkl file

**remerge_thread_layer2_fromsentences_loop.py**
Merging vectors with parallel processing, it can apply for both layer 1 and 2. setting parameters, MAX_BUCKET, sparsity, w

**remerge.py**
remerge the result from remerge_thread_layer2_fromsentences_loop.py

**train_neuspell_layer2.py**
training sentence level by using  Neuspell (SC-LSTM plus ELMO (at input)), it read data from  full_degrade5_ground_a_train80.txt and  full_degrade5_input_a_train80.txt

**eval_neuspell_layer2.py**
evaluation sentence level by using using  Neuspell (SC-LSTM plus ELMO (at input)). It read data from   full_degrade5_ground_a_train80.txt, full_degrade5_input_a_train80.txt, full_degrade5_ground_a_test80.txt, full_degrade5_input_a_test80.txt

**create_sdr_layer2_fromlayer1_seperated.py**
Create vectors in sentence level, concatenate SDR vectors from layer 1 or word level. Then it can merge by using remerge_thread_layer2_fromsentences_loop.py

**eval_sdr_layer2.py**
evaluation sentence level by using using the new method, cognition-inspired. It read data from   full_degrade5_ground_a_train80.txt, full_degrade5_input_a_train80.txt, full_degrade5_ground_a_test80.txt, full_degrade5_input_a_test80.txt and vector from create_sdr_layer2_fromlayer1_seperated.py or merging vectors from remerge.py

**eval_sdr_layer2_process.py**
evaluation sentence level by using using the new method, cognition-inspired. It is the same as eval_sdr_layer2.py but do parallel processing.

Please set PYTHONHASHSEED=0 to prevent hash changing
