import myutils_htm
import tqdm
from time import perf_counter
import sys

from spellchecker import SpellChecker
spell = SpellChecker()
def predict(word):
    return spell.candidates(word)


max_cadidate = 0
cadidates = []
def test(input_texts, target_texts):
    count, error, countdiff, errordiff, countlimit = 0, 0, 0, 0, 0
    timeperfsum = 0
    perf = 0.0
    
    for index, word in tqdm.tqdm(enumerate(input_texts)):
        word = word.strip()
        t1_start = perf_counter()
        candidates = list(spell.candidates(word))
        corrected_text = spell.correction(word)
        candidates = candidates[:30]
        t1_stop = perf_counter()
        timeperfsum += (t1_stop-t1_start)
        input_text = word
        ground_text = target_texts[index].strip()
        
        if (not ground_text == corrected_text):
            if ground_text in candidates:
                countlimit += 1
            error += 1
            errors.append([input_text, corrected_text, ground_text])
            perf = float(error)/count

        if not input_text == ground_text:
            countdiff += 1
            if not ground_text == corrected_text:
                errordiff += 1
        
        if index%100 == 0 and index != 0:
            print('error = ' + str(error) + ' count = ' + str(count) + ' countlimit = ' + str(countlimit) + ' time = ' + str(timeperfsum/count))
        count += 1
    return [count, error, countlimit, countdiff, errordiff, timeperfsum]


errors = []
if __name__ == '__main__':
    # print("Training Size:", len(training_words))

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

    args = sys.argv

    train_test_set = myutils_htm.loadpkl('train_test_a.pkl')
    input_train_words = train_test_set['train_input_words']
    ground_train_words = train_test_set['train_ground_words']
    input_test_words = train_test_set['input_test_words']
    ground_test_words = train_test_set['ground_test_words']

    if len(args)>=2:
        sample = int(args[1])
        input_test_words = input_test_words[:sample]
        ground_train_words = ground_train_words[:sample]
        ground_test_words = ground_test_words[:sample]
        input_train_words = input_train_words[:sample]
    

    print("Testing size : ", len(input_test_words))
    result = test(input_test_words, ground_test_words)
    print(result)

