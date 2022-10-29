import myutils_htm
import tqdm


ground_file_train = 'full_degrade5_ground_a_train80.txt'
input_file_train = 'full_degrade5_input_a_train80.txt'
char_set = set()

ground_train_lines = myutils_htm.readlines(ground_file_train)
input_train_lines = myutils_htm.readlines(input_file_train)
input_train_words = []
ground_train_words = []
for i, ground_line in tqdm.tqdm(enumerate(ground_train_lines)):
    input_line = input_train_lines[i]
    gwords  =ground_line.strip().split(' ')
    iwords = input_line.strip().split(' ')
    if len(gwords) == len(iwords):
        input_train_words.extend(iwords)
        char_set.update([c for w in iwords for c in w])
        ground_train_words.extend(gwords)


ground_file_test = 'full_degrade5_ground_a_test80.txt'
input_file_test = 'full_degrade5_input_a_test80.txt'
ground_test_lines = myutils_htm.readlines(ground_file_test)
input_test_lines = myutils_htm.readlines(input_file_test)

input_test_words = []
ground_test_words = []
for i, ground_line in tqdm.tqdm(enumerate(ground_test_lines)):
    input_line = input_test_lines[i]
    gwords  =ground_line.strip().split(' ')
    iwords = input_line.strip().split(' ')
    if len(gwords) == len(iwords):
        input_test_words.extend(iwords)
        ground_test_words.extend(gwords)

print(list(char_set))
myutils_htm.savepkl('train_test_a.pkl',  {
    "char_set": char_set,
    "train_input_words" : input_train_words,
    "train_ground_words" : ground_train_words,
    "input_test_words" : input_test_words,
    "ground_test_words" : ground_test_words
})