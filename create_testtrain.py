

def readlines(filename):
    with open(filename, 'r') as f:
        return f.readlines()

def savelines(filename, lines):
    with open(filename, 'w') as f:
        for line in lines:
            f.write(f"{line}\n")

ground_lines = readlines('full_sentences_ground.txt')
input_lines = readlines('full_sentences_input.txt')
sumword = 0
train_precent = 0.8

for i, ground_txt in enumerate(ground_lines):
    gwords = ground_txt.split(' ')
    sumword = sumword + len(gwords)
sumnallword = sumword

sumword = 0
train_ground_lines = []
test_ground_lines = []
train_input_lines = []
test_input_lines = []
for i, ground_txt in enumerate(ground_lines):
    input_txt = input_lines[i]
    gwords = ground_txt.split(' ')
    sumword = sumword + len(gwords)
    if sumword/sumnallword<train_precent:
        train_ground_lines.append(ground_txt)
        train_input_lines.append(input_txt)
    else:
        test_ground_lines.append(ground_txt)
        test_input_lines.append(input_txt)

savelines('full_sentences_ground_train80.txt', train_ground_lines)
savelines('full_sentences_input_train80.txt', train_input_lines)
savelines('full_sentences_ground_test80.txt', test_ground_lines)
savelines('full_sentences_input_test80.txt', test_input_lines)