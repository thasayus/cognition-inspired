from pathlib import Path
import gzip
import re
import html
import re
import collections
# from nltk.corpus import stopwords
# from string import punctuation
# from spacy.lang.en import English
# from keras.preprocessing.text import text_to_word_sequence
import json
import time
from time import perf_counter_ns
# import pandas as pd
import tqdm
from multiprocessing import Pool, Manager, freeze_support
import logging
# logging.basicConfig(filename="log.txt", level=logging.DEBUG)

# logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
# logger = logging.getLogger()

# fileHandler = logging.FileHandler("thread.log")
# fileHandler.setFormatter(logFormatter)
# logger.addHandler(fileHandler)
# consoleHandler = logging.StreamHandler()
# consoleHandler.setFormatter(logFormatter)
# logger.addHandler(consoleHandler)


TREC5_PATH = './data/confusion_track/'
CORRECT_DATA_PATH = "original"
NOISE5_DATA_PATH = "degrade5"
NOISE20_DATA_PATH = "degrade20"
DATA_PATH = [CORRECT_DATA_PATH, NOISE5_DATA_PATH, NOISE20_DATA_PATH]

def readjson(filename):
    with open(filename) as json_file:
        return json.load(json_file)
    return None

def savejson(filename, object):
    with open(filename, 'w') as outfile:
        json.dump(object, outfile)

def readlines(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return f.readlines()

def savelines(filename, lines):
    with open(filename, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(f"{line}\n")

def nested_dict():
    return collections.defaultdict(nested_dict)

def readGZIP(filename):
    f=gzip.open(filename,'rb')
    input = f.read()
    return input

def listFiles(path, filter='*.*'):
    return list(Path(path).rglob(filter))

def search(pattern, line, index=1):
    search = re.search(pattern, line)
    if search:
        return search.group(index)
def findtag(text):
    tags = []
    data = []
    found = False
    for m in re.finditer(r'<(/*\w+)>', text):
        tag = m.group(0)
        # print(tag)
        if r'/' in tag:
            if tags[-1] == tag.replace('/',''):
                tags.pop(-1)
                if len(tags) <= 0:
                    end = m.start()
                    # print('end>' + tag)
                    data.append([ftag, start, end])
                    found = False
        else:
            tags.append(m.group(0))
        if not found and not r'/' in tag:
            found = True
            ftag = tag
            start = m.end()
            # print('start>' + tag)

    return data

def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def countSameChars(str1, str2): 
    c, j = 0, 0
    for i in str1:    
        if str2.find(i)>= 0 and j == str1.find(i): 
            c += 1
        j += 1
    return c

def findX(sentense, ci, cl):
    i = ci
    if ci + cl > len(sentense):
        return None
    else:
        return ci + cl

def findWord(word, nword, de5_sentences, current_row, current_index):
    i = current_row
    count = 0
    dwords = []
    line = ''
    lines = []
    if word == 'a' and nword == '':
        print(1)
    while i < len(de5_sentences):
        find = False
        dtokens = word_tokenize(de5_sentences[i])
        if i == current_row:
            j = current_index
        else:
            j = 0
        while j<len(dtokens):
            dtoken = dtokens[j]
            line = line + dtoken
            lines.append(dtoken)
            if count>10:
                find = True
                break
            dist = levenshteinDistance(word, line.strip())

            if (len(word)>3 and dist < 3):
                dwords.append([dist, line.strip(), lines[:], i, j, word])
                find = True
                break
            else:
                dist1 = levenshteinDistance(word, dtoken)
                if len(word)<=2 and dist1 <= 0:
                    dist2 = None
                    ndtoken = None
                    if nword is not None:
                        xj = findX(dtokens, j, 1)
                        if xj is not None:
                            ndtoken = dtokens[xj]
                        if ndtoken is not None:
                            dist2 = levenshteinDistance(nword, ndtoken)
                        if dist2 is not None and dist2<(len(nword) - len(nword)/2):
                            dwords.append([dist1, dtoken.strip(), lines[:], i, j, word])
                            find = True
                            break
                dwords.append([dist, line.strip(), lines[:], i, j, word])
            
            j += 1
            count += 1
        i += 1
        if find:
            break
    if len(dwords) <= 1:
        return dwords[0]
    else:
        min = 999
        mini = 0
        count = 0
        countms = []
        for dword in dwords:
            if min == dword[0]:
                countms.append(count)
            if min>dword[0]:
                min = dword[0]
                mini = count
                countms = [mini]
            count += 1
        if len(countms)>1:
            maxj = 0
            max = -1
            for countm in countms:
                cs = countSameChars(word, dwords[countm][1])
                if cs>max:
                    max = cs
                    maxj = countm
            mini = maxj
        
        return dwords[mini]

def findXY(sentenses, ci, cj, cl):
    i = ci
    j = cj
    while i < len(sentenses):
        le = len(sentenses[i])
        if (le-j)-cl<=0:
            if i+1 >= len(sentenses):
                return None
            i = i + 1
            return findXY(sentenses, i, 0, cl - (le-j))
        else:
            j = j + cl
            return [i , j]

def word_tokenize(sentence):
    # stoplist = set(stopwords.words('english') + list(punctuation))
    # stoplist = set(['.', ':'])
    # return [token for token in nltk.word_tokenize(sentence) if token.lower() not in stoplist]
    return sentence.split(' ')# text_to_word_sequence(sentence, lower=False)


def verifySentense(orig_sentences, de5_sentences):
    aorgin_sentences = []
    ade5_sentences = []
    current_row = 0
    current_index = 0
    count_error = 0
    for orig_sentence in orig_sentences:
        otokens = word_tokenize(orig_sentence)
        aorgin_sentences.append(otokens)

    i = 0
    while i < len(aorgin_sentences):
        j = 0
        while j < len(aorgin_sentences[i]):
            otoken = aorgin_sentences[i][j]
            dwords = []
            aorigin_line = ''
            for k in range(8):
                xy = findXY(aorgin_sentences, i, j, k)
                nxy = findXY(aorgin_sentences, i, j, k+1)
                nword = None
                if nxy:
                    nword = aorgin_sentences[nxy[0]][nxy[1]]
                if xy:
                    aorigin_line  = aorigin_line + aorgin_sentences[xy[0]][xy[1]]
                    dwords.append(findWord(aorigin_line.strip(), nword, de5_sentences, current_row, current_index))
            min = 9999
            mini = 0
            for k in range(len(dwords)):
                if dwords[k][0]<min:
                    min = dwords[k][0]
                    mini = k
            dword = dwords[mini]
            pi = i
            pj = j
            if mini>0:
                otoken = ''
                for k in range(mini):
                    ade5_sentences.append([])
                    # print(':' + aorgin_sentences[i][j+k] + '=>' + ','.join([]))
                    otoken = otoken + aorgin_sentences[i][j+k]
                xy = findXY(aorgin_sentences, i, j, mini)
                i = xy[0]
                j = xy[1]
                otoken = otoken + aorgin_sentences[i][j]

            dtokens = word_tokenize(de5_sentences[dword[3]])
            if (dword[4]+1)>len(dtokens):
                current_row = dword[3] + 1
                current_index = 0
            else:
                current_index = dword[4]+1
            # print(otoken + '=>' + ','.join(dword[2]))
            if len(otoken)>=3:
                dist = levenshteinDistance(otoken, ''.join(dword[2]))
                if dist>5:
                    count_error += 1
                    if count_error < 5:
                        xy = findXY(aorgin_sentences, pi, pj, mini)
                        # print('error,' + aorgin_sentences[i][j] + '=>' + ','.join(dword[2]))
                    else:
                        return [None, None, count_error]

            ade5_sentences.append(dword[2])
            j += 1
        i += 1
    return [aorgin_sentences, ade5_sentences, count_error]

def processtext1(text):
    text = re.sub(r'(<!--.*?-->)', '\n', text).strip()
    tags = findtag(text)
    data = []
    if tags:
        currentindex = 0
        for tag in tags:
            if currentindex<(tag[1] - len(tag[0])):
                atext = text[currentindex:(tag[1] - len(tag[0]))]
                atext = atext.strip()
                if atext:
                    atext = re.sub(r'\n\s*\n', '\n', atext)
                    # atext = re.sub(r'\r\n', ' ', atext)
                    # atext = re.sub(r'\n', ' ', atext)
                    data.append(atext + "\n")

            ntext = text[tag[1]:tag[2]]
            ntext = ntext.strip()
            if ntext:
                ntext = re.sub(r'\n\s*\n', '\n', ntext)
                # ntext = re.sub(r'\r\n', ' ', ntext)
                # ntext = re.sub(r'\n', ' ', ntext)
                ar = processtext1(ntext + "\n")
                if ar:
                    data.extend(ar)
            currentindex = tag[2] + len(tag[0]) + 1
    else:
        text = text.strip()
        text = re.sub(r'\n\s*\n', '\n', text)
        # text = re.sub(r'\r\n', ' ', text)
        # text = re.sub(r'\n', ' ', text)
        if text:
            return [text + "\n"]
        else:
            return ["\n"]
    return data

def replaceESC(data):
    data = re.sub(r'&hyph;', '-', data)
    data = re.sub(r'&Gacute;', 'G', data)
    data = re.sub(r'&pacute;', 'p', data)
    data = re.sub(r'&Kuml;', 'K', data)
    data = re.sub(r'&blank;', ' ', data)
    data = re.sub(r'&G;', 'G', data)
    data = re.sub(r'&B;', 'B', data)
    data = re.sub(r'&I;', 'I', data)
    data = re.sub(r'&L;', 'L', data)
    data = re.sub(r'&D;', 'D', data)
    data = re.sub(r'&T;', 'T', data)
    data = re.sub(r'&S;', 'S', data)
    data = re.sub(r'&F;', 'F', data)
    data = re.sub(r'&P;', 'P', data)
    data = re.sub(r'&2;', '2', data)
    data = re.sub(u'\u2423', ' ', data)
    # data = re.sub(r' +', ' ', data)
    return data

def readTREC5file(filename):
    content = readGZIP(filename)
    content = html.unescape(content.decode("ascii"))
    state = None
    scan_data = nested_dict()
    for line in content.splitlines():
        if '<DOC>' in line:
            state = 'doc'
            data = ''
            docno = None
            parent = None
        if state != None:
            tag = search(r'<(\w+)>', line)
            endtag = search(r'</(\w+)>', line)
            if state == 'data':
                if endtag == 'TEXT':
                    state = 'doc'
                else:
                    # line1 = re.sub(r'(<!--.*?-->)', ' ', line)
                    # line1 = line1.strip()
                    # if line1:
                    data = data + line + "\n"
            else:
                if endtag == 'DOC':
                    state = None
                    # print(data)
                    if docno:
                        adata = processtext1(replaceESC(data))
                        # for ad in adata:
                        #     for m in re.finditer(r'(&\w+;)', ad):
                        #         escpe = m.group(0)
                        #         escepses.add(escpe)
                        # print(adata)
                        # if count>10:
                        #     sys.exit()
                        # count = count + 1
                        scan_data[docno]['parent'] = parent
                        scan_data[docno]['text'] = adata
                    else:
                        print('error no docno')
                elif tag == 'DOCNO':
                    docno = search(r'<DOCNO>(.*?)</DOCNO>', line)
                    docno = docno.strip()
                elif tag == 'PARENT':
                    parent = search(r'<PARENT>(.*?)</PARENT>', line)
                    parent = parent.strip()
                elif tag == 'TEXT':
                    state = 'data'
    return scan_data

def nested_dict():
    return collections.defaultdict(nested_dict)

def create_linefiles():
    correct_files = listFiles(TREC5_PATH + DATA_PATH[0], "*.gz")
    original_lines = []
    de5_lines = []
    de20_lines = []
    for file in tqdm.tqdm(correct_files):
        orig_filename = str(file)
        
        try:
            # print(orig_filename)
            if 'fr941202.2.gz' in orig_filename:
                continue
            orig_data = readTREC5file(orig_filename)
            de5_filename = re.sub(r'original', DATA_PATH[1], orig_filename)
            # print(de5_filename)
            de5_data = readTREC5file(de5_filename)
            de20_filename = re.sub(r'original', DATA_PATH[2], orig_filename)
            # print(de20_filename)
            de20_data = readTREC5file(de20_filename)
        except Exception as e:
            print(e)
            continue
        
        for docno in orig_data.keys():
            text = ' '.join(orig_data[docno]['text'])
            text = re.sub(r' +', ' ', text)
            text = re.sub(r'\r\n', '#', text)
            text = re.sub(r'\n', '#', text)
            original_lines.append(text)
            text = ' '.join(de5_data[docno]['text'])
            text = re.sub(r' +', ' ', text)
            text = re.sub(r'\r\n', '#', text)
            text = re.sub(r'\n', '#', text)
            de5_lines.append(text)
            text = ' '.join(de20_data[docno]['text'])
            text = re.sub(r' +', ' ', text)
            text = re.sub(r'\r\n', '#', text)
            text = re.sub(r'\n', '#', text)
            de20_lines.append(text)

        # break

    savelines('trec5_original.txt', original_lines)
    savelines('trec5_degrade5.txt', de5_lines)
    savelines('trec5_degrade20.txt', de20_lines)


# distance_dict = readjson('distance_dict.json') # nested_dict()
def process_trec5_asyc(lineNo, lines, de5words, original_newlines):
    j = 0
    for linen2, line in enumerate(lines):
        line = line.strip()
        words = line.split(' ') # word_tokenize(line)
        words = [w for w in words if w.strip() != '']
        degrade5_words = []

        if len(words)<=0:
            continue

        sum_length = 0
        sum_distance = 0
        for k, word in enumerate(words):

            min_value = 9999
            min_word = ""
            min_i = 0
            minj = 9999

            sum_length = sum_length + len(word)
            found = False
            search_length = SEARCH_LENGTH
            while found == False:

                startj = j - search_length
                endj = j + search_length
                if startj < 0 :
                    startj = 0
                if endj > len(de5words):
                    endj = len(de5words)
                
                # if len(word)>2:
                expected = 2
                # if len(word)>3:
                #     expected = len(word) * 2 / 3

                if search_length == len(de5words):
                    expected = 9999

                for ia, dword in enumerate(de5words[startj:endj]): #
                    
                    d = levenshteinDistance(word, dword) # * (abs(len(dword) - len(word))+1)/(2+abs(len(dword) - len(word)))
                    if (len(word)>4 and d>=1) or (len(word)<=4) or search_length == len(de5words):
                        if (d <= min_value) and d < expected:
                            
                            if d < min_value:
                                minj = 9999
                            
                            if abs((ia + startj) - (j)) < minj:
                                min_value = d
                                min_word = dword
                                min_i = startj + ia
                                minj = abs((ia + startj) - (j))
                
                if min_word != "":
                    sum_distance = sum_distance + min_value
                    degrade5_words.append(min_word)
                    found = True
                    j = min_i + 1
                else:
                    if len(word)<3:
                        degrade5_words.append(word)
                        # print("error not found!! 3 : ", word)
                        found = True
                        j = j + 1
                    else:
                        if search_length == len(de5words):
                            # print("error not found!!", word)
                            print("error not found!!", word)
                            degrade5_words.append(word)
                            found = True
                            j = j + 1
                        else:
                            search_length = len(de5words)
                    
        if len(degrade5_words)>0 and len(words) == len(degrade5_words):
            original_newlines.append(str(lineNo) + "||" + str(linen2) + "||" + '##'.join(words) + "||" + '##'.join(degrade5_words))
        else:
            print('error de != original')

THREAD = 30
SEARCH_LENGTH = 50

queues = []
def process_trec5(pool):

    original_newlines = manager.list()
    original_lines = readlines('trec5_original.txt')
    degrade5_lines = readlines('trec5_degrade20.txt')
    
    i = 0
    while i < len(original_lines):

        print(i)
        j = 0
        # results = []
        while len(queues) < THREAD:
            original_line = original_lines[i + j]
            degrade5_line = degrade5_lines[i + j]
            original_line = original_line.strip()
            degrade5_line = degrade5_line.strip()
            degrade5_line = re.sub(r'#', ' ', degrade5_line)
            de5words = degrade5_line.split(' ') #
            de5words = [w for w in de5words if w.strip() != '']
            lines = original_line.split('#')
            
            r = pool.apply_async(process_trec5_asyc, [i, lines, de5words, original_newlines])
            
            queues.append(r)

            j = j + 1
        
        i = i + j


        removed_threads = []
        while len(removed_threads) <= 0:
            z = 0
            while z < len(queues):
                try:
                    queue = queues[z]
                    if (queue.ready() and queue.successful()):
                        removed_threads.append(queue)
                except Exception:
                    continue
                finally:
                    z = z + 1

            if (len(removed_threads)>0):
                for r in removed_threads:
                    queues.remove(r)
            else:
                time.sleep(0.1)
        
        # if i > 10:
        #     break

        if i % 10000 == 0:
            savelines('trec5_degrade20_processed_a_' + str(i) + '.txt', original_newlines)
    
    for r in queues:
        r.wait()

    savelines('trec5_degrade20_processed.txt', original_newlines)
    # savejson('distance_dict.json', distance_dict)

if __name__ == '__main__':
    # freeze_support()
    pool=Pool(THREAD)
    manager = Manager()

    process_trec5(pool)

# print(levenshteinDistance('parts', 'department'))
# print(levenshteinDistance('LEpAhTMENT', 'DEPARTMENT'))