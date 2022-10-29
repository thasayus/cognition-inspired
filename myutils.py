from pathlib import Path
import gzip
import re
import html
import re
import collections
import sys
import nltk
from nltk.corpus import stopwords
from string import punctuation
from tensorflow.keras.preprocessing.text import text_to_word_sequence
import json
from nltk.tokenize import sent_tokenize
from spellchecker import SpellChecker
import time
from time import perf_counter_ns
from nltk.probability import FreqDist
import pandas as pd
import tqdm

def nested_dict():
    return collections.defaultdict(nested_dict)

def listFiles(path, filter='*.*'):
    return list(Path(path).rglob(filter))

def readGZIP(filename):
    f=gzip.open(filename,'rb')
    input = f.read()
    return input

def search(pattern, line, index=1):
    search = re.search(pattern, line)
    if search:
        return search.group(index)
    else:
        return None

def searchGroup(pattern, line):
    search = re.search(pattern, line)
    if search:
        return search
    else:
        return None
        
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
                    data.append(atext)

            ntext = text[tag[1]:tag[2]]
            ntext = ntext.strip()
            if ntext:
                ntext = re.sub(r'\n\s*\n', '\n', ntext)
                ar = processtext1(ntext)
                if ar:
                    data.extend(ar)
            currentindex = tag[2] + len(tag[0]) + 1
    else:
        text = text.strip()
        text = re.sub(r'\n\s*\n', '\n', text)
        if text:
            return [text]
        else:
            return []
    return data

def countSameChars(str1, str2): 
    c, j = 0, 0
    for i in str1:    
        if str2.find(i)>= 0 and j == str1.find(i): 
            c += 1
        j += 1
    return c
  

def processtext(text):
    newline = False
    data = ''
    state = None
    for line in text.splitlines():
        line1 = re.sub(r'(<!--.*?-->)', ' ', line).strip()
        if not line1:
            if not newline:
                data = data + "\n"
                newline = True
            continue
        else:
            if not state:
                tag = search(r'<(\w+)>', line)

            line1 = line.strip()
            if not tag and state == None:
                if line1:
                    data = data + line1
                    newline = False
                    continue
            elif tag and state == None:
                line1 = re.sub(r'<' + tag + r'>', ' ', line1)
                state = 'tag'
            
            if tag:
                endtag = search("<\/(" + tag + ")>", line1)
                if endtag:
                    line1 = re.sub(r'</' + tag + r'>', ' ', line1)
                    tag = None
                    state = None
            
            line1 = line1.strip()
            if line1:
                data = data + line1
                if endtag:
                     data = data + "\n"
                newline = False
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

def sentence_tokenized(texts):
    # text = ' '.join([str(x) for x in texts])
    sentences = []
    for text in texts:
        for line in text.splitlines():
            if '.' in line:
                line_end = ''
                ends = line.split(".")
                for end in ends:
                    found = False
                    tokens = word_tokenize(end)
                    if len(tokens)>2:
                        count = 0
                        for token in tokens:
                            if len(token)>2:
                                count = count + 1
                        if count>2:
                            found = True
                    if found:
                        sentences.append(line_end + end + '.')
                        line_end = ''
                    else:
                        line_end = line_end + end + '.'
            else:
                sentences.append(line)
            # sentences.append(line)
    # tokens_sentences = nltk.sent_tokenize(text)
    return sentences

def sentence_tokenized3(texts):
    # text = ' '.join([str(x) for x in texts])
    sentences = []
    special_characters = ['.', ':', ';', '\r', '?', '!', '/', '\\', '#']
    
    for text in texts:
        lines = re.split('[:.;?!|\\r\\\\#\\/]', text)
        for line in lines:
            line = line.replace('\n',' ')
            line = line.strip()
            if line:
                sentences.append(line)        
            # sentences.append(line)
    # tokens_sentences = nltk.sent_tokenize(text)
    return sentences

# def sentence_tokenized(texts):
#     # text = ' '.join([str(x) for x in texts])
#     sentences = []
#     for text in texts:
#         line = re.sub('\n', ' ', text)
#         # for line in text.splitlines():
#         sentences.append(line.strip())
#     return sentences

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

def findX(sentense, ci, cl):
    i = ci
    if ci + cl > len(sentense):
        return None
    else:
        return ci + cl

def word_tokenize(sentence):
    # stoplist = set(stopwords.words('english') + list(punctuation))
    # stoplist = set(['.', ':'])
    # return [token for token in nltk.word_tokenize(sentence) if token.lower() not in stoplist]
    return text_to_word_sequence(sentence)



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
            

def check_trec5(DATA_PATH, orig_filename, filters):
    de5_filename = re.sub(r'original', DATA_PATH[1], orig_filename)
    de20_filename = re.sub(r'original', DATA_PATH[2], orig_filename)
    print('read,' + orig_filename)
    orig_data = readTREC5file(orig_filename)
    de5_data = readTREC5file(de5_filename)
    # de20_data = readTREC5file(de20_filename)
    for d1 in orig_data:
        if d1 in filters:
            # print('exiting:' + d1)
            continue
        orig_sentences = sentence_tokenized(orig_data[d1]['text'])
        de5_sentences = sentence_tokenized(de5_data[d1]['text'])
        aorgin_sentences, ade5_sentences, count_error = verifySentense(orig_sentences, de5_sentences)
        print('errorcount,' + d1 + ',' + str(count_error))
        savefile = 'data/confusion_track/filters/' + str(count_error) + '/' +  d1 + '.json'
        with open(savefile, 'w') as outfile:
            json.dump({ 'docno' : d1, 'parent' : orig_data[d1]['parent'], 'original' : aorgin_sentences, 'degrade5' : ade5_sentences, 'counterror' : count_error}, outfile)
    #     countall_errors[str(count_error)] += 1

    #         # print(de5_data[d1]['text'])
    #         # print(de20_data[d1]['text'])
    
    # for cekey in countall_errors:
    #     print('errorall,' + cekey + ',' + countall_errors[cekey])

def read_trec5(TREC5_PATH, DATA_PATH, pool):
    scan_data = nested_dict()
    count = 0
    escepses = set()
    dpath = DATA_PATH[0]
    correct_files = listFiles(TREC5_PATH + dpath, "*.gz")

    filter_files = [filter.stem for filter in listFiles(TREC5_PATH + 'filters', "*.json")]
    # print(correct_files)
    countall_errors = collections.defaultdict(int)
    results = []
    for file in correct_files:
        orig_filename = str(file)
        result = pool.apply_async(check_trec5, [DATA_PATH, orig_filename, filter_files])
        results.append(result)
        # check_trec5(DATA_PATH, orig_filename)
    [result.wait() for result in results]
    return scan_data

def sentence_tokenized1(texts):
    sentences = []
    for text in texts:
        sentence = sent_tokenize(text)
        sentences.append(sentence)
    return sentences

def findWordInFilter(word, sentences, xindex, yindex):
    found = False
    change = 0
    last = False
    while xindex<len(sentences):
        sentence = sentences[xindex]
        if len(sentence) == 0:
            change += 1
            xindex += 1
            yindex = 0
        else:
            if yindex<len(sentence):
                s = sentence[yindex]
                if s == word:
                    if yindex == len(sentence) - 1:
                        last = True
                    found = True
                    break
                else:
                    yindex += 1
            else:
                xindex += 1
                yindex = 0
                change += 1
    return [found, xindex, yindex, change, last]

def wordsent_tokenize(orig_sentences):
    aorgin_sentences = []
    for orig_sentence in orig_sentences:
        otokens = word_tokenize(orig_sentence)
        aorgin_sentences.append(otokens)
    return aorgin_sentences

def convertMN1toMN2(sentences1, x, y, change):
    i = x
    while i < len(sentences1):
        sentence2 = sentences1[i]
        ley = len(sentence2) 
        if (ley - y) > change:
            return [i, y + change]
        else:
            if i == len(sentences1) - 1:
                return [i, ley - 1]
            change -= (ley - y)
            i += 1
            y = 0
    return None

def eval_pyspellchecker(TREC5_PATH, DATA_PATH, file, filter_files, result_data):
    orig_filename = str(file)
    orig_data = readTREC5file(orig_filename)
    de5_filename = re.sub(r'original', DATA_PATH[1], orig_filename)
    de5_data = readTREC5file(de5_filename)
    spell = SpellChecker()
    pattern_word = re.compile("^([a-zA-Z']+)$")
    for docno in orig_data:
        if docno in filter_files:
            # print(docno)
            wordcount, wrong, correct, wordcorrect, wordwrong, timeperfsum = 0, 0, 0, 0, 0, 0.0
            orig_sentences = sentence_tokenized3(orig_data[docno]['text'])
            orig_sentences = wordsent_tokenize(orig_sentences)
            de5_sentences = sentence_tokenized3(de5_data[docno]['text'])
            de5_sentences = wordsent_tokenize(de5_sentences)
            filter_file = TREC5_PATH + 'filters/0/' + docno + '.json'
            with open(filter_file) as json_file:
                filter = json.load(json_file)
            xdindex = 0
            ydindex = 0
            xoindex = 0
            yoindex = 0
            dptext = ''
            for de5_sentence in de5_sentences:
                for de5_word in de5_sentence:
                    [found, xdindex, ydindex, change, last] = findWordInFilter(de5_word, filter['degrade5'], xdindex, ydindex)
                    [xoindex, yoindex] = convertMN1toMN2(filter['original'], xoindex, yoindex, change)
                    if found:
                        if change >= 1:
                            dptext = de5_word
                        else:
                            dptext += de5_word
                        # print(de5_word + ',' + filter['original'][xoindex][yoindex] + ',' + filter['degrade5'][xdindex][ydindex])
                        if last:
                            oword = filter['original'][xoindex][yoindex]
                            dist1 = levenshteinDistance(dptext, oword)
                            dist2 = levenshteinDistance(de5_word, oword)
                            if (dist1>=dist2):
                                dword = de5_word
                            else:                        
                                dword = dptext

                            # if dist1>3:
                            #     print(str(dist1) + ',' + str(change))
                            # print(de5_word + ',' + oword)
                            t1_start = perf_counter_ns()
                            correctword = spell.correction(dword)
                            t1_stop = perf_counter_ns()
                            timeperfsum += (t1_stop-t1_start)
                            wordcount += 1
                            if correctword == oword:
                                correct += 1
                            else:
                                wrong += 1
                            if pattern_word.search(oword):
                                if correctword == oword:
                                    wordcorrect += 1
                                else:
                                    wordwrong += 1
                            dptext = ''                      
                    # else:
                        # print('not found!' + str(change))
                    ydindex += 1
        result_data[docno]['pyspellchecker']['wordcount'] = str(wordcount)
        result_data[docno]['pyspellchecker']['wrong'] = str(wrong)
        result_data[docno]['pyspellchecker']['correct'] = str(correct)
        result_data[docno]['pyspellchecker']['wordcorrect'] = str(wordcorrect)
        result_data[docno]['pyspellchecker']['wordwrong'] = str(wordwrong)
        result_data[docno]['pyspellchecker']['timeperfsum'] = str(timeperfsum)
        if wordcount>0:
            result_data[docno]['pyspellchecker']['timeperf'] = str(timeperfsum / wordcount)
        for key in result_data[docno]['pyspellchecker']:
            print(docno + ',' + key + ',' + result_data[docno]['pyspellchecker'][key])


def eval(TREC5_PATH, DATA_PATH, pool = None):
    dpath = DATA_PATH[0]
    correct_files = listFiles(TREC5_PATH + DATA_PATH[0], "*.gz")
    filter_files = [filter.stem for filter in listFiles(TREC5_PATH + 'filters/0', "*.json")]
    
    result_data = nested_dict()
    def collect_results(result):
        if result:
            for k,v in result.items():
                if  result[k]['wordcompare']:
                    result_data[k]['wordcompare'] = result[k]['wordcompare']

    timestr = time.strftime("%Y%m%d-%H%M%S")
    results = []
    for file in correct_files:
        result = pool.apply_async(gencomparedwords, [TREC5_PATH, DATA_PATH, file, filter_files], callback=collect_results)
        results.append(result)
        # gencomparedwords(TREC5_PATH, DATA_PATH, file, filter_files, result_data)
    
    # pool.close()
    # pool.join()
    [result.wait() for result in tqdm.tqdm(results)]
    # for result in results:
    #     result.wait()
    #     resultg = result.get()
    #     for k,v in resultg.items():
    #         result_data[k]['wordcompare'] = resultg[k]['wordcompare']
            
    # df.to_csv('word_ocr_compare.csv', sep='\t', encoding='utf-8')
    with open('wordcompare_' + timestr + '.json', 'w') as outfile:
        json.dump(result_data, outfile)

                        
def freqWord(sentences):
    words = nltk.word_tokenize(' '.join(sentences))
    wordfreqs = nltk.FreqDist(words)
    filter_words = [[m, n] for m, n in wordfreqs.most_common()]
    df = pd.DataFrame(filter_words, columns =['word', 'count'])
    return df

def readjson(filename):
    with open(filename) as json_file:
        return json.load(json_file)
    return None

def savejson(filename, object):
    with open(filename, 'w') as outfile:
        json.dump(object, outfile)


def gencomparedwords(TREC5_PATH, DATA_PATH, file, filter_files):
    orig_filename = str(file)
    result_data = nested_dict()
    try:
        orig_data = readTREC5file(orig_filename)
        de5_filename = re.sub(r'original', DATA_PATH[1], orig_filename)
        de5_data = readTREC5file(de5_filename)
    except:
        return result_data
    for docno in orig_data:
        if docno in filter_files:
            # print(docno)
            wordcompare = []
            orig_sentences = sentence_tokenized3(orig_data[docno]['text'])
            orig_sentences = wordsent_tokenize(orig_sentences)
            de5_sentences = sentence_tokenized3(de5_data[docno]['text'])
            de5_sentences = wordsent_tokenize(de5_sentences)
            filter_file = TREC5_PATH + 'filters/0/' + docno + '.json'
            with open(filter_file) as json_file:
                filter = json.load(json_file)
            xdindex = 0
            ydindex = 0
            xoindex = 0
            yoindex = 0
            dptext = ''
            for de5_sentence in de5_sentences:
                for de5_word in de5_sentence:
                    [found, xdindex, ydindex, change, last] = findWordInFilter(de5_word, filter['degrade5'], xdindex, ydindex)
                    [xoindex, yoindex] = convertMN1toMN2(filter['original'], xoindex, yoindex, change)
                    if found:
                        if change >= 1:
                            dptext = de5_word
                        else:
                           
                            dptext += de5_word
                        # print(de5_word + ',' + filter['original'][xoindex][yoindex] + ',' + filter['degrade5'][xdindex][ydindex])
                        if last:
                            oword = filter['original'][xoindex][yoindex]
                            dist1 = levenshteinDistance(dptext, oword)
                            dist2 = levenshteinDistance(de5_word, oword)
                            if (dist1>=dist2):
                                dist = dist2
                                dword = de5_word
                            else:
                                dist = dist1                                
                                dword = dptext

                            wordcompare.append([docno, oword, dword, dist])
                            dptext = ''              
                    # else:
                        # print('not found!' + str(change))
                    ydindex += 1
            if len(wordcompare)>0:
                # print(len(wordcompare))
                result_data[docno]['wordcompare'] = wordcompare[:]
    return result_data
    # df = pd.read_csv('1459966468_324.csv', encoding='utf8')
    
