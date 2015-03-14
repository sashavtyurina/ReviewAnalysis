
import sys
import json
import nltk
from nltk.tokenize import regexp_tokenize
from nltk.corpus import stopwords
import operator
from nltk.collocations import *

# input file - raw data.
# 1. stream topic and text
# 2. remove stop words,

# streaming
# import ijson
# filename = sys.argv[1]
# parser = ijson.parse(open(filename))
# for prefix, event, value in parser:
#     if prefix.endswith('.topic'):
#
#         v = value
#     elif prefix.endswith('.rating'):
#         v = value
#     elif prefix.endswith('.version'):
#         v = value
#     elif prefix.endswith('.user'):
#         v = value
#     elif prefix.endswith('.text'):
#         v = value

def tokenizeText(text):
    '''
    tokenize input text into words
    omit punctuation
    bring all words to lowercase
    :param text:
    :return: list of tokens
    '''

    resut = [word.lower() for word in regexp_tokenize(text, pattern='\w+')]
    return resut

def getDataFromJson(jsonfilname, fieldName):
    '''
    from raw data in json format extract data stored under the fieldName and return a string
    :param jsonfilname:
    :return:
    '''
    jsonFile = open(jsonfilname)
    reviews = json.load(jsonFile)
    result = ''

    for review in reviews['reviews']:
        topic = review[fieldName]
        result += topic

    jsonFile.close()
    return result

def removeStopWords(list):
    '''
    from input list of words remove those which are stop words (i.e. contained in nltk.corpus.stopwords)
    :param list:
    :return: clear list
    '''
    return [word for word in list if word not in stopwords.words('english')]

def countFrequencies(inputfilename, outputfilename):
    '''
    load raw data from json file
    split sentences into words (omit all punctuations)
    remove stopwords (the, a, of,...)
    count frequencies of all the words
    write them to output file
    :param inputfilename: file with raw data in json format
    :param outputfilename: output file where to write frequecnies count
    :return:nothing
    '''

    topics = getDataFromJson(inputfilename, 'topic')
    texts = getDataFromJson(inputfilename, 'text')

    topicsTokenized = tokenizeText(topics)
    textsTokenized = tokenizeText(texts)

    topicSWFiltered = removeStopWords(topicsTokenized)
    textSWFiltered = removeStopWords(textsTokenized)


    wordsBank = topicSWFiltered + textSWFiltered
    wordsFreq = {}
    for word in wordsBank:
            if word in wordsFreq.keys():
                wordsFreq[word] += 1
            else:
                wordsFreq[word] = 1

    #pretty print word frequencies
    outputfile = open(outputfilename)
    sortedTextFreq = sorted(wordsFreq.items(), key=operator.itemgetter(1), reverse=True)
    for item in sortedTextFreq:
        outputfile.write('%s - %d' % (item[0], item[1]))

    outputfile.close()




jsonfilename = sys.argv[1]
outputfilename = sys.argv[2]

texts = getDataFromJson(jsonfilename, 'text')
tokenizedText = tokenizeText(texts)


bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()

noStopWordsText = removeStopWords(tokenizedText)

finder = BigramCollocationFinder.from_words(noStopWordsText)
finder.apply_freq_filter(10)
print finder.nbest(bigram_measures.pmi, 10)






