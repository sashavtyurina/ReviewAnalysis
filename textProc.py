
import sys
import json
import nltk
from nltk.tokenize import regexp_tokenize
from nltk.corpus import stopwords
import operator
from nltk.collocations import *
import enchant

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

    resut = [word.lower() for word in regexp_tokenize(text, pattern="[A-z]+")]
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
        try:
            topic = review[fieldName]
            result += topic
        except TypeError:
            print review
            continue
    jsonFile.close()
    return result

def removeStopWords(list, additionalSW):
    '''
    from input list of words remove those which are stop words (i.e. contained in nltk.corpus.stopwords)
    :param list:
    :param additionalSW: list of additional stop words
    :return: clear list
    '''

    stop = stopwords.words('english') + additionalSW
    return [word for word in list if word not in stop]

def removeNonEnglish(list):
    '''
    removes non English words using enchant module
    :param list:
    :return: filtered list
    '''
    eng_d = enchant.Dict('en_US')
    return [word for word in list if eng_d.check(word)]


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



bi_frequency_thresh = 600
tri_frequency_thresh = 120

jsonfilename = sys.argv[1]
#outputfilename = sys.argv[2]

texts = getDataFromJson(jsonfilename, 'text')
tokenizedText = tokenizeText(texts)


bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()

cleanText = removeStopWords(tokenizedText, ['app', 'please', 'u', 'doesn', 'isn', 'aren', 'm', 're'])
cleanText = removeNonEnglish(cleanText)
words_count = len(cleanText)


finder = BigramCollocationFinder.from_words(cleanText)
finder.apply_freq_filter(bi_frequency_thresh)
# top10 = finder.nbest(bigram_measures.pmi, 10)

# BIGRAMS
scored_bigrams = finder.score_ngrams(bigram_measures.pmi)
pmi_output = open('bi_pmi_colocations.txt', 'w')
pmi_output.write('Method = PMI\nFor # of words = %d\n Frequency threshold = %d\n' % ( words_count, bi_frequency_thresh) )
for item in scored_bigrams:
    pmi_output.write('%s - %s\n' % (item[0], item[1]))
pmi_output.close()

scored_bigrams = finder.score_ngrams(bigram_measures.student_t)
student_t = open('bi_student_t_colocations.txt', 'w')
student_t.write('Method = student_t test\nFor # of words = %d\n Frequency threshold = %d\n' % ( words_count, bi_frequency_thresh) )
for item in scored_bigrams:
    student_t.write('%s - %s\n' %(item[0], item[1]))
student_t.close()

scored_bigrams = finder.score_ngrams(bigram_measures.chi_sq)
chi_sq = open('bi_chi_sq_colocations.txt', 'w')
chi_sq.write('Method = chi square\nFor # of words = %d\n Frequency threshold = %d\n' % ( words_count, bi_frequency_thresh) )
for item in scored_bigrams:
    chi_sq.write('%s - %s\n' % (item[0], item[1]))
chi_sq.close()

scored_bigrams = finder.score_ngrams(bigram_measures.jaccard)
jaccard = open('bi_jaccard_colocations.txt', 'w')
jaccard.write('Method = jaccard\nFor # of words = %d\n Frequency threshold = %d\n' % ( words_count, bi_frequency_thresh) )
for item in scored_bigrams:
    jaccard.write('%s - %s\n' % (item[0], item[1]))
jaccard.close()

scored_bigrams = finder.score_ngrams(bigram_measures.poisson_stirling)
poisson_stirling = open('bi_poisson_stirling_colocations.txt', 'w')
poisson_stirling.write('Method = poisson-stirling\nFor # of words = %d\n Frequency threshold = %d\n' % ( words_count, bi_frequency_thresh) )
for item in scored_bigrams:
    poisson_stirling.write('%s - %s\n' % (item[0], item[1]))
poisson_stirling.close()

scored_bigrams = finder.score_ngrams(bigram_measures.likelihood_ratio)
likelihood_ratio = open('bi_likelihood_ratio_colocations.txt', 'w')
likelihood_ratio.write('Method = likelihood ration\nFor # of words = %d\n Frequency threshold = %d\n' % ( words_count, bi_frequency_thresh) )
for item in scored_bigrams:
    likelihood_ratio.write('%s - %s\n' % (item[0], item[1]))
likelihood_ratio.close()

#TRIGRAMS


triFinder = TrigramCollocationFinder.from_words(cleanText)
triFinder.apply_freq_filter(tri_frequency_thresh)

# quad_frequency_thresh = 3
# quadgramFinder = QuadgramCollocationFinder.from_words(cleanText)
# quadgramFinder.apply_freq_filter(quad_frequency_thresh)


scored_trigrams = triFinder.score_ngrams(trigram_measures.pmi)
tri_pmi = open('tri_pmi_collocations.txt', 'w')
tri_pmi.write('Method = PMI\nFor # of words = %d\n Frequency threshold = %d\n' % (words_count, tri_frequency_thresh) )
for item in scored_trigrams:
    tri_pmi.write('%s - %s\n' % (item[0], item[1]))
tri_pmi.close()


scored_trigrams = triFinder.score_ngrams(trigram_measures.student_t)
tri_student_t = open('tri_student_t_collocations.txt', 'w')
tri_student_t.write('Method = student_t\nFor # of words = %d\n Frequency threshold = %d\n' % (words_count, tri_frequency_thresh))
for item in scored_trigrams:
    tri_student_t.write('%s - %s\n' % (item[0], item[1]))
tri_student_t.close()

scored_trigrams = triFinder.score_ngrams(trigram_measures.chi_sq)
tri_chi_sq = open('tri_chi_sq_collocations.txt', 'w')
tri_chi_sq.write('Method = chi square\nFor # of words = %d\n Frequency threshold = %d\n' % (words_count, tri_frequency_thresh))
for item in scored_trigrams:
    tri_chi_sq.write('%s - %s\n' % (item[0], item[1]))
tri_chi_sq.close()

scored_trigrams = triFinder.score_ngrams(trigram_measures.jaccard)
tri_jaccard = open('tri_jaccard_collocations.txt', 'w')
tri_jaccard.write('Method = jaccard\nFor # of words = %d\n Frequency threshold = %d\n' % (words_count, tri_frequency_thresh))
for item in scored_trigrams:
    tri_jaccard.write('%s - %s\n' % (item[0], item[1]))
tri_jaccard.close()

scored_trigrams = triFinder.score_ngrams(trigram_measures.poisson_stirling)
tri_poisson = open('tri_poisson_stirling_collocations.txt', 'w')
tri_poisson.write('Method = poisson stirling\nFor # of words = %d\n Frequency threshold = %d\n' % (words_count, tri_frequency_thresh))
for item in scored_trigrams:
    tri_poisson.write('%s - %s\n' % (item[0], item[1]))
tri_poisson.close()

scored_trigrams = triFinder.score_ngrams(trigram_measures.likelihood_ratio)
tri_likelihood_ratio = open('tri_likelihood_ratio_collocations.txt', 'w')
tri_likelihood_ratio.write('Method = likelihood_ratio\nFor # of words = %d\n Frequency threshold = %d\n' % (words_count, tri_frequency_thresh))
for item in scored_trigrams:
    tri_likelihood_ratio.write('%s - %s\n' % (item[0], item[1]))
tri_likelihood_ratio.close()


