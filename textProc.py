
import sys
import json
import nltk
from nltk.tokenize import regexp_tokenize
from nltk.corpus import stopwords
import operator
from nltk.collocations import *
import enchant
from nltk.stem import *
from textblob import TextBlob

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

def get_useful_information (jsonfilename, outputfilename):
    texts = getDataFromJson(jsonfilename, 'text')
    topics = getDataFromJson(jsonfilename, 'topic')
    tokenizedTexts = tokenizeText(texts)
    tokenizedTopics = tokenizeText(topics)

    additinal_stopwords = ['app', 'please', 'u', 'doesn', 'isn', 'aren', 'm', 're', 'fix', 'stars']
    cleanTexts = removeStopWords(tokenizedTexts, additinal_stopwords)
    cleanTexts = removeNonEnglish(cleanTexts)

    cleanTopics = removeStopWords(tokenizedTopics, additinal_stopwords)
    cleanTopics = removeNonEnglish(cleanTopics)

    outputfile = open(outputfilename, 'w')
    outputfile.write('%s\n%s' % (cleanTopics, cleanTexts))
    outputfile.close()

def similar_bigrams (b1, b2):
    '''
    If the bigrams are similar, we will mere them into one high-level topic.
    :param b1:
    :param b2:
    :return:Ture if bigrams have at least one word in common
     False otherwise
    '''
    if b1[0] in b2 or b1[1] in b2:
        return True
    return False

def all_similar_bigrams(b, bigrams_list):
    '''

    :param b:
    :param bigrams_list:
    :return:
    '''

    result = []
    for bigram in bigrams_list:
        if similar_bigrams(bigram, b):
            result.append(bigram)
    return result


def chunk_has_bigram(b, chunk):
    '''
    checks if chunk of text(list) has this bigram.
    words of the bigram should appear in the right order
    :param b:
    :param chunk:
    :return:
    '''

    if b[0] in chunk:
        if b[1] in chunk[chunk.index(b[0]):]:
            return True
    return False


# tri_frequency_thresh = 72

#

# trigram_measures = nltk.collocations.TrigramAssocMeasures()

# jsonfilename = sys.argv[1]
# outputfilename = sys.argv[2]
# get_useful_information(jsonfilename, outputfilename)

inputfile = open(sys.argv[1])
bi_frequency_thresh = 50
bigram_measures = nltk.collocations.BigramAssocMeasures()
items = inputfile.read().split()
words = [items[i][2:len(items[i]) - 2] for i in range(len(items))]
words[0] = words[0][1:]

print 'loaded text'

finder = BigramCollocationFinder.from_words(words, 4)
finder.apply_freq_filter(bi_frequency_thresh)
# # top10 = finder.nbest(bigram_measures.pmi, 10)

# # BIGRAMS
scored_bigrams = finder.score_ngrams(bigram_measures.pmi)
print 'scored bigrams'

#check POS tags
# allowed_pos = ['NN', 'NNS']
# bigrams = []
# for scored_bigram in scored_bigrams:
#     bigram = scored_bigram[0]
#
#     if bigram[0] == bigram[1]:
#         continue
#
#     pos_tags = nltk.pos_tag(bigram)
#
#     if pos_tags[0][1] in allowed_pos and pos_tags[1][1] in allowed_pos:
#         bigrams.append(bigram)
#
#
# print bigrams

# POS check with text blob
allowed_pos_templates = ['NN NN', 'VB NNS', 'VB NN', 'NN NNS']
bigrams = []
for scored_bigram in scored_bigrams:
    bigram = scored_bigram[0]
    if bigram[0] == bigram[1]:
        continue

    pos_tags = TextBlob(' '.join(bigram)).tags

    if ' '.join([pos_tags[0][1], pos_tags[1][1]]) in allowed_pos_templates:
        bigrams.append(bigram)


print bigrams

# STEM
# MERGE together bigrams that stem down to the same STEM
# for example, 'send message' and 'send messages' -> 'send messag'
print 'STEMMING'
stemmer = PorterStemmer()
stemmed_bigrams = {}

for bigram in bigrams:
    stem = (stemmer.stem(bigram[0]), stemmer.stem(bigram[1]))

    if stem in stemmed_bigrams:
        stemmed_bigrams[stem].append(bigram)
    else:
        stemmed_bigrams[stem] = [bigram]

# # MERGE BIGRAMS INTO TOPICS
topics = {}
keys = stemmed_bigrams.keys()
while len(keys) != 0:
    key = keys.pop()
    topics[key] = stemmed_bigrams[key]
    similar = all_similar_bigrams(key, keys)

    for s in similar:
        topics[key].append(stemmed_bigrams[s])
        keys.remove(s)


for t in topics.iteritems():
    print t




# SLIDING WINDOW
# for bigram_of_interest in bigrams:
#     window_size = 6
#     sentiment_sum = 0
#     count = 0
#     for i in range(len(words) - window_size):
#         chunk = words[i:i+window_size]
#         if chunk_has_bigram(bigram_of_interest, chunk):
#             chunk.remove(bigram_of_interest[0])
#             chunk.remove(bigram_of_interest[1])
#             b = TextBlob(' '.join(chunk))
#             # print b, ' - ', b.sentiment
#             sentiment_sum += b.sentiment.polarity
#             count += 1
#     print bigram_of_interest, ' - ', sentiment_sum/count







# pmi_output = open('bi_pmi_colocations.txt', 'w')
# pmi_output.write('Method = PMI\nFor # of words = %d\n Frequency threshold = %d\n' % ( words_count, bi_frequency_thresh) )
# for item in scored_bigrams:
#     pmi_output.write('%s - %s\n' % (item[0], item[1]))
# pmi_output.close()
#
# scored_bigrams = finder.score_ngrams(bigram_measures.student_t)
# student_t = open('bi_student_t_colocations.txt', 'w')
# student_t.write('Method = student_t test\nFor # of words = %d\n Frequency threshold = %d\n' % ( words_count, bi_frequency_thresh) )
# for item in scored_bigrams:
#     student_t.write('%s - %s\n' %(item[0], item[1]))
# student_t.close()
#
# scored_bigrams = finder.score_ngrams(bigram_measures.chi_sq)
# chi_sq = open('bi_chi_sq_colocations.txt', 'w')
# chi_sq.write('Method = chi square\nFor # of words = %d\n Frequency threshold = %d\n' % ( words_count, bi_frequency_thresh) )
# for item in scored_bigrams:
#     chi_sq.write('%s - %s\n' % (item[0], item[1]))
# chi_sq.close()
#
# scored_bigrams = finder.score_ngrams(bigram_measures.jaccard)
# jaccard = open('bi_jaccard_colocations.txt', 'w')
# jaccard.write('Method = jaccard\nFor # of words = %d\n Frequency threshold = %d\n' % ( words_count, bi_frequency_thresh) )
# for item in scored_bigrams:
#     jaccard.write('%s - %s\n' % (item[0], item[1]))
# jaccard.close()
#
# scored_bigrams = finder.score_ngrams(bigram_measures.poisson_stirling)
# poisson_stirling = open('bi_poisson_stirling_colocations.txt', 'w')
# poisson_stirling.write('Method = poisson-stirling\nFor # of words = %d\n Frequency threshold = %d\n' % ( words_count, bi_frequency_thresh) )
# for item in scored_bigrams:
#     poisson_stirling.write('%s - %s\n' % (item[0], item[1]))
# poisson_stirling.close()
#
# scored_bigrams = finder.score_ngrams(bigram_measures.likelihood_ratio)
# likelihood_ratio = open('bi_likelihood_ratio_colocations.txt', 'w')
# likelihood_ratio.write('Method = likelihood ration\nFor # of words = %d\n Frequency threshold = %d\n' % ( words_count, bi_frequency_thresh) )
# for item in scored_bigrams:
#     likelihood_ratio.write('%s - %s\n' % (item[0], item[1]))
# likelihood_ratio.close()
#
# #TRIGRAMS
#
#
# triFinder = TrigramCollocationFinder.from_words(cleanText)
# triFinder.apply_freq_filter(tri_frequency_thresh)
#
# # quad_frequency_thresh = 3
# # quadgramFinder = QuadgramCollocationFinder.from_words(cleanText)
# # quadgramFinder.apply_freq_filter(quad_frequency_thresh)
#
#
# scored_trigrams = triFinder.score_ngrams(trigram_measures.pmi)
# tri_pmi = open('tri_pmi_collocations.txt', 'w')
# tri_pmi.write('Method = PMI\nFor # of words = %d\n Frequency threshold = %d\n' % (words_count, tri_frequency_thresh) )
# for item in scored_trigrams:
#     tri_pmi.write('%s - %s\n' % (item[0], item[1]))
# tri_pmi.close()
#
#
# scored_trigrams = triFinder.score_ngrams(trigram_measures.student_t)
# tri_student_t = open('tri_student_t_collocations.txt', 'w')
# tri_student_t.write('Method = student_t\nFor # of words = %d\n Frequency threshold = %d\n' % (words_count, tri_frequency_thresh))
# for item in scored_trigrams:
#     tri_student_t.write('%s - %s\n' % (item[0], item[1]))
# tri_student_t.close()
#
# scored_trigrams = triFinder.score_ngrams(trigram_measures.chi_sq)
# tri_chi_sq = open('tri_chi_sq_collocations.txt', 'w')
# tri_chi_sq.write('Method = chi square\nFor # of words = %d\n Frequency threshold = %d\n' % (words_count, tri_frequency_thresh))
# for item in scored_trigrams:
#     tri_chi_sq.write('%s - %s\n' % (item[0], item[1]))
# tri_chi_sq.close()
#
# scored_trigrams = triFinder.score_ngrams(trigram_measures.jaccard)
# tri_jaccard = open('tri_jaccard_collocations.txt', 'w')
# tri_jaccard.write('Method = jaccard\nFor # of words = %d\n Frequency threshold = %d\n' % (words_count, tri_frequency_thresh))
# for item in scored_trigrams:
#     tri_jaccard.write('%s - %s\n' % (item[0], item[1]))
# tri_jaccard.close()
#
# scored_trigrams = triFinder.score_ngrams(trigram_measures.poisson_stirling)
# tri_poisson = open('tri_poisson_stirling_collocations.txt', 'w')
# tri_poisson.write('Method = poisson stirling\nFor # of words = %d\n Frequency threshold = %d\n' % (words_count, tri_frequency_thresh))
# for item in scored_trigrams:
#     tri_poisson.write('%s - %s\n' % (item[0], item[1]))
# tri_poisson.close()
#
# scored_trigrams = triFinder.score_ngrams(trigram_measures.likelihood_ratio)
# tri_likelihood_ratio = open('tri_likelihood_ratio_collocations.txt', 'w')
# tri_likelihood_ratio.write('Method = likelihood_ratio\nFor # of words = %d\n Frequency threshold = %d\n' % (words_count, tri_frequency_thresh))
# for item in scored_trigrams:
#     tri_likelihood_ratio.write('%s - %s\n' % (item[0], item[1]))
# tri_likelihood_ratio.close()


