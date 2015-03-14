'''Apple AppStore reviews scrapper
    version 2011-04-12
    Tomek "Grych" Gryszkiewicz, grych@tg.pl
    http://www.tg.pl
    
    based on "Scraping AppStore Reviews" blog by Erica Sadun
     - http://blogs.oreilly.com/iphone/2008/08/scraping-appstore-reviews.html
    AppStore codes are based on "appstore_reviews" by Jeremy Wohl
     - https://github.com/jeremywohl/iphone-scripts/blob/master/appstore_reviews
'''
import urllib2
from elementtree import ElementTree
import sys
import string
import argparse
import re
import xml
import json
import os
import ssl


appStores = {
'Argentina':          143505,
'Australia':          143460,
'Belgium':            143446,
'Brazil':             143503,
'Canada':             143455,
'Chile':              143483,
'China':              143465,
'Colombia':           143501,
'Costa Rica':         143495,
'Croatia':            143494,
'Czech Republic':     143489,
'Denmark':            143458,
'Deutschland':        143443,
'El Salvador':        143506,
'Espana':             143454,
'Finland':            143447,
'France':             143442,
'Greece':             143448,
'Guatemala':          143504,
'Hong Kong':          143463,
'Hungary':            143482,
'India':              143467,
'Indonesia':          143476,
'Ireland':            143449,
'Israel':             143491,
'Italia':             143450,
'Korea':              143466,
'Kuwait':             143493,
'Lebanon':            143497,
'Luxembourg':         143451,
'Malaysia':           143473,
'Mexico':             143468,
'Nederland':          143452,
'New Zealand':        143461,
'Norway':             143457,
'Osterreich':         143445,
'Pakistan':           143477,
'Panama':             143485,
'Peru':               143507,
'Phillipines':        143474,
'Poland':             143478,
'Portugal':           143453,
'Qatar':              143498,
'Romania':            143487,
'Russia':             143469,
'Saudi Arabia':       143479,
'Schweiz/Suisse':     143459, 
'Singapore':          143464,
'Slovakia':           143496,
'Slovenia':           143499,
'South Africa':       143472,
'Sri Lanka':          143486,
'Sweden':             143456,
'Taiwan':             143470,
'Thailand':           143475,
'Turkey':             143480,
'United Arab Emirates'  :143481,
'United Kingdom':     143444,
'United States':      143441,
'Venezuela':          143502,
'Vietnam':            143471,
'Japan':              143462,
'Dominican Republic': 143508,
'Ecuador':            143509,
'Egypt':              143516,
'Estonia':            143518,
'Honduras':           143510,
'Jamaica':            143511,
'Kazakhstan':         143517,
'Latvia':             143519,
'Lithuania':          143520,
'Macau':              143515, 
'Malta':              143521,
'Moldova':            143523,  
'Nicaragua':          143512,
'Paraguay':           143513,
'Uruguay':            143514
}



def getReviews(appStoreId, appId, filename, maxReviews=-1):
    ''' returns list of reviews for given AppStore ID and application Id
        return list format: [{"topic": unicode string, "review": unicode string, "rank": int}]
    '''

    reviews=[]
    i=0
    while True: 
        ret = _getReviewsForPage(appStoreId, appId, i)
        i += 1
        if ret == None:
            continue

        if len(ret) == 0: # funny do while emulation ;)
            break
        reviews += ret
        print 'Printing reviews to json'
        printReviewsToJson(ret, filename)

        
        if maxReviews > 0 and len(reviews) > maxReviews:
            break
    return reviews

def _getReviewsForPage(appStoreId, appId, pageNo):
    print "_get review for page %d" %pageNo
    userAgent = 'iTunes/9.2 (Macintosh; U; Mac OS X 10.6)'
    front = "%d-1" % appStoreId
    url = "https://itunes.apple.com/WebObjects/MZStore.woa/wa/viewContentsUserReviews?id=%s&pageNumber=%d&sortOrdering=4&onlyLatestVersion=false&type=Purple+Software" % (appId, pageNo)

    print "before urllibRequest"
    req = urllib2.Request(url, headers={"X-Apple-Store-Front": front,"User-Agent": userAgent})
    try:
        u = urllib2.urlopen(req, timeout=30)
        print "try and open urlLib2"
    except urllib2.HTTPError:
        print 'There was a problem connecting. Continue. '
        return None
    except ssl.SSLError:
        print 'There was a problem connecting. Continue. '
        return None
    except urllib2.URLError:
        print 'There was a problem connecting. Continue. '
        return None

        #print "Can't connect to the AppStore, please try again later."
        #raise SystemExit
    # print "before parsing tree"

    try:
        root = ElementTree.parse(u).getroot()
    except xml.parsers.expat.ExpatError:
        print "Page = %d has erroneous xml structure" %pageNo
        return None
   
    reviews=[]
    for node in root.findall('{http://www.apple.com/itms/}View/{http://www.apple.com/itms/}ScrollView/{http://www.apple.com/itms/}VBoxView/{http://www.apple.com/itms/}View/{http://www.apple.com/itms/}MatrixView/{http://www.apple.com/itms/}VBoxView/{http://www.apple.com/itms/}VBoxView/{http://www.apple.com/itms/}VBoxView/'):
        review = {}

        review_node = node.find("{http://www.apple.com/itms/}TextView/{http://www.apple.com/itms/}SetFontStyle")

        

        if review_node is None:
            review["review"] = None
        else:
            review["review"] = review_node.text

        version_node = node.find("{http://www.apple.com/itms/}HBoxView/{http://www.apple.com/itms/}TextView/{http://www.apple.com/itms/}SetFontStyle/{http://www.apple.com/itms/}GotoURL")
        if version_node is None:
            review["version"] = None
        else:
            review["version"] = re.search("Version [^\n^\ ]+", version_node.tail).group()
            review["date"] = re.search("[A-z]{3} [0-9]{2}, [0-9]{4}", version_node.tail).group()
    
        user_node = node.find("{http://www.apple.com/itms/}HBoxView/{http://www.apple.com/itms/}TextView/{http://www.apple.com/itms/}SetFontStyle/{http://www.apple.com/itms/}GotoURL/{http://www.apple.com/itms/}b")
        if user_node is None:
            review["user"] = None
        else:
            review["user"] = user_node.text.strip()

        rank_node = node.find("{http://www.apple.com/itms/}HBoxView/{http://www.apple.com/itms/}HBoxView/{http://www.apple.com/itms/}HBoxView")
        try:
            alt = rank_node.attrib['alt']
            st = int(alt.strip(' stars'))
            review["rank"] = st
        except KeyError:
            review["rank"] = None

        topic_node = node.find("{http://www.apple.com/itms/}HBoxView/{http://www.apple.com/itms/}TextView/{http://www.apple.com/itms/}SetFontStyle/{http://www.apple.com/itms/}b")
        if topic_node is None:
            review["topic"] = None
        else:
            review["topic"] = topic_node.text

        reviews.append(review)
    return reviews


def printReviewsToJson(reviews, filename):
    '''append a chunk of reviews to file in json format
    '''


    if len(reviews) > 0:
        file = open(filename, 'a')
        print 'opening file'
    else:
        return

    for review in reviews:
        try:
            version = review["version"]
            user = review["user"]
            rating = review["rank"]
            topic = review["topic"]
            text = review["review"]

            jsonObject = json.dumps({"version": version, "user": user, "rating": rating, "topic": topic, "text": text})
            file.write(jsonObject)
            file.write(',')
            #print jsonObject

        except TypeError:
            print "Bad review: %s" %review

    file.close()

def _print_reviews(reviews, country, filename):
    ''' returns (reviews count, sum rank)
    '''
    
    if len(reviews)>0:
        file = open(filename, 'w')

        sumRank = 0
        for review in reviews:
            try:
                version = review["version"]
                user = review["user"]
                title = "%s by %s\n" % (version, user)
                file.write(title.encode('utf8'))

                
                rating  = "*"*review["rank"]
                file.write(rating.encode('utf8'))

                text = "\n%s\n%s\n" % (review["topic"], review["review"])
                file.write(text.encode('utf8'))
                file.write("\n")
                sumRank += review["rank"]
            except TypeError:
                print "Bad review: %s" %review

        stats = "Number of reviews in %s: %d, avg rank: %.2f\n" % (country, len(reviews), 1.0*sumRank/len(reviews))
        file.write(stats.encode('utf8'))
        file.close()
        return (len(reviews), sumRank)
    else:
        return (0, 0)

def _print_rawmode(reviews):
    for review in reviews:
        print review["topic"], review["review"].replace("\n","")


def startJson(filename):
    '''opens a new file and adds a first line of json array.
    Comes together with stopJson. '''

    file = open(filename, 'w')
    file.write('{"reviews": [')
    file.close()

def stopJson(filename):
    file = open(filename, 'rb+')
    file.seek(-1, os.SEEK_END)
    file.truncate()
    file.close()

    file = open(filename, 'a')
    file.write(']}')
    file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AppStoreReviewsScrapper command line.', epilog='To get your application Id look into the AppStore link to you app, for example http://itunes.apple.com/pl/app/autobuser-warszawa/id335042980?mt=8 - app Id is the number between "id" and "?mt=0"')
    parser.add_argument('-i', '--id', default=0, metavar='AppId', type=int, help='Application Id (see below)')
    parser.add_argument('-c', '--country', metavar='"Name"', type=str, default='all', help='AppStore country name (use -l to see them)')
    parser.add_argument('-l', '--list', action='store_true', default=False, help='AppStores list')
    parser.add_argument('-m', '--max-reviews',default=-1,metavar='MaxReviews',type=int,help='Max number of reviews to load')
    parser.add_argument('-r', '--raw-mode',action='store_true',default=False,help='Print raw mode')
    args = parser.parse_args()
    if args.id == 0:
        parser.print_help()
        print "exiting..."
        raise SystemExit
    country = string.capwords(args.country)
    print "country = %s" %country
    countries=appStores.keys()
    countries.sort()
    print args.list
    if args.list:
        for c in countries:
            print c
    else:
        if (country=="All"):
            print "Country == ALL"
            rankCount = 0; rankSum = 0
            for c in countries:
                reviews = getReviews(appStores[c], args.id,maxReviews=args.max_reviews)
                (rc,rs) = _print_reviews(reviews, c)
                rankCount += rc
                rankSum += rs
            print "\nTotal number of reviews: %d, avg rank: %.2f" % (rankCount, 1.0 * rankSum/rankCount)
        else:
            try:
                filename = "twitter.txt"
                startJson(filename)

                reviews = getReviews(appStores[country], args.id, filename, maxReviews=args.max_reviews)
                stopJson(filename)


                # if args.raw_mode:
                #     _print_rawmode(reviews)
                #     print "printing raw mode"
                # else:
                #     filename = "reviewsForID%s" %args.id
                #     _print_reviews(reviews, country, filename)
                #     print "printing reviews"
            except KeyError:
                print "No such country %s!\n\nWell, it could exist in real life, but I dont know it." % country
            pass
