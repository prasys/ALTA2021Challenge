from bs4 import BeautifulSoup as bs
import re
from django.utils.html import strip_tags
import argparse
import os,sys
import pandas as pd
from sklearn import preprocessing
from cleantext import clean



def clean_text(text):
    return(clean("some input",
    fix_unicode=True,               # fix various unicode errors
    to_ascii=True,                  # transliterate to closest ASCII representation
    lower=True,                     # lowercase text
    no_line_breaks=False,           # fully strip line breaks as opposed to only normalizing them
    no_urls=False,                  # replace all URLs with a special token
    no_emails=False,                # replace all email addresses with a special token
    no_phone_numbers=False,         # replace all phone numbers with a special token
    no_numbers=True,               # replace all numbers with a special token
    no_digits=True,                # replace all digits with a special token
    no_currency_symbols=False,      # replace all currency symbols with a special token
    no_punct=False,                 # remove punctuations
    replace_with_punct="",          # instead of removing punctuations you may replace them
    replace_with_url="<URL>",
    replace_with_email="<EMAIL>",
    replace_with_phone_number="<PHONE>",
    replace_with_number="<NUMBER>",
    replace_with_digit="0",
    replace_with_currency_symbol="<CUR>",
    lang="en"
    ))


def parse_text(text, patterns=None):
    """
    delete all HTML tags and entities
    :param text (str): given text
    :param patterns (dict): patterns for re.sub
    :return str: final text
    
    usage like:
    parse_text('<div class="super"><p>Hello&ldquo;&rdquo;!&nbsp;&nbsp;</p>&lsquo;</div>')
    >>> Hello!
    """
    base_patterns = {
        '&[rl]dquo;': '',
        '&[rl]squo;': '',
        '&nbsp;': ''
    }

    patterns = patterns or base_patterns

    final_text = strip_tags(text)
    for pattern, repl in patterns.items():
        final_text = re.sub(pattern, repl, final_text)
    return final_text.strip()

def parseAbstracts(fileName,cleanText=True):
    s = ""
    texts = []
    abstracts = loadXMLcontents(fileName)
    if len(abstracts) == 1:
        s = parse_text(abstracts[0])
        if cleanText:
            s = clean_text(s)
    else:
        for abstract in abstracts:
            texts.append(parse_text(abstract))
            s = ''.join(texts)
            if cleanText:
                s = clean_text(s)
    return s

def loadXMLcontents(fileName,tag="abstracttext"):
    infile = open(fileName,"r")
    contents = infile.read()
    soup = bs(contents,'xml')
    content = soup.find_all(tag)
    return content

def loadFileAndParse(filename,XMLFolderPath,truth,docCollections,cleanText):
    file1 = open(filename, 'r')
    for line in file1:
        temp = line.split()
        if (len(temp)-2) == 1: #if we only have 1 document collection
            truth.append(temp[1]) #append the grade
            docCollections.append(parseAbstracts(XMLFolderPath+temp[2]+".xml",cleanText)) #parseXML
        elif (len(temp)-2) > 1:
                for count in range(2,len(temp)):
                    truth.append(temp[1])
                    docCollections.append(parseAbstracts(XMLFolderPath+temp[count]+".xml",cleanText)) #parseXML  
    file1.close()


def createAndSaveDataFrame(truthLabels,docCollections,fileName):
    le = preprocessing.LabelEncoder()
    truthLabels = le.fit_transform(truthLabels)
    df = pd.DataFrame(list(zip(docCollections,truthLabels)),columns =['text', 'labels'])
    print(df.head())
    df.to_pickle(fileName)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default='trainset.txt', type=str)
    parser.add_argument('--XMLdir', default='/trainset/', type=str)
    parser.add_argument('--processedFile', default='output.h5', type=str)
    parser.add_argument('--cleanText', default=True, type=bool)
    opt = parser.parse_args()
    xmlPath = os.path.abspath(os.path.dirname(os.path.abspath(__file__))) + opt.XMLdir
    docCollections = []
    truthLabels = []
    loadFileAndParse(opt.file,xmlPath,truthLabels,docCollections,opt.cleanText)
    createAndSaveDataFrame(truthLabels,docCollections,opt.processedFile)






if __name__ == '__main__':
    main()
