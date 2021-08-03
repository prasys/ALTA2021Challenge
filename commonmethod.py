from bs4 import BeautifulSoup as bs
import re
from django.utils.html import strip_tags
import argparse
import os,sys
import pandas as pd
from sklearn import preprocessing
from cleantext import clean
import spacy
import scispacy
from spacy import displacy

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

def parseAbstracts(fileName,tag,cleanText=True):
    s = ""
    texts = []
    abstracts = loadXMLcontents(fileName,tag)
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

def loadXMLcontents(fileName,tag):
    infile = open(fileName,"r", encoding="utf8")
    contents = infile.read()
    soup = bs(contents,'xml')
    content = soup.find_all(tag)
    return content

def loadFileAndParse(filename,XMLFolderPath,truth,docCollections,cleanText,tag):
    file1 = open(filename, 'r')
    for line in file1:
        temp = line.split()
        if (len(temp)-2) == 1: #if we only have 1 document collection
            truth.append(temp[1])
            docCollections.append(parseAbstracts(XMLFolderPath+temp[2]+".xml",tag,cleanText)) #parseXML
        elif (len(temp)-2) > 1:
                for count in range(2,len(temp)):
                    truth.append(temp[1])
                    docCollections.append(parseAbstracts(XMLFolderPath+temp[count]+".xml",tag,cleanText)) #parseXML  
    file1.close()


def createAndSaveDataFrame(truthLabels,docCollections,fileName):
    le = preprocessing.LabelEncoder()
    truthLabels = le.fit_transform(truthLabels)
    df = pd.DataFrame(list(zip(docCollections,truthLabels)),columns =['text', 'labels'])
    print(df.head())
    df.to_pickle(fileName)
    df.to_csv(fileName+".csv")