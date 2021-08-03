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
nlp = spacy.load('en_ner_bc5cdr_md')
nlp2 = spacy.load("en_core_sci_md") #en_core_sci_md #en_core_web_sm
nlp3 = spacy.load("en_core_web_sm")


def replaceNER(model,s):
    doc = model(s)
    return(" ".join([t.text if not t.ent_type_ else t.ent_type_ for t in doc]) )


def keepFirstAndLastSentence(text,model=nlp2):
    doc = model(text)
    assert doc.has_annotation("SENT_START")
    a = list(doc.sents)
    l = len(list(doc.sents))
    if l != 0:
        return (a[0].text + a[l-1].text)
    else:
        return ("")
        
    
    
def normalize(comment, lowercase, remove_stopwords):
    if lowercase:
        comment = comment.lower()
    comment = nlp2(comment)
    lemmatized = list()
    for word in comment:
        lemma = word.lemma_.strip()
        if lemma:
            if not remove_stopwords:
                lemmatized.append(lemma)
    string = " ".join(lemmatized)
    return string

def clean_text(text):
    # text = re.sub("[\(\[].*?[\)\]]", "", text)
    # text = keepFirstAndLastSentence(text)
    return text
    #return(replaceNER(nlp,normalize(text,lowercase=True,remove_stopwords=False)))
    #return text
    # return (replaceNER(nlp,replaceNER(nlp3,normalize(text,lowercase=True,remove_stopwords=False))))


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default='trainset.txt', type=str)
    parser.add_argument('--XMLdir', default='/trainset/', type=str)
    parser.add_argument('--processedFile', default='output.h5', type=str)
    parser.add_argument('--cleanText', default=True, type=bool)
    parser.add_argument('--XMLTag', default="abstracttext", type=str)
    opt = parser.parse_args()
    xmlPath = os.path.abspath(os.path.dirname(os.path.abspath(__file__))) + opt.XMLdir
    docCollections = []
    truthLabels = []
    loadFileAndParse(opt.file,xmlPath,truthLabels,docCollections,opt.cleanText,opt.XMLTag)
    createAndSaveDataFrame(truthLabels,docCollections,opt.processedFile)






if __name__ == '__main__':
    main()
