from bs4 import BeautifulSoup as bs
import re
from django.utils.html import strip_tags
import argparse

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

def parseAbstracts(fileName):
    abstracts = loadXMLcontents(fileName)
    if len(abstracts) == 1:
        s = parse_text(titles[0])
    else:
        for abstract in abstracts:
            texts.append(parse_text(abstract))
            s = ''.join(texts)
    return s

def loadXMLcontents(fileName,contents="abstracttext"):
    infile = open(fileName,"r")
    contents = infile.read()
    soup = bs(contents,'xml')
    content = soup.find_all(contents)
    return content

def loadFileAndParse(filename,XMLFolderPath,truth,docCollections):
    file1 = open(filename, 'r')
    for line in file1:
        temp = line.split()
        if (len(temp)-2) == 1: #if we only have 1 document collection
            truth.append(temp[1]) #append the grade
            docCollections.parseAbstracts(XMLFolderPath+temp[2]+".xml") #parseXML
        elif (len(temp)-2) > 1:
                for count in range(2,len(temp)):
                    truth.append(temp[1])
                    docCollections.parseAbstracts(XMLFolderPath+temp[count]+".xml") #parseXML  
    file1.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default='trainset.txt', type=str)
    parser.add_argument('--XMLdir', default='../trainset/', type=str)
    opt = parser.parse_args()
    docCollections = []
    truthLabels = []
    print("file is",opt.file)
    print("dir is",opt.XMLdir)
    loadFileAndParse(opt.file,opt.XMLdir,truthLabels,docCollections)
    print(docCollections)





if __name__ == '__main__':
    main()