import PyPDF2
import textract
import nltk
import gzip
import gensim
import logging
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet

def read_input(input_file):
    for line in input_file:
        yield gensim.utils.simple_preprocess(line)

filename = 'C:\\Users\\arjun\\Desktop\\Resume\\06 - High Res PDF\\Arjun Vasudevan - Resume.pdf'

pdfFileObj = open(filename,'rb')

pdfReader = PyPDF2.PdfFileReader(pdfFileObj)

num_pages = pdfReader.numPages
count = 0
text = ""

while count < num_pages:
    pageObj = pdfReader.getPage(count)
    count +=1
    text += pageObj.extractText()

if text != "":
   text = text
else:
   text = textract.process(filename, method='tesseract', language='eng')

tokens = word_tokenize(text)
punctuations = ['(',')',';',':','[',']',',']
stop_words = stopwords.words('english')
keywords = [word for word in tokens if not word in stop_words and not word in punctuations and wordnet.synsets(word)]

documents = list(read_input(keywords))
model = gensim.models.Word2Vec(documents,size=150,window=10,min_count=2,workers=10)
model.train(documents, total_examples=len(documents), epochs=10)

w1 = ["Award"]
print(
    "Most similar to {0}".format(w1),
    model.wv.most_similar(
        positive=w1,
        topn=6))
