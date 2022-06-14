import csv
import pandas as pd
import numpy as np
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix

#for split list
def convert(lst):
    return ' '.join(lst).split()


#Lancaster_stemmer
def lancaster_stemmer(word):

    lancaster=LancasterStemmer()
    lancasters=[]

    for i in range(0,len(word)):
        lancasters.append(lancaster.stem(word[i]))
        
    return lancasters



#porter_stemmer
def porter_stemmer(word):

    porter = PorterStemmer()
    porters=[]

    for i in range(0,len(word)):
        porters.append(porter.stem(word[i])) 
            
    return porters



#Lenght_each_of_word
def each_len_of_word(new_data):
    lenght=[]
    for y in range(0,len(new_data)):
        lenght.append(len(new_data[y]))
    return lenght



#stopwords
def stopword(roots):
    check_stop_word=[]

    stop_words=stopwords.words('english','French')
    stop_words.append(stopwords.words('Spanish','German'))

    for s in roots:
        if s not in stop_words:
            check_stop_word.append(s)
        else:
            check_stop_word.append('-')
    return check_stop_word


#find_word_from_sentence
def each_word(data):

    content=data["content"]

    split=[]

    for i in range(0,len(data)):
        lst = content[i]
        #print('\n',content[i],'\n')
        split.append(lst.split())
        
    text=[]
    word=[]

    for y in range(0,len(split)):
        text=split[y]
        #print('\n',word,'\n')
        for z in range(0,len(text)):
            s=text[z].replace("'","")
            s1=s.replace(".","")
            s2=s1.replace("(","")
            s3=s2.replace(")","")
            s4=s3.replace(" ","")
            s5=s4.replace("%","")
            s6=s5.replace("/","")
            word.append(s6)
    return word     



#read_data
data = pd.read_csv('dataMining.csv' , encoding="ISO-8859-1", on_bad_lines='skip')


word=each_word(data)
roots_porter=porter_stemmer(word)
roots_lancaster=lancaster_stemmer(word)

roots_porter_convert=stopword(roots_porter)
roots_lancaster_convert=stopword(roots_lancaster)

#create_csv_file_and_print_result
df = pd.DataFrame(data={"each word":each_word(data),"porter of each word":porter_stemmer(word),"lancaster of each word":lancaster_stemmer(word),"each len of word after porter": each_len_of_word(roots_porter),"each len of word after lancaster": each_len_of_word(roots_lancaster),"porter data without stop words":stopword(roots_porter),"lancaster data without stop words":stopword(roots_lancaster)})
df.to_csv("./features.csv", sep=',',index=False)



