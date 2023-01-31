import streamlit as sl
import pickle
import nltk
import string
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
stopwords = nltk.corpus.stopwords.words('english')


model = pickle.load(open('model.pkl','rb'))
vec = pickle.load(open('vectorizer.pkl','rb'))





def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
            
    return " ".join(y)


sl.title('Email/SMS Spam Classifier')

input_sms = sl.text_area('Enter message')

if sl.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = vec.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        sl.header("Spam")
    else:
        sl.header("Not Spam")
