import streamlit as st
import pickle


# Preporcessing
import re

def custom_word_tokenize(text):
    if not isinstance(text, str):
        return []

    # Preserve ellipsis as one token
    text = text.replace('...', ' <ELLIPSIS> ')

    # Tokenize using regex: keep words, contractions, punctuation
    tokens = re.findall(r"\w+(?:'\w+)?|[^\w\s]", text)

    # Replace <ELLIPSIS> placeholder back
    tokens = ['...' if token == '<ELLIPSIS>' else token for token in tokens]

    return tokens

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
import string
# string.punctuation



def transform_text(text):
    text = text.lower()
    text = custom_word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()
    for i in text :
        y.append(ps.stem(i))
    return " ".join(y)









tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title('Email/SMS Spam Classifier')

input_sms = st.text_area('Enter the message')

if st.button("Predict"):

    # Preprocessing
    transformed_sms = transform_text(input_sms)

    # Vectorize
    vector_input = tfidf.transform([transformed_sms])

    # predict
    result = model.predict(vector_input)[0]

    # Display
    if result ==1:
        st.header("Spam")
    else:
        st.header("Not Spam")