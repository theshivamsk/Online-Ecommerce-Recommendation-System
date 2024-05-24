import pandas as pd
import numpy as np
import nltk
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from PIL import Image

# Load the dataset
data = pd.read_csv('OnlineRetail.csv')

# Remove unnecessary columns
#data = data.drop('id', axis=1)

# Fill NaN values with empty strings in 'Country' and 'Description' columns
data['Country'] = data['Country'].fillna('')
data['Description'] = data['Description'].fillna('')

# Define tokenizer and stemmer
stemmer = SnowballStemmer('english')
def tokenize_and_stem(text):
    tokens = nltk.word_tokenize(text.lower())
    stems = [stemmer.stem(t) for t in tokens]
    return stems

# Create stemmed tokens column
data['stemmed_tokens'] = data.apply(lambda row: tokenize_and_stem(row['Country'] + ' ' + row['Description']), axis=1)

# Define TF-IDF vectorizer and cosine similarity function
tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize_and_stem)
def cosine_sim(text1, text2):
    # tfidf_matrix = tfidf_vectorizer.fit_transform([text1, text2])
    text1_concatenated = ' '.join(text1)
    text2_concatenated = ' '.join(text2)
    tfidf_matrix = tfidf_vectorizer.fit_transform([text1_concatenated, text2_concatenated])
    return cosine_similarity(tfidf_matrix)[0][1]

# Define search function
def search_products(query):
    query_stemmed = tokenize_and_stem(query)
    data['similarity'] = data['stemmed_tokens'].apply(lambda x: cosine_sim(query_stemmed, x))
    results = data.sort_values(by=['similarity'], ascending=False).head(10)[['InvoiceNo', 'StockCode', 'Description','Quantity','InvoiceDate','UnitPrice','CustomerID','Country']]
    return results

# web app
img = Image.open('img2.jpg')
st.image(img,width=400)
st.title("Online Retail Recommendation System")
query = st.text_input("Enter Product Details")
sumbit = st.button('Search')
if sumbit:
    res = search_products(query)
    st.write(res)