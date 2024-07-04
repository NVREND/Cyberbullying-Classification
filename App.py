import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import demoji
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from langdetect import detect, LangDetectException
import contractions
from nltk.tokenize import word_tokenize
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.models import load_model

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Inisialisasi lemmatizer untuk text cleaning
lemmatizer = WordNetLemmatizer()

# Clean emojis from text
def remove_emojis(text):
    return demoji.replace(text, '')

# Define stop words for text cleaning
stop_words = set(stopwords.words('english'))

def strip_all_entities(text):
    text = re.sub(r'\r|\n', ' ', text.lower())  # Ganti baris baru dengan spasi, dan ubah menjadi lowercase
    text = re.sub(r"(?:\@|https?\://)\S+", "", text)  # Hapus link dan mention
    text = re.sub(r'[^\x00-\x7f]', '', text)  # Hapus non-ASCII char
    text = re.sub(r'\d+', '', text)  # Hapus angka
    text = re.sub(r'#\S+', '', text)  # Hapus hashtags
    banned_list = string.punctuation
    table = str.maketrans('', '', banned_list)
    text = text.translate(table)
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

def filter_chars(text):
    return ' '.join('' if ('$' in word) or ('&' in word) else word for word in text.split())

def remove_multi_spaces(text):
    return re.sub(r"\s\s+", " ", text)

def filter_non_english(text):
    try:
        lang = detect(text)
    except LangDetectException:
        lang = "unknown"
    return text if lang == "en" else ""

def expand_contractions(text):
    return contractions.fix(text)

def lemmatize(text):
    words = word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

def remove_short_words(text, min_len=2):
    words = text.split()
    long_words = [word for word in words if len(word) >= min_len]
    return ' '.join(long_words)

def replace_elongated_words(text):
    regex_pattern = r'\b(\w+)((\w)\3{2,})(\w*)\b'
    return re.sub(regex_pattern, r'\1\3\4', text)

def remove_repeated_punctuation(text):
    return re.sub(r'[\?\.\!]+(?=[\?\.\!])', '', text)

def remove_extra_whitespace(text):
    return ' '.join(text.split())

def remove_url_shorteners(text):
    return re.sub(r'(?:http[s]?://)?(?:www\.)?(?:bit\.ly|goo\.gl|t\.co|tinyurl\.com|tr\.im|is\.gd|cli\.gs|u\.nu|url\.ie|tiny\.cc|alturl\.com|ow\.ly|bit\.do|adoro\.to)\S+', '', text)

def remove_spaces_tweets(tweet):
    return tweet.strip()

def remove_short_tweets(tweet, min_words=3):
    words = tweet.split()
    return tweet if len(words) >= min_words else ""

def clean_tweet(tweet):
    tweet = remove_emojis(tweet)
    tweet = expand_contractions(tweet)
    tweet = filter_non_english(tweet)
    tweet = strip_all_entities(tweet)
    tweet = filter_chars(tweet)
    tweet = remove_multi_spaces(tweet)
    tweet = lemmatize(tweet)
    tweet = remove_short_words(tweet)
    tweet = replace_elongated_words(tweet)
    tweet = remove_repeated_punctuation(tweet)
    tweet = remove_extra_whitespace(tweet)
    tweet = remove_url_shorteners(tweet)
    tweet = remove_spaces_tweets(tweet)
    tweet = remove_short_tweets(tweet)
    tweet = ' '.join(tweet.split())  # Menghapus spasi tambahan di antara kata
    return tweet

def user_interface():
    # Web app UI - title and input box for the question
    st.title('🌠 CyberBullying Detection')
    st.text('Capstone Project')
    user_tweet = st.text_input('Input your tweet here:')

    final_prompt = clean_tweet(user_tweet)
    return user_tweet, final_prompt

def answer_questions():
    # Call user_interface to get user input and processed prompt
    user_tweet, final_prompt = user_interface()
    
    # Load vocabulary from vocab.txt
    vocab = []
    with open('vocab.txt', 'r') as f:
        vocab = f.read().splitlines()

    # Define the TextVectorization layer
    vectorization = TextVectorization(
        max_tokens=10000,  # Maximum vocabulary size
        output_mode='int',  # Output integers
        output_sequence_length=100,  # Pad or truncate sequences to a fixed length
        standardize='lower_and_strip_punctuation',  # Convert text to lowercase and remove punctuation
        split='whitespace',  # Split text by whitespace
        vocabulary=vocab
    )
    
    # Convert text to vector
    new_text_vec = vectorization([final_prompt])

    # Load the pre-trained model
    model = load_model('cyberbullying_model.h5')

    # Predict using the model
    predictions = model.predict(new_text_vec)
    prediction = predictions[0]

    # Define label mapping
    label_mapping = {0: 'religion', 1: 'age', 2: 'ethnicity', 3: 'gender', 4: 'not_cyberbullying'}

    # Find the index of the maximum value in the list
    max_index = np.argmax(prediction)

    # Map the index to the corresponding label using the label mapping
    output = label_mapping[max_index]

    # Display output on the Web page
    formatted_output = f"""
        **Detection to your Tweet:** {user_tweet} \
        *{output}*</i>
        """
    st.markdown(formatted_output, unsafe_allow_html=True)

# Invoke the main function
answer_questions()
