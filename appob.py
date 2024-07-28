import streamlit as st
import pandas as pd
import re
import string
import demoji
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
#from langdetect import detect, LangDetectException
import contractions
from nltk.tokenize import word_tokenize
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.models import load_model

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Initialize NLTK components
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Function to clean and preprocess tweet
def strip_all_entities(tweet):
    tweet = demoji.replace(tweet, '')  # Remove emojis
    tweet = contractions.fix(tweet)   # Expand contractions
    tweet = re.sub(r'\d+', '', tweet) # Remove numbers
    tweet = re.sub(r'@\w+', '', tweet) # Remove mentions
    tweet = re.sub(r'http\S+', '', tweet) # Remove URLs
    tweet = re.sub(r'#\w+', '', tweet) # Remove hashtags
    tweet = re.sub(r'[^\x00-\x7f]', '', tweet)  # Remove non-ASCII characters
    tweet = re.sub(r'\s+', ' ', tweet).strip()  # Remove extra whitespaces
    return tweet

def filter_chars(text):
    return ' '.join('' if ('$' in word) or ('&' in word) else word for word in text.split())

def remove_multi_spaces(text):
    return re.sub(r"\s\s+", " ", text)
def remove_extra_whitespace(text):
    return ' '.join(text.split())
# Menhapus spasi di awal dan diakhir tweet
def remove_spaces_tweets(tweet):
    return tweet.strip()

def clean_tweet(tweet):
    tweet = strip_all_entities(tweet)
    tweet = filter_chars(tweet)
    tweet = remove_multi_spaces(tweet)
    tweet = lemmatize(tweet)
    tweet = remove_extra_whitespace(tweet)
    tweet = remove_spaces_tweets(tweet)
    tweet = ' '.join(tweet.split())  # Menghapus spasi tambahan diantara kata
    return tweet

# Function to tokenize and lemmatize text
def lemmatize(text):
    words = word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

# UI function using Streamlit
def user_interface():
    st.title('🌠 CyberBullying Detection')
    st.text('Capstone Project')
    user_tweet = st.text_input('Input your tweet here:')
    return user_tweet

# Function to process user input and predict using model
def predict_tweet(user_tweet):
    # Clean and preprocess user input
    cleaned_tweet = clean_tweet(user_tweet)
    lemmatized_tweet = lemmatize(cleaned_tweet)
    #final_tweet = filter_non_english(lemmatized_tweet)

    # Load vocabulary from vocab.txt
    vocab = []
    with open('vocab2.txt', 'r') as f:
        vocab = f.read().splitlines()

    # Define TextVectorization layer
    vectorization = TextVectorization(
        max_tokens=10000,
        output_mode='int',
        output_sequence_length=100,
        standardize='lower_and_strip_punctuation',
        split='whitespace',
        vocabulary=vocab
    )

    # Vectorize the tweet
    vectorized_tweet = vectorization([lemmatized_tweet])

    # Load the pre-trained model
    model = load_model('model_fix.h5')

    # Predict using the model
    prediction = model.predict(vectorized_tweet)[0]

    # Define label mapping
    label_mapping = {0: 'religion', 1: 'age', 2: 'ethnicity', 3: 'gender', 4: 'not_cyberbullying'}

    return prediction, label_mapping

# Main function to run the application
def main():
    user_tweet = user_interface()
    if st.button('Detect'):
        prediction, label_mapping = predict_tweet(user_tweet)
        
        # Display the predicted category with the highest probability
        max_index = prediction.argmax()
        predicted_label = label_mapping[max_index]
        st.success(f'Predicted Category: {predicted_label}')

        # Display the probability for each category
        # st.subheader('Prediction Probabilities:')
        # for i, prob in enumerate(prediction):
        #    st.write(f"{label_mapping[i]}: {prob:.2f}")

if __name__ == "__main__":
    main()
