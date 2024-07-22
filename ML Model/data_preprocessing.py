# data_preprocessing.py

import re
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')

arabic_stopwords = set(stopwords.words('arabic'))

def remove_emojis(text):
    emoji_pattern = re.compile(
        "["                       
        "\U0001F600-\U0001F64F"  
        "\U0001F300-\U0001F5FF"  
        "\U0001F680-\U0001F6FF"  
        "\U0001F1E0-\U0001F1FF"  
        "\U00002700-\U000027BF"  
        "\U000024C2-\U0001F251"  
        "]+", flags=re.UNICODE   
    )
    return emoji_pattern.sub(r'', text)

def normalize_arabic(text):
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "و", text)
    text = re.sub("ئ", "ي", text)
    text = re.sub("ة", "ه", text)
    return text

def remove_diacritics(text):
    arabic_diacritics = re.compile("""
                             ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                         """, re.VERBOSE)
    return re.sub(arabic_diacritics, '', text)

def remove_mentions(text):
    return re.sub(r'@\w+', '', text)

def remove_hashtags(text):
    return re.sub(r'#\w+', '', text)

def remove_urls(text):
    return re.sub(r'http\S+', '', text)

def remove_non_arabic(text):
    return re.sub(r'[^\u0621-\u064A\s]', '', text)

def remove_english(text):
    return re.sub(r'[a-zA-Z]', '', text)

def preprocess_text(text):
    text = remove_emojis(text)
    text = remove_mentions(text)
    text = remove_hashtags(text)
    text = remove_urls(text)
    text = remove_non_arabic(text)
    text = remove_english(text)
    text = normalize_arabic(text)
    text = re.sub(r'[^\w\s]', '', text)
    text = remove_diacritics(text)
    words = word_tokenize(text)
    words = [word for word in words if word not in arabic_stopwords]
    return ' '.join(words)

def preprocess_dataframe(df):
    df['text'] = df['text'].apply(preprocess_text)
    df = df.drop('id', axis=1)
    return df
