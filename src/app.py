import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import warnings

warnings.filterwarnings('ignore')

# Ensure required NLTK data packages are downloaded
nltk.download('stopwords')
nltk.download('punkt')

# Set of Arabic stopwords
arabic_stopwords = set(stopwords.words('arabic'))

# Define text preprocessing functions
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

le = pickle.load(open('le.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server 
app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                html.H1("Arabic Dialect Classifier"),
                width={"size": 6, "offset": 3},
                className="text-center mt-4"
            )
        ),
        dbc.Row(
            dbc.Col(
                dcc.Textarea(
                    id='input-text',
                    style={'width': '100%', 'height': 200},
                    placeholder='Enter an Arabic sentence here...',
                ),
                width={"size": 6, "offset": 3},
                className="mt-4"
            )
        ),
        dbc.Row(
            dbc.Col(
                dbc.Button("Classify", id='submit-button', color='primary', className='mt-3'),
                width={"size": 6, "offset": 3},
                className="text-center"
            )
        ),
        dbc.Row(
            dbc.Col(
                html.H1(id='output-class', className='text-center mt-4'),
                width={"size": 6, "offset": 3}
            )
        )
    ],
    fluid=True
)

@app.callback(
    Output('output-class', 'children'),
    Input('submit-button', 'n_clicks'),
    State('input-text', 'value')
)
def classify_text(n_clicks, input_text):
    if n_clicks is None or not input_text:
        return ''
    
    preprocessed_text = preprocess_text(input_text)
    
    transformed_text = tfidf.transform([preprocessed_text])
    
    prediction = model.predict(transformed_text)
    
    class_name = le.inverse_transform(prediction)[0]
    
    return f'The dialect is: {class_name}'

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

if __name__ == '__main__':
    app.run_server(debug=True)
