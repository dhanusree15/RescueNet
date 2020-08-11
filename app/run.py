import re
import joblib
import json
import plotly
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
# from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Histogram
from sqlalchemy import create_engine


app = Flask(__name__)


stop = stopwords.words('english')


# def tokenize(text):
#     tokens = word_tokenize(text)
#     lemmatizer = WordNetLemmatizer()

#     clean_tokens = []
#     for tok in tokens:
#         clean_tok = lemmatizer.lemmatize(tok).lower().strip()
#         clean_tokens.append(clean_tok)

#     return clean_tokens


def get_top_words(txt, num_words=10):
    tfidf = TfidfVectorizer(stop_words=stop, max_features=num_words)
    tfidf.fit(txt)
    words = tfidf.vocabulary_
    
    for word in words:
        words[word] = txt[txt.str.contains(word)].count()
    return pd.Series(words).sort_values()


def tokenize(text):
    '''
    Steps:
        Lowercase characters
        Remove punctuation
        Tokenize
        Strip white spaces
        Remove stopwords
        Stem words
    '''
    
    # Steps 1 - 3
    tokens = word_tokenize(re.sub(r'[^A-Za-z0-9]', ' ', text.lower()))
    
    # Step 4 - 5
    stopwords_removed = [word.strip() for word in tokens if word.strip() not in stop]
    
    # Step 6
    stemmer = SnowballStemmer('english')
    return [stemmer.stem(word) for word in stopwords_removed]


# Load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)


# Load model
model = joblib.load("../models/classifier.pkl")


# Home page
@app.route('/')
@app.route('/index')
def index():
    
    # Genre counts
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # Message word counts
    word_counts = df.message.apply(lambda s: len(s.split()))
    word_counts = word_counts[word_counts <= 100]
    
    # Category counts
    cat_counts = df.iloc[:, 4:].sum().sort_values()[-10:]
    cat_names = cat_counts.index.tolist()
    
    # Top word counts
    top_counts = get_top_words(df.message)
    top_words = top_counts.index.tolist()
    
    # Create visuals
    graphs = [
        
        # Genre counts
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],
            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {'title': "Number of messages"},
                'xaxis': {'title': "Genre"}
            }
        },
        
        # Message word counts
        {
            'data': [
                Histogram(
                    x=word_counts
                )
            ],
            'layout': {
                'title': 'Distribution of Message Word Counts',
                'yaxis': {'title': "Number of messages"},
                'xaxis': {'title': "Word count"}
            }
        },
        
        # Category counts
        {
            'data': [
                Bar(
                    x=cat_counts,
                    y=cat_names,
                    orientation='h'
                )
            ],
            'layout': {
                'title': 'Top Message Categories',
                'yaxis': {'title': "Category"},
                'xaxis': {'title': "Number of Messages"},
                'margin': {'l': 100}
            }
        },
        
        # Top word counts
        {
            'data': [
                Bar(
                    x=top_counts,
                    y=top_words,
                    orientation='h'
                )
            ],
            'layout': {
                'title': 'Most Common Words Found in Messages',
                'yaxis': {'title': "Word"},
                'xaxis': {'title': "Number of Messages"},
                'margin': {'l': 100}
            }
        }
    ]
    
    # Encode visuals in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Render home page
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# Classification results page
@app.route('/go')
def go():
    
    # Save user input
    query = request.args.get('query', '') 

    # Classify message
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # Render the results page
    return render_template('go.html', query=query, classification_result=classification_results)


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()