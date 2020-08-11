import sys
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

import joblib
import re
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('SELECT * FROM messages', engine)
    X = df['message'].copy()
    Y = df.iloc[:, 4:].copy()
    categories = Y.columns.tolist()
    return X, Y, categories


stop = stopwords.words('english')
    
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


def build_model():
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=tokenize, ngram_range=(1, 1), max_df=0.2, max_features=1000)),
        ('clf', MultiOutputClassifier(LogisticRegression(max_iter=1000, random_state=0)))
    ])
    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    pred = pd.DataFrame(model.predict(X_test), columns=category_names)
    
    for col in category_names:
        print('-' * 53)
        print(f'Label: {col}')
        print(f'Accuracy: {accuracy_score(Y_test[col], pred[col])}')
        print(classification_report(Y_test[col], pred[col]))
        
    return pred
        

def save_model(model, model_filepath):
    joblib.dump(model, model_filepath)
    return


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()