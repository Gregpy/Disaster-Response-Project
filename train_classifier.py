import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
import pickle
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score

def load_data(database_filepath):
    
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    
    df = pd.read_sql_table(table_name='InsertTableName', con= engine)
    
    X = df.message.values
    
    Y = df.iloc[:,4:].values

    category_names = df.columns
    
    return X, Y, category_names
    
def tokenize(text):
    
    tokens = word_tokenize(text)
    
    lemmatizer = WordNetLemmatizer()
       
    clean_tokens = []
    
    for tok in tokens:
        
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    
    pipeline = Pipeline([
             ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])), ])),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))])

    parameters = {'clf__estimator__n_estimators': [10, 20],
                'clf__estimator__criterion' :['gini', 'entropy'],
                'clf__estimator__max_depth' : [4,6],
                'clf__estimator__max_features': ['auto', 'sqrt']
                }

    cv = GridSearchCV(pipeline,param_grid=parameters)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
     
    Y_pred = model.predict(X_test)

    for i in range(len(Y_test[0])):
        print("Reports for {}:\n {}".format(category_names[i],classification_report(Y_test[:,i],Y_pred[:,i])))
    
def save_model(model, model_filepath):
    
    cv_filename = 'cv_classifier.pkl'
    
    cv_pkl = open(model_filepath, 'wb')
    
    pickle.dump(model, cv_pkl)
    
    cv_pkl.close()


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
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