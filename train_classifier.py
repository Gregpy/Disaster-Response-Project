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
    
    """
    INPUT
    database_filepath - The filepath of where the SQL database is
    OUTPUT
    X - Message values from the database
    Y - Binary values corresponding to the type of message categories
    category_names - categories of messages
    
    Perform to obtain X, Y, and category_names 
    This function takes database_filepath  
    and does the following steps to produce X, Y, and category_names 
    1. Create a sqlite engine from database_filepath
    2. Read the SQL database table 'InsertTableName' to a pandas dataframe df
    3. Take the message column values as X
    4. Take the numerical values as Y (from column 4 on)
    5. Take the column names from df as category_names 
    
    """ 
    
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    
    df = pd.read_sql_table(table_name='InsertTableName', con= engine)
    
    X = df.message.values
    
    Y = df.iloc[:,4:].values

    category_names = df.columns
    
    return X, Y, category_names
    
def tokenize(text):
    
    """
    INPUT
    text - Text to tokenize
    OUTPUT
    clean_tokens - The text cleaned and tokenized
    
    Perform to obtain clean tokens
    This function takes text and does the following steps to produce clean tokens
    1. Uses nltk to case normalize, lemmatize, and tokenize the text.
    
    """ 
        
    tokens = word_tokenize(text)
    
    lemmatizer = WordNetLemmatizer()
       
    clean_tokens = []
    
    for tok in tokens:
        
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    
    """
    INPUT
    -
    OUTPUT
    cv - A cross validation model
    
    Perform to obtain cv
    This function does the following steps to produce cv
    1. Creates a pipeline of tokenizer, transformer and classifier
    2. Select parameters to use in cv
    3. Creates cv from the pipeline and parameters
    
    """ 
    
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
    
    """
    INPUT
    model - Model to use for ML
    X_test - Messages in the test data
    Y_test - Binary values for type of message
    category_names - names of categories of messages
    OUTPUT
    Prints f1 score, precision, and accuracy of model for each category.
    
    Perform to obtain scores
    This function takes model, X_test, Y_test, and category_names 
    and does the following steps to produce f1 score, precision, and accuracy
    1. Take the predictions from the trained model with X_test as Y_pred
    2. Print the classification report that contains the relevant scores for each column in Y_pred and Y_test
    
    """ 
    
    Y_pred = model.predict(X_test)

    for i in range(len(Y_test[0])):
        print("Reports for {}:\n {}".format(category_names[i],classification_report(Y_test[:,i],Y_pred[:,i])))
    
def save_model(model, model_filepath):
    
    """
    INPUT
    model - Model used for ML
    model_filepath - Filepath to store the model

    OUTPUT
    Pickles and saves the model
    
    Perform to obtain saved pickle model
    This function takes model and model_filepath
    and does the following steps to save the model as a pickle
    1. Take a filename for the pickle
    2. Open the pickle filepath at model_filepath
    3. Dump the model into the opened pickle
    4. Close the pickle
    """ 
    
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