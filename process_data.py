import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    
    """
    INPUT
    messages_filepath - the file path where the csv of messages is located
    categories_filepath - the file path where the csv of messages is located
    
    OUTPUT
    df - pandas dataframe with messages and categories
    
    Perform to obtain df
    This function takes messages_filepath and categories_filepath 
    and does the following steps to produce df
    1. Read the csv from messages_filepath as messages and categories_filepath as categories
    2. Merge messages and categories on id into dataframe df and return df
    
    """ 
    
    messages = pd.read_csv(messages_filepath)
    
    categories = pd.read_csv(categories_filepath)

    df = messages.merge(categories, on='id')
    
    return df
    
def clean_data(df):

     """
    INPUT
    df - pandas dataframe containing messages and categories
    
    OUTPUT
    df - pandas dataframe with messages and categories cleaned
    
    Perform to obtain df
    This function takes df and does the following steps to produce df
    1. Read the df categories column splitting along ; as categories
    2. Takes the values from a row and removes non letter values as category_colnames
    3. For each column the last value is taken for each row (0 or 1)
    4. Categories replaces the original in df
    5. Duplicates are dropped in df
    
    """ 
    
    categories = df['categories'].str.split(';', expand=True)
    
    row = categories.iloc[0,:]
    
    category_colnames = row.apply(lambda x: str(x)[:-2])
    
    categories.columns = category_colnames 
    
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).apply(lambda x: str(x)[-1])
    
    # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    df = df.drop('categories', axis=1)
    
    df = pd.concat([df,categories],axis=1,sort=True)
    
    df = df.drop_duplicates()

    return df
    
def save_data(df, database_filename):
    
    """
    INPUT
    df - pandas dataframe containing cleaned messages and categories
    database_filename - the filename of where the SQL database will be
    OUTPUT
    SQL database with table 'InsertTableName' with data from df
    
    Perform to obtain SQL database
    This function takes df and does the following steps to produce a SQL database 
    1. Create a sqlite engine
    2. Use pandas to send df to SQL
    
    """ 
    
    engine = create_engine('sqlite:///{}'.format(database_filename))

    df.to_sql('InsertTableName', engine, index=False) 

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()