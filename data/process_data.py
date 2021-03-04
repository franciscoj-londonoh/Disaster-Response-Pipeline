# import libraries
import sys
import pandas as pd
import numpy as np

from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Load and merge databases into a dataframe.

    Args:
    messages_filepath: str. Path to messages database
    categories_filepath: str. Path to categories database

    Returns:
    pd.merge(messages, categories, on="id"): merged dataframe. A dataframe with merged information (by id) of         messages and categories 
    """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    return pd.merge(messages, categories, on="id")


def clean_data(df):
    """Clean dataframe.

    Args:
    df: dataframe. The dataframe to be cleaned

    Returns:
    df: dataframe. Cleaned dataframe 
    """
    
    # extract and set category column names
    categories = df.categories.str.split(";", expand=True)
    row = categories.iloc[0, :]
    category_colnames = list(row.apply(lambda x: x[:-2]))
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
    
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    df = df.drop(columns=["categories"])     
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1, join='inner', sort=False)

    # check number of duplicates
    df.duplicated().value_counts()
    # drop duplicates
    df = df.drop_duplicates()
    
    return df
    
def save_data(df, database_filename):
    """Save dataframe.

    Args:
    df: dataframe. The dataframe to be saved
    database_filename: str. File name

    Output:
    df: sqlite database. Saved database 
    """
    
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('CleanTable', engine, index=False)  


def main():
    """Main function to perform ETL pipeline."""
    
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print(df)
        
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
