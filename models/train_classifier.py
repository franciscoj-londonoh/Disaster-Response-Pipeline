# import libraries
import sys
import pandas as pd
import numpy as np
import re

from sqlalchemy import create_engine
import pickle

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger'])

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import *
from nltk.corpus import stopwords  

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin


def load_data(database_filepath):
    """Load sqlite database into dataframes.

    Args:
    messages_filepath: str. Path to messages database
    categories_filepath: str. Path to categories database

    Returns:
    X: dataframe. X matrix
    Y: dataframe. response variable
    category_names: dataframe. Column names for categories
    """
    
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('CleanTable', engine) 
    # 
    X = df.message
    Y = df.iloc[:,4:]
    category_names = df.columns.values[4:]
    # Label 2 is converted to label 0
    Y[Y["related"] == 2] = 0
    
    return X, Y, category_names


def tokenize(text):
    """Split a string into a sequence of tokens.

    Args:
    Text: str. Sentence to be tokenized

    Returns:
    words_lemmed: list. A list of tokenized, stemmed and lemmed words
    """
    
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    stop_words = stopwords.words("english")
    
    # tokenize
    words = word_tokenize(text)
    
    # stemming
    stemmed = [PorterStemmer().stem(w) for w in words]
    
    # lemmatizing
    words_lemmed = [WordNetLemmatizer().lemmatize(w) for w in stemmed if w not in stop_words]
   
    return words_lemmed

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """Transformer to identify if each text starts with a verb.
    
    Returns:
    pd.DataFrame(X_tagged): dataframe. Tagged X dataframe indicating if the sentence starts with a verb (1) or       not (0)
    """
    
    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            if not pos_tags:
                return 0
            else:
                first_word, first_tag = pos_tags[0]
                if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                    return 1
        return 0
    
    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
 
        return pd.DataFrame(X_tagged)


def build_model():
    """Build a model.
 
    Returns:
    model: model. A Random Forest Classifier
    """
    
    # Pipeline of transforms with a final estimator
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', RandomForestClassifier())
    ])
    
    # Parameters of pipeline transforms
    parameters = {
        'features__text_pipeline__vect__ngram_range': [(1, 1)],
        'features__text_pipeline__vect__max_df': [0.5],
        'features__text_pipeline__vect__max_features': [None],
        'features__text_pipeline__tfidf__use_idf': [False],
        'clf__n_estimators': [100],
        'clf__min_samples_split': [2, 3, 4],
        'features__transformer_weights': (
            {'text_pipeline': 1, 'starting_verb': 0.5},
        )
    }
         
    # Search over specified parameter values for an estimator
    model = GridSearchCV(pipeline, param_grid=parameters)
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluate the performance of the trained model.

    Args:
    model: model. Trained model
    X_test: dataframe. X test matrix
    Y_test: dataframe. Response test variable
    category_names: dataframe. Column names for categories
    
    
    Outputs:
    classification_report: string / dict. Report with the main classification metrics
    accuracy_score: float. Accuracy classification score
    f1_score: float or array of float. Balanced F-score or F-measure
    precision_score: float or array of float. Precision score
    recall_score: float or array of float. Recall score
    """
    
    y_pred = model.predict(X_test)
    
    print(category_names)
    for i, j in enumerate(Y_test):
        print(classification_report(Y_test.iloc[:,i], y_pred[:,i] , target_names=category_names))
    
    from sklearn.metrics import accuracy_score
    print(accuracy_score(Y_test, y_pred))

    from sklearn.metrics import f1_score
    print(f1_score(Y_test, y_pred, average='macro'))

    from sklearn.metrics import precision_score
    print(precision_score(Y_test, y_pred, average='macro'))

    from sklearn.metrics import recall_score
    print(recall_score(Y_test, y_pred, average='macro'))


def save_model(model, model_filepath):
    """Save the trained model.

    Args:
    model: model. Trained model
    model_filepath: str. File path to save the model
    
    Outputs:
    model.pkl: pickle file. Saved model in pickle format
    """
    
    # save the classifier
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    """Main function to perform ML pipeline."""
    
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
