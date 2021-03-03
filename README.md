# Disaster-Response-Pipeline


## Project Components
There are three components you'll need to complete for this project.

## 1. ETL Pipeline
The Python script process_data.py presents a data cleaning pipeline that:

* Loads the messages and categories datasets
* Merges the two datasets
* Cleans the data
* Stores it in a SQLite database

## 2. ML Pipeline
The Python script, train_classifier.py, presents a machine learning pipeline that:

* Loads data from the SQLite database
* Splits the dataset into training and test sets
* Builds a text processing and machine learning pipeline
* Trains and tunes a model using GridSearchCV
* Outputs results on the test set
* Exports the final model as a pickle file

## 3. Flask Web App
The flask web app:

* Visualizes statistics of the dataset
* Receives input messages from the user and classifies it
* Shows the classification results

## Structure of the Project

- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- InsertDatabaseName.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

- README.md
