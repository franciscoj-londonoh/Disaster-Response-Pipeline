# Disaster-Response-Pipeline
This project builds a model for an API that classifies disaster messages from a data set containing real messages that were sent during disaster events. The project creates a machine learning pipeline to categorize these events so that messages can be sent to an appropriate disaster relief agency. The project includes a web app not only displays visualizations of the data but also allows an emergency worker to input a new message and get classification results in several categories. 


## Project Instructions
#### 1. Clone Project and Install requirements 
```
git clone https://github.com/franciscoj-londonoh/Disaster-Response-Pipeline.git
cd Disaster-Response-Pipeline
pip install -r requirements.txt
```

#### 2. Run the following commands in the project's root directory to set up the database and model:

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

#### 3. Run the following command in the app's directory to run your web app:
    `python run.py`

#### 4. Go to http://0.0.0.0:3001/


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

```
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
```
