# Disaster-Response-Project

## Installation

All libaries are included in Anaconda distribution. The code works with Python versions 3.*.

## Project Motivation

Figure 8 data of tweets from distaster areas was used to check for patterns in the data that would lead to knowledge of the urgency of the tweets, by knowing the category of tweet they would fall under.

The results showed correlations of messages with the types of messages they could convey.

## File Descriptions

The file process_data.py loads, cleans the data, and saves it as an SQL database. The file train_classifier.py loads the data from the SQL database, trains, predicts and evaluates the results, and pickles and saves the model. The file run.py creats a web app that can be used to classify tweet messages.

## How to Interact with the Project

From a terminal prompt at the location of relevent files for data training in the zip folder, one can run

* 'python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db' to process the data
* 'python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl' to train the data
* 'python run.py' to run the web app where messages can be classified into relevant categories 

## Licensing and Acnknowledgements
The data was collected by Figure 8. Licensing and and information on the data can be found on Figure 8 [here](https://www.figure-eight.com/). Feel free to use the code otherwise.

