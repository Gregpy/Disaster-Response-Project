# Disaster-Response-Project

## Installation

All libaries are included in Anaconda distribution. The code works with Python versions 3.*.

## Project Motivation

Figure 8 data of tweets from distaster areas was used to check for patterns in the data that would lead to knowledge of the urgency of the tweets, by knowing the category of tweet they would fall under.

The results showed correlations of messages with the types of messages they could convey.

## File Descriptions

The file process_data.py loads, cleans the data, and saves it as an SQL database. The file train_classifier.py loads the data from the SQL database, trains, predicts and evaluates the results, and pickles and saves the model. The file run.py creats a web app that can be used to classify tweet messages.

## How to Interact with the Project

From a terminal prompt, with relevent files for data training in the zip folder, run process_data.py to process the data. Then run train_classifier.py to train the data. And finally run run.py to run the web app where messages can be classified into relevant categories. 

## Licensing and Acnknowledgements
[here](https://www.figure-eight.com/)
