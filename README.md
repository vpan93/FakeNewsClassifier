# Fake news detection

This is a fake news classifier created using a modification of the [LIAR](https://paperswithcode.com/dataset/liar) data set. The dataset consists of thousands of statements (quotes) made by politicians and other public figures. The data includes various information on each statement and on the person who made It. The truthfulness of each statement was also evaluated by expert human editors, who then assigned an appropriate truthfulness label.

All the data is included in the provided data.csv file.The file includes one comma-separated line for each statement. The line includes the following attributes (columns) for each statement:
* label: a truthfulness label assigned by human annotators. Possible values (from most to least truthful) are true, mostly-true, half-true, barely-true, false, extremely-false
* statement: the text of the actual statement that is being evaluated
* subjects: a list of topics related to the statement. These topics are separated by a ‘$’ sign
* speaker_name: the name of the person who made the statement
* speaker_job: the job title of the person who made the statement
* speaker_state: the US state associated with the person who made statement
* speaker_affiliation: the political or professional affiliation of the person who made the statement
* statement_context: the context in which the statement was made 

The code creates a Python package that implements a truthfulness classifier for statements such as those found in data.csv. The package is accompanied by brief instructions on how to install and use that you can find in the "app" [folder](/FakeNewsClassifierPackage/app/README.md). Please have a look in the powerpoint presentation as well to get a better understanting of the methodology followed for exploration and the final model that was implemented for the project.

The user can run the app and upload csv file for preprocessing and model training and a json file for testing predictions. The model returns a True/False prediction and a short explanation that justifies the prediction.


## Author

Vasilis Panagaris
