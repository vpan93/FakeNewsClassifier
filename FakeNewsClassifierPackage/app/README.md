## Getting Started

These instructions will guide you on how to install the created Python package and test the model with your own data.

## Project Structure

The `FakeNewsClassifierPackage` is organized as follows:

```
FakeNewsClassifierPackage
│   LICENSE
│   requirements.txt
│   run.py
│   setup.py
└───app
     └──FakeNewsPredictor
            └─── src
                   └───
                   | __init__.py
                   | FakeNewsPredictor.py
                   | utils.py
            └─── test
                   └───
                   | __init__.py
                   | test_FakeNewsPredictor.py 
            __init__.py
        __init__.py
        README.md
    └───results
            └─── predictions.txt 
    
    └───static
            └─── satalia_logo.jpeg
            └─── scripts.js
            └─── styles.css
    └───templates
            └─── index.html
            └─── upload.html
    └───model
            └───latest_trained_model.joblib
```


### Prerequisites

Clone this repository and create a virtual envirnoment. Then run the following command to install the Python package

```bash
pip3 install -e FakeNewsClassifierPackage
```

After succesfull installation of the python package run the run.py file using the following command
```bash
python3 FakeNewsClassifierPackage/run.py
```

The Python script will execute and display instructions in the terminal on how to access the webpage for training the algorithm and making predictions, along with the script's logs. To access the webpage, wait until the app is running, open a browser and click the following URL:

[Local Server](http://127.0.0.1:5000)

Some basic logs related to the process are displayed in the frontend while the Python scripts are running. Once you notice that the frontend indicates the successful processing of the uploaded files, you can navigate to the 'results' folder in the project and open the 'predictions.txt' file to view the output generated by the algorithm.

Additionally, in the 'model' folder, you will find the 'latest_trained_model.joblib' file. This is because the app retrains the model every time a .csv file is uploaded, even if it's the same file. This .joblib file enables you to load the model and make predictions using your own JSON file. Finally, the app supports the upload of either a single JSON object or a list of JSON objects.

## Running the tests

Nagivate to the FakeNewsClassifierPackage directory and run the following command:

```bash
pytest FakeNewsClassifierPackage
```

## Author

Vasilis Panagaris


