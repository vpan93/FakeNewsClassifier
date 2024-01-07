# imports 
import pandas as pd
from .utils import *
import logging
import keras_tuner
from kerastuner import Hyperband

# Basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocessing(df: pd.DataFrame):
    """
    Executes the preprocessing pipeline on the provided DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing the raw data with columns 'statement', 
                       'subjects', and 'statement_context' among others.

    Returns:
    Tuple containing split data and tokenization artifacts:
    - X_train (pd.DataFrame): Training features.
    - X_val (pd.DataFrame): Validation features.
    - X_test (pd.DataFrame): Test features.
    - y_train (np.ndarray): Training labels.
    - y_val (np.ndarray): Validation labels.
    - y_test (np.ndarray): Test labels.
    - tokenizer (Tokenizer): Fitted Keras Tokenizer on the processed text.
    - max_sequence_length (int): The length of the longest sequence after tokenization.
    """
    # Log the start of preprocessing
    logging.info("Starting preprocessing of data")
    
    # Apply predefined preprocessing to subjects and labels
    df = preprocessingSubjectsAndLabel(df)
    
    # Create a new column by combining 'statement', 'subjects', and 'statement_context' after applying a text preprocessing function
    df['processed_text'] = df.apply(lambda row: preprocess_text(f"{row['statement']} {row['subjects']} {row['statement_context']}"), axis=1)
    
    # Tokenize the 'processed_text' and get the tokenized sequences, labels, tokenizer object, and the max sequence length for padding
    X, y, tokenizer, max_sequence_length = create_tokenizer(df)
    
    # Split the data into training and temporary sets, allocating 80% to training and 20% to a temporary holdout set
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Split the temporary set equally into validation and test sets
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Return the dataset splits, tokenizer, and max sequence length to be used for padding during training
    return X_train, X_val, X_test, y_train, y_val, y_test, tokenizer, max_sequence_length


def tune_hyperparameters(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, tokenizer: Tokenizer, max_sequence_length: int):
    """
    Conducts hyperparameter tuning using the Hyperband optimization algorithm.

    Parameters:
    X_train (np.ndarray): The feature data for training.
    y_train (np.ndarray): The target labels for training.
    X_val (np.ndarray): The feature data for validation.
    y_val (np.ndarray): The target labels for validation.
    tokenizer (Tokenizer): The tokenizer used to process text data for model input.
    max_sequence_length (int): The maximum length of sequences for model input.

    Returns:
    HyperParameters: The best hyperparameters found by the Hyperband tuner.
    """
    # Define a function to build a model given hyperparameters, used by the tuner
    def build_model(hp):
        # Specify the hyperparameters to be tuned with ranges and default values
        hyperparams = {
            'lstm_units': hp.Int('lstm_units', min_value=32, max_value=512, step=32),
            'dense_units': hp.Int('dense_units', min_value=32, max_value=512, step=32),
            'dropout_rate': hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1),
            'learning_rate': hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG'),
            'batch_size': hp.Choice('batch_size', values=[32, 64, 128]),
            'epochs': 5  # This is set to a fixed value and not tuned
        }

        # Create the model with the selected hyperparameters
        model, _ = create_model(tokenizer, max_sequence_length, hyperparams)
        
        # Compile the model with the Adam optimizer and binary cross-entropy loss, tracking accuracy
        model.compile(optimizer=Adam(learning_rate=hyperparams['learning_rate']), loss='binary_crossentropy', metrics=['accuracy'])

        return model

    # Instantiate the Hyperband tuner which will use the build_model function to tune the hyperparameters
    tuner = Hyperband(
        build_model,
        objective='val_accuracy',
        max_epochs=3,
        directory='hyperparam_tuning',
        project_name='FakeNewsPredictor'
    )

    # Execute the hyperparameter search using the training and validation data
    tuner.search(X_train, y_train, epochs=5, validation_data=(X_val, y_val))

    # Retrieve and return the best hyperparameters after the search concludes
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    return best_hps

def train_algorithm(tokenizer: Tokenizer, max_sequence_length: int, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, hyperparams: dict):
    """
    Trains a machine learning model using the provided dataset and hyperparameters.

    Parameters:
    tokenizer (Tokenizer): The tokenizer used to process text data for model input.
    max_sequence_length (int): The maximum length of sequences for model input.
    X_train (np.ndarray): The feature data for training.
    y_train (np.ndarray): The target labels for training.
    X_val (np.ndarray): The feature data for validation.
    y_val (np.ndarray): The target labels for validation.
    hyperparams (dict): A dictionary of hyperparameters for model training.

    Returns:
    Tuple[Model, Optimizer]: The trained model and its optimizer.
    """
    # Log the start of model training, noting the hyperparameters being used
    logging.info("Starting model training with hyperparameters: %s", str(hyperparams))

    # Create the model architecture based on the tokenizer and max sequence length, and initialize with hyperparameters
    model, optimizer = create_model(tokenizer, max_sequence_length, hyperparams)

    # Compile the model specifying the optimizer, loss function, and metrics to monitor
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # Iteratively train the model for a set number of epochs
    for epoch in range(1, 6):
        # Log the epoch number at the start of each epoch
        logging.info(f"Starting Epoch {epoch}/5")

        # Train the model on the training data, evaluate against the validation set, and use the hyperparameter-specified batch size
        model.fit(X_train, y_train, epochs=1, validation_data=(X_val, y_val), batch_size=hyperparams['batch_size'])

        # Log the end of the epoch
        logging.info(f"Finished Epoch {epoch}/5")
    
    # Log that the model training has been completed
    logging.info("Completed model training")

    # Return the trained model and the optimizer used
    return model, optimizer

def predict(json_data,model, tokenizer, max_sequence_length):
    """
    Generates predictions for the provided json data using the trained model.

    Parameters:
    json_data (str): JSON formatted string containing the data to be predicted.
    model (Model): The trained model to use for predictions.
    tokenizer (Tokenizer): The tokenizer used to process text data for model input.
    max_sequence_length (int): The maximum length of sequences that the model can handle.

    Returns:
    List[str]: A list containing the prediction labels for the input data.
    """
    logging.info("Starting prediction")
    # Make a prediction
    predictions_with_labels = predict_fake_news(model, json_data, tokenizer, max_sequence_length)
    return predictions_with_labels

