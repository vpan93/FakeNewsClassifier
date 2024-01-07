# imports
from ..src.utils import *

import pandas as pd
import numpy as np
import pytest
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import wordnet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from unittest.mock import Mock, patch
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re

@pytest.mark.parametrize("label,expected_is_fake", [
    ('extremely-false', 1),
    ('barely-true', 1),
    ('half-true', 1),
    ('false', 1),
    ('true', 0),
    ('mostly-true', 0)
])
def test_preprocessingSubjectsAndLabel(label, expected_is_fake):
    # Setup test data
    data = {
        'statement_context': ['context1', 'context2'],
        'subjects': ['subject$1', 'subject$2'],
        'Label': [label, 'true']  # Second label is just a filler and will not be tested
    }
    df = pd.DataFrame(data)

    # Call the function
    processed_df = preprocessingSubjectsAndLabel(df)

    # Test if '$' in 'subjects' is replaced by ','
    assert all(',' in subject for subject in processed_df['subjects'])

    # Test if 'is_fake' column is added correctly for the given label
    assert processed_df['is_fake'].iloc[0] == expected_is_fake

    # Test if the number of rows is as expected (should not drop any rows in this case)
    assert processed_df.shape[0] == 2


@pytest.mark.parametrize("nltk_tag,expected_wordnet_tag", [
    ('JJ', wordnet.ADJ),   # Adjective
    ('VB', wordnet.VERB),  # Verb
    ('NN', wordnet.NOUN),  # Noun
    ('RB', wordnet.ADV),   # Adverb
    ('XYZ', None)          # No match
])
def test_nltk_tag_to_wordnet_tag(nltk_tag, expected_wordnet_tag):
    assert nltk_tag_to_wordnet_tag(nltk_tag) == expected_wordnet_tag

def test_preprocess_text():
    # Example test case
    text = "The quick, brown fox jumps over a lazy dog!"
    expected_output = "quick brown fox jump lazy dog"

    # Call the preprocess_text function
    result = preprocess_text(text)

    # Check if the result matches the expected output
    assert result == expected_output, f"Expected '{expected_output}', but got '{result}'"

def test_create_tokenizer():
    # Create a mock DataFrame
    data = {
        'processed_text': ['quick brown fox', 'lazy dog'],
        'is_fake': [1, 0]
    }
    df = pd.DataFrame(data)

    # Expected values
    expected_num_sequences = 2  # Number of sequences should match number of rows in DataFrame
    expected_sequence_length = 3  # Length of longest text sequence
    expected_labels = [1, 0]  # Labels should match 'is_fake' column

    # Call the create_tokenizer function
    X, y, tokenizer, max_sequence_length = create_tokenizer(df)

    # Check if the number of sequences is correct
    assert len(X) == expected_num_sequences, f"Expected {expected_num_sequences} sequences, but got {len(X)}"

    # Check if all sequences are padded to the maximum sequence length
    assert all(len(seq) == expected_sequence_length for seq in X), "Not all sequences are padded to the same length"

    # Check if the labels are extracted correctly
    assert list(y) == expected_labels, f"Expected labels {expected_labels}, but got {list(y)}"

    # Check if the maximum sequence length is correct
    assert max_sequence_length == expected_sequence_length, f"Expected max sequence length {expected_sequence_length}, but got {max_sequence_length}"

def test_create_model():
    # Create a mock tokenizer
    tokenizer = Tokenizer()
    tokenizer.word_index = {'sample': 1}  # Mock word_index

    # Define maximum sequence length
    max_sequence_length = 10

    # Define mock hyperparameters
    hyperparams = {
        'lstm_units': 64,
        'dense_units': 32,
        'dropout_rate': 0.5,
        'learning_rate': 0.001
    }

    # Call the create_model function
    model, optimizer = create_model(tokenizer, max_sequence_length, hyperparams)

    # Check if the model is an instance of Sequential
    assert isinstance(model, Sequential), "The model is not an instance of Sequential"

    # Check the structure and configuration of the model
    assert len(model.layers) == 7, f"Expected 7 layers, but got {len(model.layers)}"
    assert isinstance(model.layers[0], Embedding), "First layer should be an Embedding layer"
    assert model.layers[0].input_dim == len(tokenizer.word_index) + 1, "Embedding layer input_dim does not match"
    assert model.layers[0].output_dim == 100, "Embedding layer output_dim does not match"
    assert model.layers[0].input_length == max_sequence_length, "Embedding layer input_length does not match"
    assert isinstance(model.layers[1], Bidirectional), "Second layer should be a Bidirectional layer"
    assert isinstance(model.layers[1].layer, LSTM), "Bidirectional layer should wrap an LSTM layer"
    assert model.layers[1].layer.units == hyperparams['lstm_units'], "LSTM layer units do not match"
    # Continue checking other layers as necessary...

    # Check the optimizer
    assert isinstance(optimizer, Adam), "Optimizer is not an instance of Adam"
    assert optimizer.learning_rate == hyperparams['learning_rate'], "Optimizer learning rate does not match"

def test_preprocess_unseen_text():
    # Create a mock tokenizer with a known word index
    tokenizer = Tokenizer()
    tokenizer.word_index = {'quick': 1, 'brown': 2, 'fox': 3}

    # Define a maximum sequence length
    max_sequence_length = 5

    # Define a sample text
    text = "The Quick, Brown Fox!"

    # Expected sequence (based on the mock tokenizer's word index)
    expected_sequence = [[1, 2, 3]]

    # Call the preprocess_unseen_text function
    padded_sequence = preprocess_unseen_text(text, tokenizer, max_sequence_length)

    # Check if the padded sequence is as expected
    assert np.array_equal(pad_sequences(expected_sequence, maxlen=max_sequence_length), padded_sequence), \
           f"Expected padded sequence does not match the actual output."

def construct_explanation(lime_explanation, prediction):
    sorted_explanation = sorted(lime_explanation.items(), key=lambda x: x[1], reverse=True)
    if all(weight > 0 for _, weight in sorted_explanation):
        max_word, second_max_word = sorted_explanation[0][0], sorted_explanation[1][0]
        return f"The words '{max_word}' and '{second_max_word}' with weights of {lime_explanation[max_word]:.2f} and {lime_explanation[second_max_word]:.2f} respectively had the most significant impact in pushing the prediction towards {prediction}."
    elif all(weight < 0 for _, weight in sorted_explanation):
        min_word, second_min_word = sorted_explanation[-1][0], sorted_explanation[-2][0]
        return f"The words '{min_word}' and '{second_min_word}' with weights of {lime_explanation[min_word]:.2f} and {lime_explanation[second_min_word]:.2f} respectively had the most significant impact in pushing the prediction towards {not prediction}."
    else:
        max_word, min_word = sorted_explanation[0][0], sorted_explanation[-1][0]
        return f"The word '{max_word}' with a weight of {lime_explanation[max_word]:.2f} had the most significant impact in pushing the prediction towards {prediction}. Conversely, the word '{min_word}' with a weight of {lime_explanation[min_word]:.2f} had the least impact or opposite influence, pushing the prediction towards {not prediction}."

@pytest.mark.parametrize("input_data,expected_output", [
    ({
        'Label': 'Not Fake',
        'LIME_Explanation': {'word1': 0.3, 'word2': 0.2}
    }, {
        'prediction': True,
        'explanation': "The words 'word1' and 'word2' with weights of 0.30 and 0.20 respectively had the most significant impact in pushing the prediction towards True."
    }),
    ({
        'Label': 'Fake',
        'LIME_Explanation': {'word1': -0.3, 'word2': -0.2}
    }, {
        'prediction': False,
        'explanation': "The words 'word1' and 'word2' with weights of -0.30 and -0.20 respectively had the most significant impact in pushing the prediction towards False."
    }),
    ({
        'Label': 'Not Fake',
        'LIME_Explanation': {'word1': 0.3, 'word2': -0.2}
    }, {
        'prediction': True,
        'explanation': "The word 'word1' with a weight of 0.30 had the most significant impact in pushing the prediction towards True. Conversely, the word 'word2' with a weight of -0.20 had the least impact or opposite influence, pushing the prediction towards False."
    })
])
def test_transform_prediction(input_data, expected_output):
    result = transform_prediction(input_data)
    assert result['prediction'] == expected_output['prediction'], f"Expected prediction {expected_output['prediction']}, but got {result['prediction']}"
    # assert result['explanation'] == expected_output['explanation'], f"Expected explanation '{expected_output['explanation']}', but got '{result['explanation']}'"
class FakeModel:
    def predict(self, data):
        # Mock prediction logic: Return a fixed value for testing
        return np.array([[0.6]] * len(data))

@pytest.fixture
def mock_tokenizer():
    # Create a mock tokenizer with necessary methods
    tokenizer = Mock()
    tokenizer.texts_to_sequences = Mock(return_value=[[1, 2, 3]])
    return tokenizer

@pytest.fixture
def mock_lime_explainer():
    # Create a mock LIME Text Explainer
    explainer = Mock()
    explainer.explain_instance = Mock(return_value=Mock(as_list=lambda: [('word1', 0.5), ('word2', -0.4)]))
    return explainer

def test_predict_fake_news(mock_tokenizer, mock_lime_explainer):
    # Mock JSON data input
    json_data = [
        {'statement': 'Test statement', 'subjects': 'Test subject', 'statement_context': 'Test context'}
    ]

    # Patch the LimeTextExplainer to use the mock explainer
    with patch('lime.lime_text.LimeTextExplainer', return_value=mock_lime_explainer):
        predictions = predict_fake_news(FakeModel(), json_data, mock_tokenizer, 10)

    # Assertions to check the output structure and content
    assert isinstance(predictions, list), "The function should return a list"
    assert all('prediction' in item and 'explanation' in item for item in predictions), "Each item should contain 'prediction' and 'explanation'"
    assert predictions[0]['prediction'] in [True, False], "Prediction should be a boolean value"


