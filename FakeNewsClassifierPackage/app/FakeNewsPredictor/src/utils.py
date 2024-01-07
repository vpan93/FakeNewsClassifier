import pandas as pd
import numpy as np
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, Dropout, Bidirectional
from keras.optimizers import Adam
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet
import re, string,json
from keras.models import model_from_json
from lime.lime_text import LimeTextExplainer

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

def preprocessingSubjectsAndLabel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses a pandas DataFrame by cleaning and transforming the data.

    Parameters:
    df (pd.DataFrame): The DataFrame to be processed. It should contain the columns 
                       'statement_context', 'subjects', and 'Label'.

    Returns:
    pd.DataFrame: The processed DataFrame with NaNs removed, the 'subjects' column 
                  modified, and a new 'is_fake' column added.
    """
    # Remove rows with missing 'statement_context'
    df = df.dropna(subset=['statement_context'])

    # Replace '$' with ',' in 'subjects'
    df['subjects'] = df['subjects'].str.replace('\$', ',', regex=True)

    # Add 'is_fake' column, marking certain labels as fake
    df['is_fake'] = np.where(df['Label'].isin(['extremely-false', 'barely-true', 'half-true', 'false']), 1, 0)

    return df


def nltk_tag_to_wordnet_tag(nltk_tag):
    """
    Converts a part-of-speech tag from the NLTK format to the WordNet format.

    Parameters:
    nltk_tag (str): A part-of-speech tag in NLTK's format.

    Returns:
    A corresponding WordNet part-of-speech tag, or None if there is no equivalent.
    """
    # Check if the NLTK tag is for an adjective
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    # Check if the NLTK tag is for a verb
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    # Check if the NLTK tag is for a noun
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    # Check if the NLTK tag is for an adverb
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    # Return None if no match is found
    else:          
        return None


def preprocess_text(text):
    """
    Preprocesses the given text by applying lowercasing, removing non-alphabetic characters,
    tokenizing, removing stopwords, performing part-of-speech tagging, and lemmatizing.

    Parameters:
    text (str): The text to be preprocessed.

    Returns:
    str: The preprocessed and lemmatized text.
    """
    # Convert text to lowercase and remove non-alphabetic characters
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)

    # Tokenize the text into individual words
    tokens = word_tokenize(text)

    # Remove stopwords to reduce noise in the text
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Tag tokens with their part-of-speech using NLTK
    nltk_tagged = nltk.pos_tag(tokens)  
    
    # Convert NLTK part-of-speech tags to WordNet's format
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)

    # Initialize the WordNet lemmatizer
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentences = []

    # Lemmatize each word with its corresponding part-of-speech tag
    for word, tag in wordnet_tagged:
        if tag is None:
            lemmatized_sentences.append(word)
        else:        
            lemmatized_sentences.append(lemmatizer.lemmatize(word, tag))
    
    # Join the lemmatized words back into a single string
    return ' '.join(lemmatized_sentences)

def create_tokenizer(df: pd.DataFrame):
    """
    Creates a tokenizer based on the 'processed_text' column of the DataFrame, 
    converts the texts to sequences, and pads them to the same length.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the 'processed_text' and 'is_fake' columns.

    Returns:
    tuple: A tuple containing the padded sequences (X), labels (y), the tokenizer object, 
           and the maximum sequence length.
    """
    # Initialize the tokenizer
    tokenizer = Tokenizer()
    
    # Fit the tokenizer on the 'processed_text' column of the DataFrame
    tokenizer.fit_on_texts(df['processed_text'])

    # Convert the texts in 'processed_text' to sequences of integers
    sequences = tokenizer.texts_to_sequences(df['processed_text'])

    # Determine the maximum length of any sequence
    max_sequence_length = max(len(x) for x in sequences)

    # Pad the sequences so that they all have the same length
    X = pad_sequences(sequences, maxlen=max_sequence_length)

    # Extract the labels from the DataFrame
    y = df['is_fake'].values

    # Return the padded sequences, labels, tokenizer, and maximum sequence length
    return X, y, tokenizer, max_sequence_length


def create_model(tokenizer, max_sequence_length, hyperparams):
    """
    Creates a Sequential model for text classification with LSTM layers and hyperparameters.

    Parameters:
    tokenizer: Tokenizer object used for text preprocessing.
    max_sequence_length (int): The maximum length of the input sequences.
    hyperparams (dict): A dictionary containing hyperparameters for the model.

    Returns:
    tuple: A tuple containing the created model and the optimizer.
    """
    # Initialize a Sequential model
    model = Sequential()

    # Add an Embedding layer
    model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, 
                        output_dim=100, 
                        input_length=max_sequence_length))

    # Add a Bidirectional LSTM layer
    model.add(Bidirectional(LSTM(hyperparams['lstm_units'], return_sequences=False)))

    # Add a Dense layer with ReLU activation
    model.add(Dense(hyperparams['dense_units'], activation='relu'))

    # Add a Dropout layer
    model.add(Dropout(hyperparams['dropout_rate']))

    # Add another Dense layer 
    model.add(Dense(32, activation='relu'))

    # Add another Dropout layer with the same rate for consistency
    model.add(Dropout(hyperparams['dropout_rate']))

    # Add the output Dense layer with sigmoid activation for binary classification
    model.add(Dense(1, activation='sigmoid'))

    # Define the optimizer with a learning rate from hyperparameters
    optimizer = Adam(learning_rate=hyperparams['learning_rate'])

    # Return the constructed model and optimizer
    return model, optimizer


def preprocess_unseen_text(text, tokenizer, max_sequence_length):
    """
    Preprocesses a given text (not seen during training) for prediction, 
    using a specified tokenizer and max sequence length.

    Parameters:
    text (str): The text to be preprocessed.
    tokenizer: The tokenizer used during the training of the model.
    max_sequence_length (int): The maximum length of sequences used in the model.

    Returns:
    numpy array: The padded sequence of the processed text.
    """
    # Convert text to lowercase and remove non-alphabetic characters
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Initialize PorterStemmer and WordNetLemmatizer
    porter = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    # Stem and lemmatize the tokens
    processed = [lemmatizer.lemmatize(porter.stem(word)) for word in tokens]

    # Convert processed text to a sequence using the tokenizer
    sequence = tokenizer.texts_to_sequences([' '.join(processed)])

    # Pad the sequence to the maximum sequence length
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length)

    # Return the padded sequence for model input
    return padded_sequence


def transform_prediction(data):
    """
    Transforms the raw prediction data into a more interpretable format, including the prediction
    and an explanation based on the LIME (Local Interpretable Model-agnostic Explanations) weights.

    Parameters:
    data (dict): A dictionary containing the prediction label and LIME explanation. 
                 The 'Label' key should have the value 'Not Fake' or other, and 
                 'LIME_Explanation' key should contain a dictionary of words with their corresponding weights.

    Returns:
    dict: A dictionary with the boolean prediction and a textual explanation.
    """
    # Convert 'Label' to a boolean 'prediction'
    prediction = True if data['Label'] == 'Not Fake' else False

    # Extract the LIME explanation
    lime_explanation = data['LIME_Explanation']

    # Sort the words based on their weights in the LIME explanation
    sorted_explanation = sorted(lime_explanation.items(), key=lambda x: x[1])

    # Determine the direction and significance of the words in the explanation
    if all(weight > 0 for _, weight in sorted_explanation):
        # Case when all weights are positive
        max_word, second_max_word = sorted_explanation[-1][0], sorted_explanation[-2][0]
        explanation = f"The words '{max_word}' and '{second_max_word}' with weights of {lime_explanation[max_word]:.2f} and {lime_explanation[second_max_word]:.2f} respectively had the most significant impact in pushing the prediction towards '{prediction}'."
    elif all(weight < 0 for _, weight in sorted_explanation):
        # Case when all weights are negative
        min_word, second_min_word = sorted_explanation[0][0], sorted_explanation[1][0]
        explanation = f"The words '{min_word}' and '{second_min_word}' with weights of {lime_explanation[min_word]:.2f} and {lime_explanation[second_min_word]:.2f} respectively had the most significant impact in pushing the prediction towards '{not prediction}'."
    else:
        # Case when weights are mixed
        max_word, min_word = sorted_explanation[-1][0], sorted_explanation[0][0]
        explanation_direction_max = prediction if lime_explanation[max_word] > 0 else not prediction
        explanation_direction_min = not prediction if lime_explanation[min_word] < 0 else prediction
        explanation = f"The word '{max_word}' with a weight of {lime_explanation[max_word]:.2f} had the most significant impact in pushing the prediction towards '{explanation_direction_max}'. Conversely, the word '{min_word}' with a weight of {lime_explanation[min_word]:.2f} had the least impact or opposite influence, pushing the prediction towards '{explanation_direction_min}'."

    # Prepare the JSON response with the prediction and explanation
    json_to_return = {
        "prediction": prediction,
        "explanation": explanation
    }

    # Return the transformed prediction data
    return json_to_return


def predict_fake_news(model, json_data, tokenizer, max_sequence_length):
    """
    Predicts whether news items in the given JSON data are fake or not using the specified model, 
    tokenizer, and max sequence length. It also provides explanations for the predictions using LIME.

    Parameters:
    model: The trained machine learning model used for predictions.
    json_data (list of dicts): The data to be predicted, in JSON format.
    tokenizer: The tokenizer used for text preprocessing.
    max_sequence_length (int): The maximum length of sequences used in the model.

    Returns:
    list of dicts: A list of dictionaries, each containing the prediction and its explanation.
    """
    # Initialize a LimeTextExplainer
    explainer = LimeTextExplainer(class_names=['Not Fake', 'Fake'])

    def model_predict_proba(texts):
        # Preprocess and predict probabilities for each text
        processed_sequences = [preprocess_unseen_text(text, tokenizer, max_sequence_length) for text in texts]
        predictions = model.predict(np.vstack(processed_sequences))
        # Return probabilities in a format suitable for LIME
        return np.hstack((1-predictions, predictions))

    json_to_return = []

    for item in json_data:
        # Combine text elements from JSON item
        combined_text = f"{item['statement']} {item['subjects']} {item['statement_context']}"
        processed_sequence = preprocess_unseen_text(combined_text, tokenizer, max_sequence_length)
        prediction = model.predict(processed_sequence)
        # Determine the label based on the prediction
        prediction_label = "Not Fake" if prediction[0][0] < 0.5 else "Fake"
        item['Label'] = prediction_label

        # Generate an explanation using LIME
        exp = explainer.explain_instance(combined_text, model_predict_proba, num_features=10)
        explanation = exp.as_list()
        # Convert the explanation to a dictionary
        item['LIME_Explanation'] = {feature: weight for feature, weight in explanation}

        # Transform the prediction into a more interpretable format
        transformed_prediction = transform_prediction(item)
        json_to_return.append(transformed_prediction)

    # Return the list of predictions with explanations
    return json_to_return
