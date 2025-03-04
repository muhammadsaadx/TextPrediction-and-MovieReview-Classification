# Text Prediction and Movie Review Classification

This repository contains an implementation of text prediction and sentiment classification on movie reviews. The project utilizes deep learning models to predict words in a given text sequence and classify movie reviews as positive or negative.

## Features
- Implements text prediction using NLP techniques.
- Performs movie review classification using deep learning models.
- Utilizes PyTorch and TensorFlow/Keras for model development.
- Processes text using tokenization and embeddings.
- Evaluates model performance using accuracy and loss metrics.

## Dataset
The project works with a text dataset containing:
- A corpus for text prediction tasks.
- Movie reviews labeled as positive or negative for sentiment classification.
- Preprocessing includes tokenization, sequence padding, and vocabulary creation.

## Model Architectures
### Text Prediction Model
- **Tokenization & Embedding**: Converts text into numerical representations using word embeddings.
- **Recurrent Neural Network (RNN) / LSTM**: Predicts the next word in a sequence based on the previous words.
- **Softmax Layer**: Outputs probabilities for the next word.

### Movie Review Classification Model
- **Embedding Layer**: Transforms input words into dense vector representations.
- **LSTM/GRU or Transformer-based Encoder**: Captures sequential dependencies in text.
- **Fully Connected Layer**: Processes encoded features for classification.
- **Sigmoid Activation**: Outputs a probability score indicating sentiment.

## Training Process
1. **Data Preprocessing**:
   - Tokenization of text into sequences.
   - Padding sequences for uniform input size.
2. **Model Training**:
   - Cross-entropy loss for classification.
   - Adam optimizer with learning rate scheduling.
   - Training on GPU for faster computation.
3. **Validation & Evaluation**:
   - Accuracy and loss metrics for performance assessment.
   - Sample text predictions and sentiment classification outputs.

## Evaluation
- **Accuracy**: Measures correct predictions in classification tasks.
- **Loss Curves**: Shows training and validation loss trends.
- **Sample Predictions**: Demonstrates qualitative performance on test data.

## Usage
- The text prediction model generates the next word in a given text input.
- The movie review classifier determines the sentiment (positive/negative) of an input review.
