
# SMS Spam Detection with Machine Learning

This project aims to detect spam messages in SMS (Short Message Service) using machine learning techniques. We implement a classification model to differentiate between spam and non-spam (ham) messages.

## Introduction

Spam messages are unsolicited and unwanted messages sent in bulk, often for advertising or fraudulent purposes. Detecting spam messages is crucial for maintaining user privacy and security. In this project, we leverage machine learning algorithms to automatically classify SMS messages as spam or ham.

## Data

The dataset used in this project is the [SMS Spam Collection Dataset](https://www.kaggle.com/uciml/sms-spam-collection-dataset) from Kaggle. It contains a collection of SMS messages labeled as spam or ham.

## Approach

1. **Data Preprocessing**: We preprocess the SMS messages by cleaning the text, removing punctuation, and converting text to lowercase.

2. **Feature Engineering**: We extract features from the text using the Bag of Words approach, which represents each message as a vector of word frequencies.

3. **Model Training**: We train a Naive Bayes classifier using the Multinomial Naive Bayes algorithm.

4. **Model Evaluation**: We evaluate the performance of the classifier using metrics such as accuracy, precision, recall, and F1-score.

## Usage

1. **Dataset**: Download the `spam.csv` dataset from [Kaggle](https://www.kaggle.com/uciml/sms-spam-collection-dataset) and place it in the project directory.

2. **Running the Code**: Execute the provided Python script `sms_spam_detection.py` in your Python environment.

## Repository Structure

```
.
├── spam.csv                     # SMS Spam Collection Dataset
├── sms_spam_detection.py        # Python script for SMS spam detection
└── README.md                    # This README file
```

## Dependencies

- numpy
- pandas
- nltk
- matplotlib
- seaborn
- scikit-learn

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
