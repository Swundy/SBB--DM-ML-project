# SBB--DM-ML-project

## SBB Team: Alex Bystritzsky & Samuel Salgado.

## Detecting the difficulty level of French texts

The purpose of this project is to build a model that can predict the difficulty level of French texts for English speakers.  The goal is to provide texts with a mix of known and unknown words so that the user can improve their language skills while still being able to understand the text. The difficulty levels range from A1 to C2, the objective is to have the best prediction accuracy and we will try different models to see which one is the best.

## Data used

- training_data.csv - the training set. Composed of 4800 unique values.
- unlabelled_test_data.csv - the test set. Composed of 1200 unique values.

[Data used](https://www.kaggle.com/competitions/detecting-french-texts-difficulty-level-2022/data?select=unlabelled_test_data.csv)

## Methodology

First we used the followinf models without data cleaning :

- Logistic Regression.
- KeyNearestNeighbours (KNN).
- Random Forest Classifier.
- Decision Tree Classifier.

Then we tried to improve our accuracy with some data cleaning and more advanced techniques such as the "camembert" technique, which is a popular choice for natural language processing tasks. Camembert is trained on a large dataset of French language text and has been shown to perform well on a variety of tasks.

We used "TrainingArguments* which is a class in the transformers library that allows you to specify a number of arguments that control the training process of a transformer model.

We used "Preprocess_function" which is a function that takes in raw data and applies some preprocessing steps to it before it is fed into a machine learning model. This can include tasks such as tokenization, stemming, and removing stop words.

We also used "The trainer" which refers to an object that is responsible for training a machine learning model. It takes in a number of arguments, including the model to be trained, the training data, and any hyperparameters that need to be optimized. The trainer handles the details of training the model, such as splitting the data into batches, performing gradient descent, and updating the model weights.

## Summary of the model's performance

In the following table we'll have a sum up of our models including accuracy, recall and F1-score. 



Based on the results seen before, it appears that the logistic regression model is a simple and efficient model that is well-suited for binary classification tasks, such as predicting the difficulty of a French written text.

The KNN (k-nearest neighbors) model performed relatively poorly, with a precision of 0.32. This may be because KNN is a more complex model that requires a larger amount of data to train and may not perform as well on smaller datasets. Additionally, KNN can be sensitive to the choice of hyperparameters, such as the value of k, which may have affected its performance.

Both the random forest classifier and the decision tree classifier had similar performance, with precision scores of 0.42 and 0.3, respectively. These models are decision tree-based models that can be prone to overfitting, especially if the trees are deep and the data is not sufficiently diverse. This may have contributed to their relatively lower performance compared to the logistic regression model

Our SBB_model has a precision score of 0.54 when evaluated on a test set. This suggests that the model is able to accurately predict the difficulty of French written texts with a high degree of accuracy. It is worth noting that the specific techniques and algorithms used, as well as the hyperparameter optimization that was performed, likely contributed to the model's performance.

## Confusion Matrix


With example of good and bad predictions.

## Explicative video presentation of our model

## Conclusion

We buid a model that is pretty accurate, but how would we be able to improve our model ? Lack of data can be a major obstacle as it is necessary for training and evaluating models. Insufficient data can lead to underfitting, where a model does not accurately capture the underlying patterns in the data. To improve model performance, it is important to gather and utilize as much relevant data as possible, and maybe with a larger number of data we could have had a better prediction.

Source inconsistency can be a problem as it can lead to conflicting or contradictory data which can adversely affect model performance. Ensuring the consistency of sources can improve the reliability and credibility of the data being used, ultimately resulting in more accurate and effective models.

Overfitting is a common problem in machine learning and data analysis, where a model becomes too closely tailored to the specific data it was trained on. Preventing overfitting through cross-validation and limiting parameters improves model performance and reliability.
