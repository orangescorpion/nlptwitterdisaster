import numpy as np # linear algebra
import pandas as pd # data processing, I/O
from sklearn import feature_extraction, linear_model, model_selection, preprocessing # ML library
import nltk # natural language toolkit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

train = pd.read_csv("train.csv", index_col="id") # read training dataset
test = pd.read_csv("test.csv", index_col="id") # read test dataset

#train.info()
#test.info()

traintext = pd.DataFrame(train[["text", "target"]]) # ignoring keyword and location
traintext.info()

### Data processing

# TODO: consider capitalization because this will count as different words (but then again all caps might be significant)
# TODO: consider stemming/lemmatization

# TODO: n-grams, word2vec ?

# create bag of words
count_vectorizer=feature_extraction.text.CountVectorizer() # from sklearn

trainvectors = count_vectorizer.fit_transform(traintext["text"])
testvectors = count_vectorizer.fit_transform(test["text"])
print(trainvectors.shape)
print(testvectors.shape)


### Model training
lrdge=linear_model.RidgeClassifier() # Ridge Regression
logistic=LogisticRegression() # Logistic
rf=RandomForestClassifier() # Random Forest
mlp=MLPClassifier() # multi layer perceptron classifier
knn=KNeighborsClassifier() # K Nearest Neighbours

#naive = nltk.classify.NaiveBayesClassifier() # Naive Bayes

# Model fitting
#lrdge.fit(trainvectors, train["target"])

# Model assessment
#lrdge_pred = lrdge.predict(trainvectors) # Then take difference bewteen expected and actual

# Model prediction with best model

# Save final prediction as submission.csv

