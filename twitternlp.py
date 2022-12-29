# imports
import pandas as pd # data processing, I/O
from sklearn import feature_extraction, linear_model, model_selection, preprocessing # ML library
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

# Data load
train = pd.read_csv("train.csv", index_col="id") # read training dataset
test = pd.read_csv("test.csv", index_col="id") # read test dataset

# Bag of words and data partition
count_vectorizer=feature_extraction.text.CountVectorizer() # from sklearn
trainvectors = count_vectorizer.fit_transform(train["text"]) # vectorise the entire training set
df = pd.DataFrame(trainvectors.toarray())
df["target"]=pd.Series.to_numpy(train["target"])
train_train, train_test = train_test_split(df, train_size=0.7, random_state=42) #split the dataset for holdout validation

### Data processing
# TODO: consider capitalization because this will count as different words (but then again all caps might be significant)
# TODO: consider stemming/lemmatization
# TODO: remove punctuation?

### Model training
lrdge=linear_model.RidgeClassifier() # Ridge Regression
logistic=LogisticRegression() # Logistic
rf=RandomForestClassifier() # Random Forest
mlp=MLPClassifier() # multi layer perceptron classifier
knn=KNeighborsClassifier() # K Nearest Neighbours
#naive = nltk.classify.NaiveBayesClassifier() # Naive Bayes
train_targets = pd.Series.to_numpy(train_test["target"]) # creates a numpy array for comparison to predictions

trainx = train_train[train_train.columns.difference(["target"])] # Training data minus target
train_testy = train_test[train_test.columns.difference(["target"])] # Test data minus target

# Ridge regression model fitting
lrdge.fit(trainx, train_train["target"]) # fits model on training data
lrdge_pred = lrdge.predict(train_testy) # Makes predictions based on fitted model

# This can be a function
lrdge_accuracy = 0
for x in lrdge_pred:
    if lrdge_pred[x] == train_targets[x]:
        lrdge_accuracy += 1
lrdge_accuracy = (lrdge_accuracy/len(lrdge_pred))*100
print("Ridge regression accuracy: "+str(lrdge_accuracy))

# TODO: Model prediction on test data with best model
# TODO: cross validation
# TODO: Save final prediction as submission.csv

