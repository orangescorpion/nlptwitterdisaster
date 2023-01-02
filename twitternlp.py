# imports
import pandas as pd # data processing, I/O
import numpy as np
from sklearn import feature_extraction, linear_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump, load
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # stop tensorflow from printing warnings
import tensorflow as tf

gpus = tf.config.list_physical_devices()
for gpu in gpus:
    details = tf.config.experimental.get_device_details(gpu)
    print(details.get('device_name'))

# Function for calculating error
def accuracy(predicted, actual): # Function to return proportion of correct, TODO: false positives / negatives
    accuracy = 0
    for x in predicted:
        if predicted[x] == actual[x]:
            accuracy += 1
    accuracy = (accuracy/len(predicted))
    return accuracy

# Data load
train = pd.read_csv("train.csv", index_col="id") # read training dataset
test = pd.read_csv("test.csv") # read test dataset

### Data preprocessing
# TODO: consider stemming/lemmatization

# Bag of words and data partition
count_vectorizer=feature_extraction.text.CountVectorizer(strip_accents='unicode', stop_words='english', binary=True) # from sklearn
trainvectors = count_vectorizer.fit_transform(train["text"]) # vectorise the entire training set
df = pd.DataFrame(trainvectors.toarray())
df["target"]=pd.Series.to_numpy(train["target"]) # full dataset vectorised with target
# datasets for holdout
train_train, train_test = train_test_split(df, train_size=0.8, random_state=42) #split the dataset
holdx = train_train[train_train.columns.difference(["target"])] # Training data minus target
holdy = pd.Series.to_numpy(train_train["target"])
testx = train_test[train_test.columns.difference(["target"])] # Test data minus target
testy = pd.Series.to_numpy(train_test["target"])
# datasets for CV
cv_x = df[df.columns.difference(["target"])]
cv_y = pd.Series.to_numpy(train["target"])
# vectorizing submission data
testvectors = count_vectorizer.transform(test["text"])
testdf = pd.DataFrame(testvectors.toarray())
### Model training holdout
logistic=linear_model.LogisticRegression() # Logistic
# Lasso regression (L1 regularization)
lrdge=linear_model.RidgeClassifier() # Ridge Regression TODO: how to choose alpha
rf=RandomForestClassifier() # Random Forest
knn=KNeighborsClassifier(weights = 'distance', n_neighbors = 2) # K Nearest Neighbours
#naive = nltk.classify.NaiveBayesClassifier() # Naive Bayes

# Ridge
lrdge.fit(holdx, holdy) # fits model on training data
lrdge_pred = lrdge.predict(testx) # Makes predictions based on fitted model
print("Ridge score: "+str(accuracy(lrdge_pred, testy))) # Print accuracy for model

# K-neighbours
knn.fit(holdx, holdy)
knn_pred = knn.predict(testx)
print("KNN score: "+str(accuracy(knn_pred, testy)))

### Cross validation models
logistic_cv = linear_model.LogisticRegressionCV(max_iter = 100, solver = 'sag')
lasso_cv = linear_model.LogisticRegressionCV(max_iter = 200, solver = 'saga', penalty = "l1")
ridge_cv = linear_model.RidgeClassifierCV(alphas = (0.1, 0.5, 0.7, 1, 3, 5, 7, 10))
# Logstic
# TODO
# Lasso
#lasso_cv.fit(cv_x, cv_y)
#print("Lasso CV score: "+str(lasso_cv.score(cv_x, cv_y))) # TODO: does not converge
# Ridge
# ridge_cv.fit(cv_x, cv_y)
# print("Ridge CV score: "+str(ridge_cv.score(cv_x, cv_y))) # 94.8

### NN
nnmodel = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(21360,)),  # input shape required
  tf.keras.layers.Activation(activation=tf.nn.sigmoid)
])

nnmodel.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

with tf.device('gpu:0'):
  nnmodel.fit(holdx, holdy, epochs=50)
print("Test results:")
nnmodel.evaluate(testx, testy)

# ### Final fitting using best model on submission data
# sgd.fit(df[df.columns.difference(["target"])], df["target"])
# results = sgd.predict(testdf)
# resultsdf = pd.DataFrame({'id': test["id"], 'target': results})

# ### Save final prediction as submission.csv
# resultsdf.to_csv('submission.csv', index = False)
