# imports
import pandas as pd # data processing, I/O
import numpy as np
from sklearn import feature_extraction, linear_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump, load

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
sgd = linear_model.SGDClassifier(loss = "log_loss", max_iter = int(np.ceil((10**6)/len(testy))), penalty = 'l2', alpha = 0.0005) # SGD
perceptron = linear_model.Perceptron(max_iter = int(np.ceil((10**6)/len(testy))), alpha = 0.000001) # perceptron
mlp=MLPClassifier(hidden_layer_sizes= (3,3), activation = 'relu', solver = 'sgd', max_iter = 50) # multi layer perceptron

# # SGD
# sgd.fit(holdx, holdy)
# print("SGD score: "+str(sgd.score(testx, testy)))
# # TODO: GridSearchCV or RandomizedSearchCV to choose alpha
# # TODO: data should be scaled to [0,1] or [-1,1] or standardize to mean 0 variance 1 for sgd

# # Perceptron
# perceptron.fit(holdx, holdy)
# print("Perceptron score: "+ str(perceptron.score(testx, testy)))

# MLP
mlp.fit(holdx, holdy)
print("MLP score: " + str(mlp.score(testx, testy)))
dump(mlp, 'mlp.joblib')

# ### Final fitting using best model on submission data
# sgd.fit(df[df.columns.difference(["target"])], df["target"])
# results = sgd.predict(testdf)
# resultsdf = pd.DataFrame({'id': test["id"], 'target': results})

# ### Save final prediction as submission.csv
# resultsdf.to_csv('submission.csv', index = False)
