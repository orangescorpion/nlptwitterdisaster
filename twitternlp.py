# imports
import pandas as pd # data processing, I/O
from sklearn import feature_extraction, linear_model, model_selection, preprocessing # ML library
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

# Data load
train = pd.read_csv("train.csv", index_col="id") # read training dataset
test = pd.read_csv("test.csv") # read test dataset

### Data preprocessing
# TODO: consider capitalization because this will count as different words (but then again all caps might be significant)
# TODO: consider stemming/lemmatization
# TODO: remove punctuation?

# Bag of words and data partition
count_vectorizer=feature_extraction.text.CountVectorizer() # from sklearn
trainvectors = count_vectorizer.fit_transform(train["text"]) # vectorise the entire training set
df = pd.DataFrame(trainvectors.toarray())
df["target"]=pd.Series.to_numpy(train["target"]) # full dataset vectorised with target
# datasets for holdout
train_train, train_test = train_test_split(df, train_size=0.7, random_state=42) #split the dataset
trainx = train_train[train_train.columns.difference(["target"])] # Training data minus target
train_testy = train_test[train_test.columns.difference(["target"])] # Test data minus target
train_targets = pd.Series.to_numpy(train_test["target"]) # creates a numpy array for comparison to predictions
# datasets for CV
cv_x = df[df.columns.difference(["target"])]
cv_y = pd.Series.to_numpy(train["target"])
# vectorizing submission data
testvectors = count_vectorizer.transform(test["text"])
testdf = pd.DataFrame(testvectors.toarray())

### Model training holdout
logistic=linear_model.LogisticRegression() # Logistic
# Lasso regression (L1 regularization)
lrdge=linear_model.RidgeClassifier(alpha = 1) # Ridge Regression TODO: how to choose alpha
rf=RandomForestClassifier() # Random Forest
mlp=MLPClassifier() # multi layer perceptron classifier
knn=KNeighborsClassifier() # K Nearest Neighbours
#naive = nltk.classify.NaiveBayesClassifier() # Naive Bayes

# Function for calculating error
def accuracy(predicted, actual): # Function to return accuracy of predicted to actual, TODO: false positives / negatives
    accuracy = 0
    for x in predicted:
        if predicted[x] == actual[x]:
            accuracy += 1
    accuracy = (accuracy/len(predicted))*100
    return accuracy

# Ridge regression model fitting holdout
lrdge.fit(trainx, train_train["target"]) # fits model on training data
lrdge_pred = lrdge.predict(train_testy) # Makes predictions based on fitted model
print("Ridge regression accuracy: "+str(accuracy(lrdge_pred, train_targets))) # Print accuracy for model

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

### Unsupervised models
sgd = linear_model.SGDClassifier(loss = "log_loss")

#sgd
sgd.fit(cv_x, cv_y)
print("SGD score: "+str(sgd.score(cv_x, cv_y))) # 98.0

### Final fitting using best model on submission data
results = sgd.predict(testdf)
resultsdf = pd.DataFrame({'id': test["id"], 'target': results})

# TODO: Save final prediction as submission.csv
resultsdf.to_csv('submission.csv', index = False)
