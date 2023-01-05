# imports
import pandas as pd # data processing, I/O
import numpy as np # array handling
from sklearn import feature_extraction # vectorization
from sklearn.model_selection import train_test_split, GridSearchCV # data split for CV
from scikeras.wrappers import KerasClassifier
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # tensorflow verbosity, don't print warnings
import tensorflow as tf
tf.random.set_seed(42) # random seed for replication

# Check that tensorflow can access GPU
devices = tf.config.list_physical_devices()
for device in devices:
    details = tf.config.experimental.get_device_details(device)
    print(details.get('device_name'))
    print(device.name)
# TODO: tensorflow killing when VSCode opened through WSL terminal (using GPU), not locating GPU when opened through desktop.

# Data load
train = pd.read_csv("train.csv", index_col="id") # read training dataset
test = pd.read_csv("test.csv") # read test dataset

### Data preprocessing
# Bag of words and training dataset
count_vectorizer=feature_extraction.text.CountVectorizer(strip_accents='unicode', stop_words='english', binary=True) # from sklearn
trainvectors = count_vectorizer.fit_transform(train["text"]) # vectorise the entire training set
df = pd.DataFrame(trainvectors.toarray())
df["target"]=pd.Series.to_numpy(train["target"]) # full dataset vectorised with target
# TODO: consider stemming/lemmatization

# datasets for holdout CV
train_train, train_test = train_test_split(df, train_size=0.7, random_state=42) #split the dataset
holdx = train_train[train_train.columns.difference(["target"])] # Text data for training
holdy = pd.Series.to_numpy(train_train["target"]) # predictor for training
testx = train_test[train_test.columns.difference(["target"])] # Text data for test
testy = pd.Series.to_numpy(train_test["target"]) # predictor for test
full_x = df[df.columns.difference(["target"])] # Text data for final model fitting
full_y = pd.Series.to_numpy(train["target"]) # Predictor for final model fitting

# submission text dataset
testvectors = count_vectorizer.transform(test["text"])
testdf = pd.DataFrame(testvectors.toarray())

## NN model design
def model_create(neurons=1):
  nnmodel = tf.keras.Sequential([
    tf.keras.layers.Dense(neurons, activation=tf.nn.selu, input_shape=(21360,)),  # input shape required
    tf.keras.layers.Dropout(0.5, seed=42),
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
  ])
  # compile model
  nnmodel.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
  return nnmodel

# Hyperparameter tuning
model = KerasClassifier(build_fn=model_create, epochs=100, batch_size=10, neurons=[5,10])
param_grid = dict(neurons=[5, 10])
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_result = grid.fit(full_x, full_y)

best_params=gs.best_params_
accuracy=gs.best_score_

print("Best params: "+best_params)
print("Best score: "+accuracy)

es = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=5) # callback to stop training early

# Fitting on training data with cross validation
# with tf.device('gpu:0'): # specify to use GPU rather than CPU
#   nnmodel.fit(holdx, holdy, epochs=100, callbacks = [es])
# print("Test results:")
# nnmodel.evaluate(testx, testy)

## Final fitting using best model on submission data
# nnmodel.fit(full_x, full_y, epochs=200, callbacks=[es])
# nnmodel.save('fullmodel.tf') # Save model

## Load model and make final predictions
# model = tf.keras.models.load_model('fullmodel.tf') # load saved model
# results = model.predict(testdf) # Make final predictions

# ## Some wrangling to get data types compatible with dataframes
# id_csv = (test["id"]).values
# results = results.tolist()
# floats = [item for sublist in results for item in sublist]

# # Conversion to binary (for relu the target is anything > 0, for sigmoid it is >= 0.5)
# target_csv = []
# for x in floats:
#   if x >= 0.5:
#     target_csv.append(1)
#   else:
#     target_csv.append(0)

# resultsdf = pd.DataFrame({'id': id_csv, 'target': target_csv}) # Create dataframe for submission
# resultsdf.to_csv('submission.csv', index = False) # Save submission as csv
