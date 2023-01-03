# imports
import pandas as pd # data processing, I/O
import numpy as np
from sklearn import feature_extraction # vectorization
from sklearn.model_selection import train_test_split
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # stop tensorflow from printing warnings
import tensorflow as tf

# Check that tensorflow can access GPU
devices = tf.config.list_physical_devices()
for device in devices:
    details = tf.config.experimental.get_device_details(device)
    print(details.get('device_name'))
    print(device.name)
## TODO: tensorflow killing when VSCode opened through WSL terminal (using GPU), not locating GPU when opened through desktop.

# Data load
train = pd.read_csv("train.csv", index_col="id") # read training dataset
test = pd.read_csv("test.csv", index_col="id") # read test dataset

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

# datasets for final model training
full_x = df[df.columns.difference(["target"])]
full_y = pd.Series.to_numpy(train["target"])

# dataset for submission
testvectors = count_vectorizer.transform(test["text"])
testdf = pd.DataFrame(testvectors.toarray())

### NN
nnmodel = tf.keras.Sequential([
  tf.keras.layers.Dense(2000, activation=tf.nn.selu, input_shape=(21360,)),  # input shape required
  tf.keras.layers.Dropout(0.4, seed=42),
  tf.keras.layers.Dense(1, activation=tf.nn.selu),
  tf.keras.layers.Activation(activation=tf.nn.sigmoid)
])

nnmodel.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

## Fitting on training data
# with tf.device('gpu:0'):
#   nnmodel.fit(holdx, holdy, epochs=10)
# print("Test results:")
# nnmodel.evaluate(testx, testy)

## Final fitting using best model on submission data
nnmodel.fit(full_x, full_y, epochs=300)
nnmodel.save('fullmodel.tf') # Save model
model = tf.keras.models.load_model('fullmodel.tf')
results = model.predict(testdf) # Make final predictions

print(model.summary())
print(type(results))
print(results.shape)

resultsdf = pd.DataFrame({'id': test["id"], 'target': results}) # Create dataframe for submissio
resultsdf.to_csv('submission.csv', index = False) # Save submission to csv
