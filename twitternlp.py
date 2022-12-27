import numpy as np # linear algebra
import pandas as pd # data processing, I/O
from sklearn import feature_extraction, linear_model, model_selection, preprocessing # ML library

train = pd.read_csv("train.csv") # read training dataset
test = pd.read_csv("test.csv") # read test dataset

print(train)
print(test)