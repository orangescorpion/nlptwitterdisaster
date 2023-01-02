# nlptwitterdisaster
Natural Language Processing with Disaster Tweets (https://www.kaggle.com/competitions/nlp-getting-started) is an ongoing Kaggle competition for newcomers to Kaggle and ML beginners. This project contains my solution.

# Modules used
## pandas
The development of pandas began in 2008 and is now an open-source project aimed at being the most powerful data analysis/manipulation tool. It has its own DataFrame object and is capable of reading and writing files as well as pivoting, merging, indexing, and subsetting datasets.

## scikit-learn
Scikit-learn (aka sklearn) is an open source machine learning module for Python, built on NumPy, SciPy, and matplotlib. Its functions include data preprocessing, dimensionality reduction, model selection, clustering, regression, and classification. This includes support vector machines and neural networks.

## nltk
Natural Language Toolkit is a platform for natural language processing with libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning.

# Data processing
## Dealing with capitalization
Since ML models will treat capital letters as distinct characters from lowercase letters, the words "Book" and "book" would be counted as two separate words in model training. Although capitalisation can sometimes provide clues about sentiment (e.g. ALL CAPS OFTEN SIGNIFIES STRONG EMOTION), in this case all characters will be converted to lowercase for the benefit of matching words.

## Punctuation
When separating a sentence into separate words to create word count vectors, spaces are used to distinguish separate words. So a sentence like "sadly, I won't be here" would turn into the array ["sadly,", "I", "won't", "be", "here"]. Evidently, "sadly," includes the comma which would make it a different word from "sadly". For this reason, punctuation is removed as part of the data cleaning stage. Important to note that if all punctuation is removed, words like "o'clock" will become "oclock".

## Stemming/Lemmatization
Stemming is the process of reducing words to their roots/stems. For example, "leader", "leading", "leads" would all become "lead". Words are not always reduced to a dictionary word, however, "troubling", "trouble" and "troubled" all become "troubl". For machine learning purposes, stemming can reduce redundancy, thereby producing more robust models. There are several popular algorithms for stemming.

Lemmatization is the process of changing words to their lemma form (the base dictionary meaning of the word). For example, "is", "are", and "am" would all become "be". Lemmatization can be more time consuming than stemming and requires a dictionary with a morphological understanding of words.

# Natural Language Processing techniques
## Ordered text: Markov models and n-grams
For NLP that is used to generate text or validate grammar, the order of words is important. n-grams are ordered sequences of n items from a sequence. They are used to determine words or letters that are commonly found together.

## Unordered text: bag of words
For classification and prediction, the order of words in a sentence is less necessary than for other AI applications such as translation or chatbots. By representing text as an unordered collection of words, text can be represented as numerical vectors which are used to train models. An emphasis can be placed on term frequency rather than sequences.

# Models Used
## Ridge Regression Classification
Ridge classification is used to analyse linear discriminant models. It is a form of regularisation that prevents overfitting by penalizing model coefficients (adding a penalty term to the cost function that discourages complexity). The penalty term is typically the sum of squared coefficients of the features in the model. By increasing the penalty term, regularization is increased resulting in smaller coefficient values. Can be beneficial when training data is small in size.
### Assumptions
Assumptions are similar to linear regression: linearity, constant variance, and independence. However, normal distribution of errors is not necessary as ridge does not provide CIs.
### Loss function
Mean square loss with L2 regularization as penalty term
### Prediction
Outputs are between -1 and 1 and classification is based on greater than or less than 0.
### Multiclass prediction
In multiclass prediction, each class is treated as a separate binary outcome.
### Parameters (scikit-learn)
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html 
#### alpha
Alpha is the regularization strength, must be a positive float, larger values correspond to stronger regularization
#### solver
Solver to use in the computational routines
 - auto (default): chooses automatically based on data type
 - svd: Singular Value Decomposition of X, most stable solver particularly moreso than 'cholesky' for singular matrices but slower
 - cholesky: uses standard scipy.linalg.solve function to obtain closed-form solution
 - sparse_cg: uses the conjugate gradient solver as found in scipy.sparse.linalg.cg. As an iterative algorithm, this solver is more appropriate than ‘cholesky’ for large-scale data
 - lsqr: dedicated regularized least-squares routine scipy.sparse.linalg.lsqr. Fastest and uses iterative approach
 - sag: Stochastic Average Gradient descent, and ‘saga’ uses its unbiased and more flexible version named SAGA. Both methods use an iterative procedure, and are often faster than other solvers when both n_samples and n_features are large. Note that ‘sag’ and ‘saga’ fast convergence is only guaranteed on features with approximately the same scale

## Running tf from WSL2
open terminal and launch WSL2:
```
    wsl.exe
```
Activate conda and tensorflow environments:
```
    source $HOME/miniconda3/bin/activate
    conda activate tf
```
Launch VS Code in environment:
```
    code
```