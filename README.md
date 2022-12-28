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

## Models Used
TODO
