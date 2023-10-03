# -*- coding: utf-8 -*-
"""
Due by March 19, 11pm
# Application of ML in analyzing text documents
In this project I took advantage of scikit-learn in working with text documents. 
"""

# loading need libraries
import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

"""# Dataset characteristics

Here is the structure/properties of the dataset. 
To have a faster code, I just pick 2 class labels out of 20 from this dataset. Each data point is a text document.

"""

categories = ['comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train',
    categories=categories, shuffle=False, random_state=42)
twenty_test = fetch_20newsgroups(subset='test',
    categories=categories, shuffle=False, random_state=42)

print("Dataset properties on train section:")
print("\t Number of training data points: %d" % len(twenty_train.data))
print("\t Number of test data points: %d" % len(twenty_test.data))
print("\t Number of Classes: %d" % len(categories))
print("\t Class names: " ,(twenty_train.target_names))

""" A sample of dataset
"""

print("\n".join(twenty_train.data[0].split("\n")))

"""the category name of the instance can be found as follows:"""

print(twenty_train.target[0])
print(twenty_train.target_names[twenty_train.target[0]])

"""To get the categries of a range of data, e.g., first 10 samples, I can do something like this:"""

for t in twenty_train.target[:10]:
    print(twenty_train.target_names[t])

""" Feature extraction
since the data is text, to run classification models on the dataset I turn them into vectors with numerical features.
Therefore, in this section, we extract features using the **Bag of Words** method. To this regard,


*   Assign an integer ID to each word in the dataset (like a dictionary).
*   For each data point ( document i), count the number of occurances of word w and put it in X[i,j] where i is the i'th document and j is the index of the word w in the dictionary.
Thus, if we have e.g., 10000 data points and 100000 words in the dictionary, then X will be a 10,000 by 100,000 matrix, which is huge! 
The good news is that most elements of the matrix X are zero (not all the words are used in every document). 
Therefore, it is possible to (somehow) just store non-zero elements and save up a lot of memory. 
Fortunately, the library that I use supports using "sparse" data representations, meaning that it does not actually store all the zero-values.

# Tokenizing with scikit-learn
In the following part I extract whole words that have been used in the dataset and compute their occurance count in each document. 
This shows number of documents are 1178 and number of features (unique words in the whole set of documents) is 24614.

"""

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
X_train_counts.shape

"""Vectorize test data without fitting."""

X_test_counts = count_vect.transform(twenty_test.data)
X_test_counts.shape


tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
print(X_train_tf.shape)

"""Transform test data without fitting."""

X_test_tf = tf_transformer.transform(X_test_counts)
print(X_test_tf.shape)

"""# Document classification
Support Vector Machines (SVMs) are one of the most common classifiers in practices. 
Here I train an SVM classifier on the transformed features, and get classification accuracy on test data.
"""

clf = SVC().fit(X_train_tf, twenty_train.target)

preds = clf.predict(X_test_tf)

accuracy = np.mean(preds == twenty_test.target)
print(f'Accuracy on test data is {accuracy*100}%.')

""" Classify two tiny documents using the trained classifier and print out classification results."""

docs_new = ['OpenGL on the GPU is fast', 'Doctor takes care of patients']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)

for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, twenty_train.target_names[category]))

"""Part 1 - Naive Bayes with Bernoulli : train a naive bayes classifier with bernoulli features.

As the first step, convert all features to binary values.
"""

X_train_b = X_train_tf.sign()
X_test_b = X_test_tf.sign()

"""Vectorized data is stored using sparse matrix. Since most of the entries of the matrix are zero, it is much more efficient to somehow only store the non-zero values; that is what sparse matrices do.

It is not efficient to directly loop through all the elements of sparse matrices. It is better to only loop over non-zero (in our case "1") features with the following code:
"""

# array of document index
print(X_train_b.nonzero()[0])
# array of feature index
print(X_train_b.nonzero()[1])

# i is the index of data points.
# j represents that the jth feature of vector x_i equals to 1.
for i, j in zip(*X_train_b.nonzero()):
  print(i, j)
  break

"""
This means the 360th feature of 0th document, ... , and 24464th feature of 1177th document have non-zero values.
Access a specific element by providing both document and feature indexs.

"""

print(X_train_b[0, 360])
print(X_train_b[0, 361])

"""
Let (x,y) be a datapoint where x=\[x_1, x_2, ...x_d] is a d-dimensional vector and y∈\{0, 1}. Each x_i\{0,1} is a binary feature.

Naive bayes assumes that the features are independent given the class label:

+ P(x|y=0)=∏ P(x_i|y=0, θ)
+ P(x|y=1)=∏ P(x_i|y=1, θ)

The set of parameters of the classifier are denoted by θ=\{α, β, γ}. The parameters are defined by

+ α=\[α_1, α_2, ..., α_d\]
+ for i∈ \{1,..., d}, we have α_i=P(x_i=1|y=0, θ)
+ In other words, α_i is the parameter of the Bernoulli distribution P(x_i=1|y=0, θ)
+ similarly:
+ β=\[β_1, β_2, ..., β_d\]
+ for i∈ \{1,..., d}, we have, β_i=P(x_i=1|y=1, θ)
+ Finally, γ = P(y=0|θ) is the probability that a randomly generated point from the distribution is from class 0

Derive the maximum likelihood formula for estimation of α, β and γ based on the observed data points.
Use the derived formula to implement the following functions. 
Compute the maximum likehood estimation of α, β and γ based on the training set.
"""

def estimate_alpha_with_mle():
  # Derive expression of alpha value with MLE.
  # alpha = P(x_i=1|y=0, theta)
  a = 0
  for i in range (X_train_b.shape[0]):
    if twenty_train.target[i] == 0:
      a += 1

  m = np.zeros(X_train_b.shape[1])
  for i, j in zip(*X_train_b.nonzero()):
    if twenty_train.target[i] == 0:
      m[j] += 1

  return m/a

def estimate_beta_with_mle():
  # Derive expression of beta value with MLE
  # beta = P(x_i=1|y=1, theta)
  b = 0
  for i in range (X_train_b.shape[0]):
    if twenty_train.target[i] == 1:
      b += 1

  n = np.zeros(X_train_b.shape[1])
  for i, j in zip(*X_train_b.nonzero()):
    if twenty_train.target[i] == 1:
      n[j] += 1

  return n/b


def estimate_gamma_with_mle():
  # Derive expression of gamma value with MLE
  # gamma = P(y=0|theta)
  a = 0
  for i in range (X_train_b.shape[0]):
    if twenty_train.target[i] == 0:
      a += 1
  return a/X_train_b.shape[0]

gamma = estimate_gamma_with_mle()
beta = estimate_beta_with_mle()
alpha = estimate_alpha_with_mle()

print(f'gamma is {gamma}')
print(f'beta is {beta[:10]}')
print(f'alpha is {alpha[:10]}')

"""
Now use the estimated α, β and γ for our Naive Bayes classifier and classify data points. 
Because P(y|x, θ)=P(x|y, θ) P(y|θ) / P(x|θ), caluclate P(y|x, θ) with derived α, β and γ. 
Implement the predict function of the Naive bayes classfier to conduct the calculation of P(y|x, θ). 
The predict function should return the predicted labels for the input dataset X (For a data point x, if P(y=1|x, θ) > P(y=0|x,θ), 
the model should predict x with label y=1).
"""

class NaiveBayes:
    def __init__(self, gamma, alpha, beta) -> None:
      self.gamma = gamma
      self.alpha = alpha
      self.beta = beta

    # X is the input dataset.
    def predict(self, X):
      predictions = np.zeros(X.shape[0])

      for i in range(X.shape[0]):
        p_0 = gamma
        p_1 = 1-gamma
        for j in range(X.shape[1]):
          if X[i,j] == 0:
            p_0 *= 1-alpha[j]
            p_1 *= 1-beta[j]
          else:
            p_0 *= alpha[j]
            p_1 *= beta[j]

        if p_0 >= p_1:
          predictions[i] = 0
        else: predictions[i] = 1

      return predictions

"""Report the classification accuracy on both the train dataset and on the test dataset."""

NBC = NaiveBayes(gamma, alpha, beta)
correct = 0
X_train_b_predict = NBC.predict(X_train_b)
for i in range(X_train_b.shape[0]):
  if X_train_b_predict[i] == twenty_train.target[i]:
    correct +=1
print("The accuracy on train data is",correct/X_train_b.shape[0])

correct = 0
X_test_b_predict = NBC.predict(X_test_b)
for i in range(X_test_b.shape[0]):
  if X_test_b_predict[i] == twenty_test.target[i]:
    correct +=1
print("The accuracy on test data is",correct/X_test_b.shape[0])

"""
Estimate α, β and γ with maximum a posteriori (MAP).

Use the following prior distribution:
+ P(θ) = P(γ).(∏P(α_i)).(∏P(β_i))
+ For simplicity, we assume that all the parameters have the same prior distribution. In other words, for all values of a∈\[0,1] we have

 P(γ=a)=P(α_1=a)=P(α_2=a)=...=P(β_1=a)=P(β_2=a)=....=f(a). Here, f(a) is the p.d.f of these distributions and is defined similar to what we had page 8 of lecture 9:

+ for a <= 0.5 then f(a) = 4a
+ for a > 0.5 then f(a) = 4 - 4a

* Derive the formula for MAP and write it in your report. Estimate the parameters using MAP on the training set.
"""

def estimate_alpha_with_map():
  # Derive expression of alpha value with MAP.
  # alpha = P(x_i=1|y=0, theta)

  a = 0
  for i in range (X_train_b.shape[0]):
    if twenty_train.target[i] == 0:
      a += 1

  m = np.zeros(X_train_b.shape[1])
  for i, j in zip(*X_train_b.nonzero()):
    if twenty_train.target[i] == 0:
      m[j] += 1

  for i in range(m.shape[0]):
    if m[i]/(a+1) >= 0.5: return (m+1)/(a+1)
  else: return m/(a+1)

def estimate_beta_with_map():
  # Derive expression of beta value with MLE
  # beta = P(x_i=1|y=1, theta)
  b = 0
  for i in range (X_train_b.shape[0]):
    if twenty_train.target[i] == 1:
      b += 1

  n = np.zeros(X_train_b.shape[1])
  for i, j in zip(*X_train_b.nonzero()):
    if twenty_train.target[i] == 1:
      n[j] += 1

  for i in range(n.shape[0]):
    if n[i]/(b+1) >= 0.5: return (n+1)/(b+1)
  else: n/(b+1)

def estimate_gamma_with_map():
  # Derive expression of gamma value with MAP
  # gamma = P(y=0|theta)
  a = 0
  for i in range (X_train_b.shape[0]):
    if twenty_train.target[i] == 0:
      a += 1

  if (a/(X_train_b.shape[0]+1) < 0.5): return a/(X_train_b.shape[0]+1)
  else: return (a+1)/(X_train_b.shape[0]+1)

gamma = estimate_gamma_with_map()
beta = estimate_beta_with_map()
alpha = estimate_alpha_with_map()

print(f'gamma is {gamma}')
print(f'beta is {beta[:10]}')
print(f'alpha is {alpha[:10]}')

"""Run the naive bayes model with estimated parameters (using MAP) both on the train set and test set. Report the classification accuracies."""

NBC = NaiveBayes(gamma, alpha, beta)
correct = 0
X_train_b_predict = NBC.predict(X_train_b)
for i in range(X_train_b.shape[0]):
  if X_train_b_predict[i] == twenty_train.target[i]:
    correct +=1
print("The accuracy on train data is",correct/X_train_b.shape[0])

correct = 0
X_test_b_predict = NBC.predict(X_test_b)
for i in range(X_test_b.shape[0]):
  if X_test_b_predict[i] == twenty_test.target[i]:
    correct +=1
print("The accuracy on test data is",correct/X_test_b.shape[0])

"""
Part2--SVM Classifier : In the following, we will train SVM classifiers with linear/Gaussian kernels.
"""

categories = ['alt.atheism', 'soc.religion.christian',
              'comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train',
    categories=categories, shuffle=True, random_state=42)
twenty_test = fetch_20newsgroups(subset='test',
    categories=categories, shuffle=True, random_state=42)

"""Train a linear SVM classifier with train data. Report classification accuracy on train and test data."""

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import SVC

# construct CountVectorizer and TfidfTransformer
count_vect = CountVectorizer()
tfidf_transformer = TfidfTransformer(use_idf=False)
# Transform the trian and test data
X_train_counts = count_vect.fit_transform(twenty_train.data)
X_train_transformed = tfidf_transformer.fit_transform(X_train_counts)

X_test_counts = count_vect.transform(twenty_test.data)
X_test_transformed = tfidf_transformer.transform(X_test_counts)

# Train a SVM classifier
SVM_classifier = SVC(kernel='linear').fit(X_train_transformed, twenty_train.target)

# Predict on train and test data
train_pred = SVM_classifier.predict(X_train_transformed)
test_pred = SVM_classifier.predict(X_test_transformed)

# Calculate the accuracy of train data
train_accuracy = 0
for i in range(train_pred.shape[0]):
  if twenty_train.target[i] == train_pred[i]:
    train_accuracy+=1

# Calculate the accuracy of test data
test_accuracy = 0
for i in range(test_pred.shape[0]):
  if twenty_test.target[i] == test_pred[i]:
    test_accuracy+=1

print("The accuracy on train data is", train_accuracy/train_pred.shape[0])
print("The accuracy on test data is", test_accuracy/test_pred.shape[0])

"""

* Use RBF SVM (which is a version of SVM that uses Gaussian kernel) on the above. 
Report the classification accuracy on three different gamma values on the test set: 0.70, 0.650, and 0.60 (see https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html).

"""

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import SVC

# construct CountVectorizer and TfidfTransformer
count_vect = CountVectorizer()
tfidf_transformer = TfidfTransformer(use_idf=False)
# Transform the trian and test data
X_train_counts = count_vect.fit_transform(twenty_train.data)
X_train_transformed = tfidf_transformer.fit_transform(X_train_counts)

X_test_counts = count_vect.transform(twenty_test.data)
X_test_transformed = tfidf_transformer.transform(X_test_counts)

# Train an RBF SVM classifier with different gamma values
for gamma_val in [0.7, 0.65, 0.6]:
    clf = SVC(kernel='rbf', gamma=gamma_val)
    clf.fit(X_train_transformed, twenty_train.target)


    # Predict on train and test data
    train_pred = clf.predict(X_train_transformed)
    test_pred = clf.predict(X_test_transformed)

    # Calculate the accuracy of train data
    train_accuracy = 0
    for i in range(train_pred.shape[0]):
      if twenty_train.target[i] == train_pred[i]:
        train_accuracy+=1

    # Calculate the accuracy of test data
    test_accuracy = 0
    for i in range(test_pred.shape[0]):
      if twenty_test.target[i] == test_pred[i]:
        test_accuracy+=1

    print("the value of gamma is ",gamma_val)
    print("The accuracy on train data is", train_accuracy/train_pred.shape[0])
    print("The accuracy on test data is", test_accuracy/test_pred.shape[0])

"""Turn on TfidfTransformer(use_idf=True) and Run kernel SVM, report the results and justify them."""

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import SVC

# construct CountVectorizer and TfidfTransformer
count_vect = CountVectorizer()
tfidf_transformer = TfidfTransformer(use_idf=True)
# Transform the trian and test data
X_train_counts = count_vect.fit_transform(twenty_train.data)
X_train_transformed = tfidf_transformer.fit_transform(X_train_counts)

X_test_counts = count_vect.transform(twenty_test.data)
X_test_transformed = tfidf_transformer.transform(X_test_counts)

# Train an RBF SVM classifier with different gamma values
for gamma_val in [0.7, 0.65, 0.6]:
    clf = SVC(kernel='rbf', gamma=gamma_val)
    clf.fit(X_train_transformed, twenty_train.target)


    # Predict on train and test data
    train_pred = clf.predict(X_train_transformed)
    test_pred = clf.predict(X_test_transformed)

    # Calculate the accuracy of train data
    train_accuracy = 0
    for i in range(train_pred.shape[0]):
      if twenty_train.target[i] == train_pred[i]:
        train_accuracy+=1

    # Calculate the accuracy of test data
    test_accuracy = 0
    for i in range(test_pred.shape[0]):
      if twenty_test.target[i] == test_pred[i]:
        test_accuracy+=1

    print("the value of gamma is ",gamma_val)
    print("The accuracy on train data is", train_accuracy/train_pred.shape[0])
    print("The accuracy on test data is", test_accuracy/test_pred.shape[0])
