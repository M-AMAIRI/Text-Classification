# Text-Classification
Text Classification


### Introduction

Text classification is one of the most important tasks in Natural Language Processing. It is the process of classifying text strings or documents into different categories, depending upon the contents of the strings.


# Editor.md

![](https://pandao.github.io/editor.md/images/logos/editormd-logo-180x180.png)

**Table of Contents**

[TOCM]

[TOC]

#H1 header

###Blockquotes


###Links

[Links](http://localhost/)

### importing library and initialisation

```
import numpy as np  
import re  
import nltk  
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

nltk.download('stopwords')
nltk.download('wordnet')

import pickle  
from nltk.corpus import stopwords

movie_data = load_files("txt_sentoken")  
X, y = movie_data.data, movie_data.target


print(y) # listee all files data with labels class
```


####Text Preprocessing 

Once the dataset has been imported, the next step is to preprocess the text. Text may contain numbers, special characters,
and unwanted spaces. Depending upon the problem we face, we may or may not need to remove these special characters and numbers from text.
However, for the sake of explanation, we will remove all the special characters, numbers, and unwanted spaces from our text. 
Execute the following script to preprocess the data:
```
documents = []

from nltk.stem import WordNetLemmatizer

stemmer = WordNetLemmatizer()

for sen in range(0, len(X)):  
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(X[sen]))

    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 

    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)

    # Converting to Lowercase
    document = document.lower()

    # Lemmatization
    document = document.split()

    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)

    documents.append(document)
```
####Converting Text to Numbers :

Different approaches exist to convert text into the corresponding numerical form. The Bag of Words Model and the Word Embedding Model 
are two of the most commonly used approaches. But on this code , we will use the bag of words model to convert our text to numbers.

for more information about that Bag of words model and Word embedding :
- https://en.wikipedia.org/wiki/Bag-of-words_model
- https://en.wikipedia.org/wiki/Word_embedding


max_features :
Therefore we set the max_features parameter to 1500, which means that we want to use 1500 most occurring words 
as features for training our classifier.

min_df :
The next parameter is min_df and it has been set to 5. This corresponds to the minimum number of documents that should contain this feature.
 So we only include those words that occur in at least 5 documents.

 max_df :
 Here 0.7 means that we should include only those words that occur in a maximum of 70% of all the documents. Words that occur in almost every document 
 are usually not suitable for classification because they do not provide any unique information about the document.
 
 Bag of Words
The following script uses the bag of words model to convert text documents into corresponding numerical features:

```
tfidfconverter = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))  # or tfidfconverter = TfidfTransformer()
X = tfidfconverter.fit_transform(documents).toarray()
```
documents = 5 text * 3 folders(eatch folder is a class) ==> X is a matrix of 15 line eatch line with Term Frequency of eatch unique word
detected on the documents 

to store this matrix you can use this script :

```
import numpy as np
mat = np.matrix(X)
with open('outfile.txt','wb') as f:
    for line in mat:
        np.savetxt(f, line, fmt='%.2f')
```

####Training and Testing Sets 

Like any other supervised machine learning problem, we need to divide our data into training and testing sets. To do so, we will use the 
train_test_split utility from the sklearn.model_selection library. Execute the following script:
The above script divides data into 20% test set and 80% training set.
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 
```

####an example about train_test_split

```
>>> import numpy as np
>>> from sklearn.model_selection import train_test_split
>>> X, y = np.arange(10).reshape((5, 2)), range(5)
>>> X
array([[0, 1],
       [2, 3],
       [4, 5],
       [6, 7],
       [8, 9]])
>>> list(y)
[0, 1, 2, 3, 4]


>>> X_train, X_test, y_train, y_test = train_test_split(
...     X, y, test_size=0.33, random_state=42)
...
>>> X_train
array([[4, 5],
       [0, 1],
       [6, 7]])
>>> y_train
[2, 0, 3]
>>> X_test
array([[2, 3],
       [8, 9]])
>>> y_test
[1, 4]
>>>
>>> train_test_split(y, shuffle=False)
[[0, 1, 2], [3, 4]]

```

###Training Text Classification Model and Predicting Sentiment :
We have divided our data into training and testing set. Now is the time to see the real action. 
We will use the Random Forest Algorithm to train our model. You can you use any other model of your choice.

for more information about random forest :
- https://stackabuse.com/random-forest-algorithm-with-python-and-scikit-learn/
- random forest for python (https://towardsdatascience.com/random-forest-in-python-24d0893d51c0)

```
classifier = RandomForestClassifier(n_estimators=1000, random_state=0)  
classifier.fit(X_train, y_train)
```
Now is the time to see the performance of the model that you just created

####Evaluating the Model :
To evaluate the performance of a classification model such as the one that we just trained, 
we can use metrics such as the confusion matrix, F1 measure, and the accuracy.
To find these values, we can use classification_report, confusion_matrix, and accuracy_score utilities 
from the sklearn.metrics library.

to read more about accuracy,precision,recall and F1 Score : 
- https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9

to compute precision and recall for a multi-class classification problem :
- http://text-analytics101.rxnlp.com/2014/10/computing-precision-and-recall-for.html
- https://stats.stackexchange.com/questions/51296/how-do-you-calculate-precision-and-recall-for-multiclass-classification-using-co/51301#51301

```
y_pred = classifier.predict(X_test) 
# simple example about predict function : https://chrisalbon.com/machine_learning/trees_and_forests/random_forest_classifier_example/
print("print(y_pred)")
print([y_pred[1]])

print("print(y_test) : ")
print(y_test)

print("print(y_pred) : ")
print(y_pred)

print("print(confusion_matrix(y_test,y_pred)) : ")
print(confusion_matrix(y_test,y_pred))  
#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html


target_names = ['class 0', 'class 1', 'class 2']
print("print(classification_report(y_test,y_pred)) : ")
print(classification_report(y_test,y_pred,target_names=target_names))
#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html


print("print(accuracy_score(y_test, y_pred)) : ")
print(accuracy_score(y_test, y_pred))
#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
```

Saving and Loading the Model :
We can save our model as a pickle object in Python. To do so, execute the following script:

```
with open('text_classifier', 'wb') as picklefile:  
    pickle.dump(classifier,picklefile)
```


and then To load the model, we can use the following code:

```
with open('text_classifier', 'rb') as training_model:  
    model = pickle.load(training_model)

y_pred2 = model.predict(X_test)

print(confusion_matrix(y_test, y_pred2))  
print(classification_report(y_test, y_pred2))  
print(accuracy_score(y_test, y_pred2))
```



__ M-AMAIRI
