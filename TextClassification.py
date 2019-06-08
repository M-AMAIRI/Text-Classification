import numpy as np  
import re  
import nltk  
from sklearn.datasets import load_files  
nltk.download('stopwords')
nltk.download('wordnet')

import pickle  
from nltk.corpus import stopwords

movie_data = load_files("txt_sentoken")  
X, y = movie_data.data, movie_data.target



#print(y) # [0 1 1 ... 1 0 0] array with 2000 elements
#print(len(X))
#We have two categories: "neg" and "pos", therefore 1s and 0s have been added to the target array




#Once the dataset has been imported, the next step is to preprocess the text. Text may contain numbers, special characters, and unwanted spaces. Depending upon the problem we face, we may or may not need to remove these special characters and numbers from text. However, for the sake of explanation, we will remove all the special characters, numbers, and unwanted spaces from our text. Execute the following script to preprocess the data


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


from sklearn.feature_extraction.text import TfidfVectorizer  
tfidfconverter = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))  
X = tfidfconverter.fit_transform(documents).toarray()

#print(X)

'''
a small example about TfidfVectorizer

>>> from sklearn.feature_extraction.text import TfidfVectorizer
>>> corpus = [
...     'This is the first document.',
...     'This document is the second document.',
...     'And this is the third one.',
...     'Is this the first document?',
... ]
>>> vectorizer = TfidfVectorizer()
>>> X = vectorizer.fit_transform(corpus)
>>> print(vectorizer.get_feature_names())
['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this']
>>> print(X.shape)
(4, 9)


'''


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 
# Split arrays or matrices into random train and test subsets



#print("len(X_train)")
#print(len(X_train))

#print("len(X_test)")
#print(len(X_test))

#print("len(y_train)")
#print(len(y_train))

#print("len(y_test)")
#print(len(y_test))


'''

an example about train_test_split :

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

'''


# Training Text Classification Model and Predicting Sentiment
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=1000, random_state=0)  
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)  

#print(len(y_pred))


# Evaluating the Model


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))  
print(accuracy_score(y_test, y_pred))


#Saving and Loading the Model
with open('text_classifier', 'wb') as picklefile:  
    pickle.dump(classifier,picklefile)




#To load the model, we can use the following code:
'''with open('text_classifier', 'rb') as training_model:  
    model = pickle.load(training_model)
    y_pred = model.predict(X_test)
'''



 
