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


# Text Preprocessing :

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


# Converting Text to Numbers :

tfidfconverter = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))  # or tfidfconverter = TfidfTransformer()
X = tfidfconverter.fit_transform(documents).toarray()


# Training and Testing Sets :

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 


# Training Text Classification Model and Predicting Sentiment :

classifier = RandomForestClassifier(n_estimators=1000, random_state=0)  
classifier.fit(X_train, y_train)


# Evaluating the Model 

y_pred = classifier.predict(X_test) 

print("print(confusion_matrix(y_test,y_pred)) : ")
print(confusion_matrix(y_test,y_pred))  

target_names = ['class 0', 'class 1', 'class 2']
print("print(classification_report(y_test,y_pred)) : ")
print(classification_report(y_test,y_pred,target_names=target_names))

print("print(accuracy_score(y_test, y_pred)) : ")
print(accuracy_score(y_test, y_pred))


'''
Saving and Loading the Model :
We can save our model as a pickle object in Python. To do so, execute the following script:

with open('text_classifier', 'wb') as picklefile:  
    pickle.dump(classifier,picklefile)



and then To load the model, we can use the following code:

with open('text_classifier', 'rb') as training_model:  
    model = pickle.load(training_model)

y_pred2 = model.predict(X_test)

'''
