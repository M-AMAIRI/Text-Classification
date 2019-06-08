# Text-Classification
Text Classification


# Introduction !

Text classification is one of the most important tasks in [Natural Language Processing](https://stackabuse.com/what-is-natural-language-processing/). It is the process of classifying text strings or documents into different categories, depending upon the contents of the strings.


# Dataset


The dataset consists of a total of 2000 documents. Half of the documents contain positive reviews regarding a movie while the remaining half contains negative reviews. Further details regarding the dataset can be found at [this link](http://www.cs.cornell.edu/people/pabo/movie-review-data/poldata.README.2.0.txt).

Unzip or extract the dataset once you download it .
  File names consist of a cross-validation tag plus the name of the
  original html file.  The ten folds used in the Pang/Lee ACL 2004 paper's
  experiments were:

     fold 1: files tagged cv000 through cv099, in numerical order
     fold 2: files tagged cv100 through cv199, in numerical order     
     ...
     fold 10: files tagged cv900 through cv999, in numerical order

  Hence, the file neg/cv114_19501.txt, for example, was labeled as
  negative, served as a member of fold 2, and was extracted from the
  file 19501.html in polarity_html.zip (see below).


##  Text Preprocessing

#Once the dataset has been imported, the next step is to preprocess the text. Text may contain numbers, special characters, and unwanted spaces. Depending upon the problem we face, we may or may not need to remove these special characters and numbers from text. However, for the sake of explanation, we will remove all the special characters, numbers, and unwanted spaces from our text. Execute the following script to preprocess the data


## Converting Text to Numbers Using TF-IDF

In information retrieval, tf–idf or TFIDF, short for term frequency–inverse document frequency, is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus . It is often used as a weighting factor in searches of information retrieval, text mining, and user modeling. The tf–idf value increases proportionally to the number of times a word appears in the document and is offset by the number of documents in the corpus that contain the word

'''
from sklearn.feature_extraction.text import TfidfVectorizer tfidfconverter = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english')) X = tfidfconverter.fit_transform(documents).toarray()

'''

## Training and Testing Sets

just for using train_test_split() function to Split arrays or matrices into random train and test subsets

"""
from sklearn.model_selection import train_test_split X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
"""

and then To load the model, we can use the following code:

"""
with open('text_classifier', 'rb') as training_model: 
	model = pickle.load(training_model)

"""

## Save the Model

We can save our model as a  `pickle`  object in Python. To do so, execute the following script:
"""
with open('text_classifier', 'wb') as picklefile: 
  pickle.dump(classifier,picklefile)

"""

`Refrences :
[https://stackabuse.com/text-classification-with-python-and-scikit-learn/](https://stackabuse.com/text-classification-with-python-and-scikit-learn/)

[https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)

[https://en.wikipedia.org/wiki/Tf%E2%80%93idf](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)



