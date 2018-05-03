
# coding: utf-8

# In[1]:


import nltk
from nltk.corpus import movie_reviews
from random import shuffle


# In[3]:


documents = [(list(movie_reviews.words(fileid)), category) for category in movie_reviews.categories()for fileid in movie_reviews.fileids(category)]
shuffle(documents)
print (documents[0])


# In[4]:


len(documents)


# In[7]:


#拿出2000个最常用的词
all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
#print (all_words)
word_features = list(all_words.keys())[:2000]


# In[8]:


def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features


# In[10]:


print (document_features(movie_reviews.words('pos/cv957_8737.txt')))


# In[11]:


featuresets = [(document_features(d), c) for (d, c) in documents]


# In[12]:


train_set, test_set = featuresets[100:], featuresets[:100]


# In[13]:


classifier = nltk.NaiveBayesClassifier.train(train_set)


# In[15]:


print (nltk.classify.accuracy(classifier, test_set))


# In[16]:


classifier.show_most_informative_features(10)


# In[17]:


test_text = "I love this movie, very interesting"
test_set = document_features(test_text.split())
test_set


# In[18]:


print (classifier.classify(test_set))

