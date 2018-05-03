
# coding: utf-8

# In[6]:


from nltk.corpus import names
import random


# In[7]:


names = ([(name, 'male') for name in names.words('male.txt')] + [(name, 'female') for name in names.words('female.txt')])
random.shuffle(names)


# In[8]:


len(names)


# In[9]:


names[0:10]


# In[10]:


def gender_features(word):
    return {'last_letter': word[-1]}


# In[11]:


featuresets = [(gender_features(n), g) for (n, g) in names]
featuresets[0:10]


# In[12]:


train_set, test_set = featuresets[500:], featuresets[:500]


# In[13]:


from nltk import NaiveBayesClassifier
nb_classifier = NaiveBayesClassifier.train(train_set)
nb_classifier.classify(gender_features('Gary'))


# In[14]:


from nltk import classify
classify.accuracy(nb_classifier, test_set)


# In[15]:


nb_classifier.show_most_informative_features(5)

