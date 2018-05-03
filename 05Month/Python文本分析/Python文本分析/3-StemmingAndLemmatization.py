
# coding: utf-8

# In[2]:


from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()


# ### 词形归一化  ###
# - stemming:词干提取 去掉ing和ed之类的
# - lemmatization：转成单数

# In[4]:


porter_stemmer.stem('maximum')


# In[5]:


porter_stemmer.stem('presumably')


# In[6]:


porter_stemmer.stem('multiply')


# In[9]:


#安装WordNet
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
wordnet_lemmatizer.lemmatize('dogs')

