
# coding: utf-8

# In[2]:


from gensim.models import word2vec
from nltk.corpus import gutenberg
gutenberg.fileids()


# In[4]:


import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
bible_kjv_words = gutenberg.words('bible-kjv.txt')
bible_kjv_sents = gutenberg.sents('bible-kjv.txt')  
len(bible_kjv_words)


# In[5]:


from string import punctuation
punctuation


# In[6]:


discard_punctuation_and_lowercased_sents = [[word.lower() for word in sent if word not in punctuation] for sent in bible_kjv_sents]


# In[7]:


bible_kjv_sents[0]


# In[8]:


discard_punctuation_and_lowercased_sents[0]


# In[9]:


bible_kjv_word2vec_model = word2vec.Word2Vec(discard_punctuation_and_lowercased_sents, min_count=5, size=200)


# In[10]:


bible_kjv_word2vec_model.save("bible_word2vec_gensim")


# In[11]:


bible_kjv_word2vec_model.most_similar(["god"], topn=30)

