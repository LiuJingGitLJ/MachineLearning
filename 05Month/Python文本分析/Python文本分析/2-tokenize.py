
# coding: utf-8

# In[3]:


import nltk.data
text = 'this’s a sent tokenize test. this is sent two. is this sent three? sent 4 is cool! Now it’s your turn.'
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
tokenizer.tokenize(text)


# In[5]:


spanish_tokenizer = nltk.data.load('tokenizers/punkt/spanish.pickle')
spanish_tokenizer.tokenize('Hola amigo. Estoy bien.')


# In[22]:


#中文用结吧分词
import jieba
seg_list = jieba.cut("我来到北京清华大学",cut_all=True)
print ("Full Mode:", "/ ".join(seg_list)) #全模式

seg_list = jieba.cut("我来到北京清华大学",cut_all=False)
print ("Default Mode:", "/ ".join(seg_list)) #精确模式


# In[23]:


from nltk.tokenize import word_tokenize
word_tokenize('Hello World.')


# In[7]:


text = nltk.word_tokenize('Dive into NLTK: Part-of-speech tagging and POS Tagger')
text


# In[8]:


nltk.pos_tag(text)


# In[12]:


#安装tagset
nltk.help.upenn_tagset('NNP')


# In[14]:


#安装treebank
from nltk.corpus import treebank
len(treebank.tagged_sents())


# In[15]:


train_data = treebank.tagged_sents()[:3000]
test_data = treebank.tagged_sents()[3000:]
train_data[0]


# In[16]:


from nltk.tag import tnt
tnt_pos_tagger = tnt.TnT()
tnt_pos_tagger.train(train_data)
tnt_pos_tagger.evaluate(test_data)


# In[25]:


import pickle
f = open('tnt_treebank_pos_tagger.pickle', 'wb+')
pickle.dump(tnt_pos_tagger, f)
f.close()


# In[27]:


tnt_pos_tagger.tag(nltk.word_tokenize('this is a tnt treebank tnt tagger'))

