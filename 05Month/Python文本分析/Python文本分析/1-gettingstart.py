
# coding: utf-8

# In[2]:


import nltk
nltk.download()
# http://www.nltk.org/


# In[1]:


from nltk.corpus import brown

brown.words()[0:10]


# In[2]:


brown.tagged_words()[0:10]


# In[3]:


from nltk import sent_tokenize, word_tokenize, pos_tag

text = "Machine learning is the science of getting computers to act without being explicitly programmed. In the past decade, machine learning has given us self-driving cars, practical speech recognition, effective web search, and a vastly improved understanding of the human genome. Machine learning is so pervasive today that you probably use it dozens of times a day without knowing it. Many researchers also think it is the best way to make progress towards human-level AI. In this class, you will learn about the most effective machine learning techniques, and gain practice implementing them and getting them to work for yourself. More importantly, you'll learn about not only the theoretical underpinnings of learning, but also gain the practical know-how needed to quickly and powerfully apply these techniques to new problems. Finally, you'll learn about some of Silicon Valley's best practices in innovation as it pertains to machine learning and AI."


# In[4]:


#将文章转换成句子的组合，先得下载好punkt
sents = sent_tokenize(text)
sents


# In[5]:


##将文章转换成词的组合，先得下载好
tokens = word_tokenize(text)
tokens


# In[6]:


#词性
tagged_tokens = pos_tag(tokens)
tagged_tokens

