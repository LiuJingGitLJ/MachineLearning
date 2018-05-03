
# coding: utf-8

# In[1]:


get_ipython().magic('matplotlib inline')
import matplotlib.pylab as plt
import seaborn as sns
from tsfresh.examples.robot_execution_failures import download_robot_execution_failures, load_robot_execution_failures
from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report


#http://tsfresh.readthedocs.io/en/latest/text/quick_start.html


# In[2]:


download_robot_execution_failures()
df, y = load_robot_execution_failures()
df.head()


# In[3]:


df[df.id == 3][['time', 'a', 'b', 'c', 'd', 'e', 'f']].plot(x='time', title='Success example (id 3)', figsize=(12, 6));
df[df.id == 20][['time', 'a', 'b', 'c', 'd', 'e', 'f']].plot(x='time', title='Failure example (id 20)', figsize=(12, 6));


# In[4]:


extraction_settings = ComprehensiveFCParameters()


# In[5]:


#column_id (str) – The name of the id column to group by
#column_sort (str) – The name of the sort column.
X = extract_features(df, 
                     column_id='id', column_sort='time',
                     default_fc_parameters=extraction_settings,
                     impute_function= impute)


# In[6]:


X.head()


# In[7]:


X.info()


# In[8]:


X_filtered = extract_relevant_features(df, y, 
                                       column_id='id', column_sort='time', 
                                       default_fc_parameters=extraction_settings)


# In[9]:


X_filtered.head()


# In[10]:


X_filtered.info()


# In[11]:


X_train, X_test, X_filtered_train, X_filtered_test, y_train, y_test = train_test_split(X, X_filtered, y, test_size=.4)


# In[12]:


cl = DecisionTreeClassifier()
cl.fit(X_train, y_train)
print(classification_report(y_test, cl.predict(X_test)))


# In[13]:


cl.n_features_


# In[14]:


cl2 = DecisionTreeClassifier()
cl2.fit(X_filtered_train, y_train)
print(classification_report(y_test, cl2.predict(X_filtered_test)))


# In[15]:


cl2.n_features_

