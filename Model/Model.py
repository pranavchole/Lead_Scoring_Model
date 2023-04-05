#!/usr/bin/env python
# coding: utf-8

# In[21]:


# importing Libraries 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt


# In[22]:


# importing data
data = pd.read_csv('/Users/pranavchole/Documents/Projects/untitled folder 2/data.csv')
data


# In[23]:


# Removing Duplicates 
leads_modified_df = data.drop_duplicates(keep = 'first')


# In[24]:


# Understanding data 
print('Shape: ',data.shape)
print()
print(data.info())
print()
print('Number of unique elements in each column: ')
print(data.nunique())


# In[25]:


# Removing all rows with status other than won and lost  
data = data[(data['status'] == 'WON') | (data['status'] == 'LOST')]


# In[26]:


# Count Number of WON and LOST 
count = data['status'].value_counts()
print(count)


# In[27]:


# replacing 9b2d5b4678781e53038e91ea5324530a03f27dc1d0e5f6c9bc9d493a23be9de0 with NULL as given
data.replace('9b2d5b4678781e53038e91ea5324530a03f27dc1d0e5f6c9bc9d493a23be9de0', np.NAN, inplace=True)


# In[28]:


# Calculating Number of null values in each column
for col in data.columns:
    print(col,"=",data[col].isnull().sum())


# In[29]:


# Because of high frequency of null values Drop lead_id column
good_col = [ 'Agent_id','status', 'budget', 'lease','source', 'source_city', 'source_country', 'utm_source',
       'utm_medium', 'des_city','lost_reason','Unnamed: 0', 'room_type', 'des_country','movein']
bad_col = ['lead_id']


# In[30]:


# droping column
data = data.drop(bad_col,axis =1)


# In[31]:


# droping rows having high % null values
print(data.shape)
perc = 70.0 
min_count =  int(((100-perc)/100)*data.shape[1] + 1)
data = data.dropna(axis=0, thresh=min_count)
data.shape


# In[32]:


# Replacing Null Values With mode of its column
for column_name in good_col:
    data[column_name].fillna(data[column_name].mode()[0], inplace=True)


# In[33]:


# Encoding Categorical Values
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
data[good_col] = ordinal_encoder.fit_transform(data[good_col])


# In[34]:


# Dividing data into X and y
X = data[[ 'Agent_id','budget', 'lease', 'source', 'source_city', 'source_country', 'utm_source',
       'utm_medium', 'des_city','lost_reason','Unnamed: 0', 'des_country','movein']]
y = data[['status']]


# In[35]:


# Selecting Features Using Chi-square Test 
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

chi2_f = SelectKBest(chi2,k=9)
X_k = chi2_f.fit_transform(X,y)

X = X_k


# In[36]:


# Splitting Data into test and train 
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state = 1)


# In[37]:


# Using Random Forest Classifier Algoritm with 100 estimators for modeling

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier.fit(train_X,train_y)


# In[38]:


# predictng Values and score
y_pred = classifier.predict(test_X)
y_prob = classifier.predict_proba(test_X)

Score = []
for i in range(9264):
    Score.append(int(100*y_prob[i][1]))


# In[39]:


# Evaluating the performance using metrics such as accuracy, precision, recall and F1-score
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score,precision_score, recall_score
cm = confusion_matrix(test_y, y_pred)
print('Confusion Matrix: ')
print(cm)
print('f1 Score: ',f1_score(test_y, y_pred))
print('Accuracy: ',accuracy_score(test_y, y_pred))
print('Precision Score: ',precision_score(test_y, y_pred))
print('Recall Score: ',recall_score(test_y, y_pred))


# In[40]:


# Score for chances of conversion of lead to WON. 
Score[0]

