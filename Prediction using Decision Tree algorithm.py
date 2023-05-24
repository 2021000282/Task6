#!/usr/bin/env python
# coding: utf-8

# In[44]:


#importing libraries
import numpy as np
import pandas as pd
import sklearn.metrics as sm
import seaborn as sns
import matplotlib.pyplot as mt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree


# In[45]:


#loading dataset
data=pd.read_csv("C:/Users/user9/Downloads/Iris.csv")
data


# In[46]:


data=data.drop(['Id'],axis=1)
data


# In[31]:


data.describe()   #gives summary of data


# In[32]:


sns.pairplot(data,hue="Species") #gives scatterplot of Iris data


# Here, we can see that the Setosa iris species makes different cluster as compared to other two species.

# In[33]:


data.corr() #gives correlation table of Iris data


# In[34]:


data.drop(['Species'],axis=1)


# In[38]:


#Training the model
x=data.iloc[:, [0,1,2,3]].values
y=data['Species'].values
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=42)
print("Train split:",xtrain.shape)
print("Test split:",xtest.shape)     #gives dimension of train test data  


# In[39]:


#make decision tree classifier
dtree=DecisionTreeClassifier()
dtree.fit(xtrain,ytrain)


# In[40]:


#classification report
ypred=dtree.predict(xtest)
print(classification_report(ytest,ypred))      #the accuracy of data is 1 or 100%


# In[41]:


#confusion matrix
cm=confusion_matrix(ytest,ypred)
cm


# In[43]:


#visualising the model
mt.figure(figsize=(15,8))
target=data['Species']
tree=plot_tree(dtree,feature_names=data.columns,precision=2,filled=True,rounded=True,class_names=target.values)


# The Decision tree classifier is created and visualize it graphically where the 100% accuracy is evaluated.
