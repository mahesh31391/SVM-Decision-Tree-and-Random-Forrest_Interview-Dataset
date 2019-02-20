
# coding: utf-8

# In[2]:

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
get_ipython().magic('matplotlib inline')


# In[3]:

df=pd.read_csv('E:\\pyathon_Class\\Class_Home_Practice\\Interview_project\\Interview_Dummies_2.csv')


# In[4]:

df.head()


# In[5]:

X=df.drop('Observed Attendance',axis=1)


# In[6]:

y=df['Observed Attendance']


# In[7]:

from sklearn.model_selection import train_test_split


# In[8]:

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=50)


# In[9]:

from sklearn.linear_model import LogisticRegression


# In[10]:

logmodel=LogisticRegression()


# In[11]:

logmodel.fit(X_train,y_train)


# In[12]:

predictions=logmodel.predict(X_test)


# In[13]:

from sklearn.metrics import confusion_matrix,classification_report


# In[14]:

print ('Confusion Matrix')
print (confusion_matrix(y_test,predictions))
print('\n')
print ('Classification Reprot')
print (classification_report(y_test,predictions))       


# KNN 

# In[15]:

from sklearn.neighbors import KNeighborsClassifier


# In[16]:

error_rate = []

# Will take some time
for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


# In[17]:

plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# In[18]:

# NOW WITH K=20
knn = KNeighborsClassifier(n_neighbors=20)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=20')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))


# # Appling SVM

# In[19]:

from sklearn.svm import SVC


# In[20]:

model=SVC()


# In[21]:

model.fit(X_train,y_train)


# In[22]:

predictions = model.predict(X_test)


# In[23]:

print ('Confusion Matrix')
print (confusion_matrix(y_test,predictions))
print('\n')
print ('Classification Reprot')
print (classification_report(y_test,predictions))   


# # Finding best parameters

# In[24]:

param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf','linear']} 


# In[25]:

from sklearn.model_selection import GridSearchCV


# In[26]:

grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)


# In[27]:

grid.fit(X_train,y_train)


# In[28]:

grid.best_estimator_


# In[29]:

grid.best_params_


# In[30]:

grid_predictions = grid.predict(X_test)


# In[31]:

print('\n')
print(confusion_matrix(y_test,grid_predictions))
print('\n')
print(classification_report(y_test,grid_predictions))


# # Decision Tree

# In[32]:

from sklearn.tree import DecisionTreeClassifier


# In[33]:

dtree=DecisionTreeClassifier(criterion='gini')


# In[34]:

dtree.fit(X_train,y_train)


# In[35]:

predictions=dtree.predict(X_test)


# In[36]:

print('\n')
print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))


# # Random Forest

# In[37]:

from sklearn.ensemble import RandomForestClassifier


# In[38]:

rfc = RandomForestClassifier(n_estimators=100)


# In[39]:

rfc.fit(X_train,y_train)
Predictions_rfc=rfc.predict(X_test)


# In[40]:

print(confusion_matrix(y_test,Predictions_rfc))
print('\n')
print(classification_report(y_test,Predictions_rfc))


# # Voting Classifier

# In[41]:

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier


# In[42]:

clf1=LogisticRegression(random_state=101)
clf2=RandomForestClassifier(random_state=101)
clf3=GaussianNB()


# In[43]:

X=df.drop(['Observed Attendance'],axis=1)
y=df['Observed Attendance']
eclf1=VotingClassifier(estimators=[('lr',clf1),('rf',clf2),('gnb',clf3)],weights=(1,2,3),voting='hard')
eclf1=eclf1.fit(X_train,y_train)
eclf1=eclf1.fit(X,y)
print(eclf1.predict(X))


# In[44]:

eclf2=VotingClassifier(estimators=[('lr',clf1),('rf',clf2),('gnb',clf3)],weights=(1,2,3),voting='hard')
eclf2=eclf2.fit(X_train,y_train)
predict=eclf2.predict(X_test)
print(classification_report(y_test,predict))


# In[45]:

eclf3=VotingClassifier(estimators=[('lr',clf1),('rf',clf2),('gnb',clf3)],weights=(3,2,1),voting='soft')
eclf3=eclf3.fit(X_train,y_train)
predict1=eclf3.predict(X_test)
print(classification_report(y_test,predict1))


# # Conclusion

# Logistics Regresion has Avg precision of 80%
# KNN model has Avg precision of 79%
# SVM model has Avg precision of 80%
# DT Model has precision of 79%
# RF Model has precision of 79%
# Logistic regression or SVM is better models to predict interview attendance 
# 

# In[ ]:



