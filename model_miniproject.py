#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn import preprocessing
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import normalize
import seaborn as sns
from sklearn.model_selection import GridSearchCV


# In[2]:


data=pd.read_excel("Nanomaterials.xlsx")


# In[3]:


data.head()


# In[4]:


data.tail()


# In[5]:


data=data.drop(["Index","Ligand1 SMILES","Ligand2 SMILES","Ligand3 SMILES","Ligand4 SMILES"],axis=1)


# In[6]:


label_encoder=preprocessing.LabelEncoder()


# In[7]:


data['Core']=label_encoder.fit_transform(data['Core'])


# In[8]:


data['Shape']=label_encoder.fit_transform(data['Shape'])


# In[9]:


data['Size'] = label_encoder.fit_transform(data['Size'].astype(str))


# In[10]:


data['Size']=label_encoder.fit_transform(data['Size'])


# In[11]:


data


# In[12]:


mean1=data["Zeta potential in water (mv)"].mean()


# In[13]:


mean1


# In[14]:


data["Zeta potential in water (mv)"]=data["Zeta potential in water (mv)"].fillna(mean1)


# In[15]:


mean2=data["Cellular uptake in A549 (106 nm2 cell-1)"].mean()


# In[16]:


mean2


# In[17]:


data["Cellular uptake in A549 (106 nm2 cell-1)"]=data["Cellular uptake in A549 (106 nm2 cell-1)"].fillna(mean2)


# In[18]:


mean3=data["logP"].mean()


# In[19]:


mean3


# In[20]:


data["logP"]=data["logP"].fillna(mean3)


# In[21]:


data


# In[22]:


data["#Ligand1"]=data["#Ligand1"].replace("-",np.nan)
data["#Ligand2"]=data["#Ligand2"].replace("-",np.nan)
data["#Ligand3"]=data["#Ligand3"].replace("-",np.nan)
data["#Ligand4"]=data["#Ligand4"].replace("-",np.nan)


# In[23]:


data


# In[24]:

data=data.fillna(0)
data


# In[25]:


d=normalize(data)


# In[26]:


df_new=pd.DataFrame(d,columns=data.columns)


# In[27]:


df_new


# In[28]:


x=df_new.drop(["logP","Zeta potential in water (mv)","Cellular uptake in A549 (106 nm2 cell-1)",],axis=1)


# In[29]:


y=df_new["Cellular uptake in A549 (106 nm2 cell-1)"]


# In[30]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=9)


# In[31]:


model=KNeighborsRegressor()


# In[32]:


model.fit(x_train,y_train)


# In[33]:


model.score(x_test,y_test)


# In[39]:


import pickle


# In[40]:


#save the model as a pickle in the file
pickle.dump(model,open('saved_model1.ipynb','wb'))


# In[ ]:





# In[ ]:




