#!/usr/bin/env python
# coding: utf-8

# In[4]:


pip install tensorflow


# In[10]:


import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Flatten
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


# In[4]:


(X_train,y_train),(X_test,y_test)=keras.datasets.mnist.load_data()


# In[6]:





# In[11]:


plt.imshow(X_train[0])


# In[12]:


X_train = X_train/255
X_test = X_test/255


# In[13]:


X_train[0]


# In[15]:


model = Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(128,activation="relu"))
model.add(Dense(10,activation="softmax"))


# In[16]:


model.summary()


# In[17]:


model.compile(loss='sparse_categorical_crossentropy',optimizer='Adam')


# In[18]:


model.fit(X_train,y_train,epochs=10,validation_split=0.2)


# In[20]:


y_prob =model.predict(X_test)
y_pred = y_prob.argmax(axis=1)


# In[25]:


from sklearn.metrics import accuracy_score 
accuracy_score(y_test,y_pred)


# In[29]:


plt.imshow(X_test[1])


# In[28]:


model.predict(X_test[1].reshape(1,28,28)).argmax(axis=1)


# In[30]:


plt.imshow(X_test[5])


# In[31]:


model.predict(X_test[5].reshape(1,28,28)).argmax(axis=1)

