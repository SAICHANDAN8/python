#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[4]:


df = pd.read_csv('Morgan.csv')
df


# In[5]:


df.head(5)


# In[6]:


df.tail(5)


# In[7]:


df.dropna


# In[8]:


df.info()


# In[9]:


df.fillna(0)


# In[10]:


df.shape


# In[11]:


X = df['High']
Y = df['Low']
print(X)
print(Y)


# In[30]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression


# In[31]:


X_train = np.array(['X']).reshape(-1,1)
Y_train = np.array(['Y']).reshape(-1,1)
print(X_train.shape)
print(Y_train.shape)


# In[32]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)


# In[33]:


X_train.shape


# In[34]:


X_test.shape


# In[35]:


print(X.shape)


# # Linear Regression

# In[39]:


import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

model = LinearRegression()
X_train_reshaped = np.array(X_train).reshape(-1, 1)
Y_train_reshaped = np.array(Y_train).reshape(-1, 1)
model.fit(X_train_reshaped, Y_train_reshaped)

X_test_reshaped = np.array(X_test).reshape(-1, 1)
y_pred = model.predict(X_test_reshaped)
mse = mean_squared_error(Y_test, y_pred)
print("Mean Squared Error:", mse)


# In[46]:


r2 = r2_score(y_test, y_pred)
print("R-squared score:", r2)


# In[47]:


import matplotlib.pyplot as plt
plt.scatter(X_test, y_pred, color='red', label='Predicted')
plt.plot(X_test, y_pred, color='green', label='Regression Line')
plt.ylabel('y')
plt.title('Linear Regression')
plt.show()


# # Logistic Regression

# In[53]:





# In[ ]:





# In[ ]:




