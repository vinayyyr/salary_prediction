#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set_theme(style="ticks", color_codes=True)
df = pd.read_csv("/Users/vinayraj/Downloads/survey_results_public.csv")
df.head()


# In[2]:


df.info()


# In[3]:


df.columns


# In[7]:


df = df[["Country", "EdLevel", "YearsCodePro", "Employment", "ConvertedCompYearly"]]
df = df.rename({"ConvertedCompYearly": "Salary"}, axis=1)
df.head()


# In[8]:


df = df[df["Salary"].notnull()]
df.head()


# In[9]:


df.info()


# In[10]:


df = df.dropna()
df.isnull().sum()


# In[11]:


df = df[df["Employment"] == "Employed full-time"]
df = df.drop("Employment", axis=1)
df.info()


# In[12]:


df['Country'].value_counts()


# In[13]:


def short_cat(categories, cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = 'Other'
    return categorical_map


# In[14]:


country_map = short_cat(df.Country.value_counts(), 400)
df['Country'] = df['Country'].map(country_map)
df.Country.value_counts()


# In[15]:


fig, ax = plt.subplots(1,1, figsize=(12, 7))
df.boxplot('Salary', 'Country', ax=ax)
plt.suptitle('Salary (US$) v Country')
plt.title('')
plt.ylabel('Salary')
plt.xticks(rotation=90)
plt.show()


# In[16]:


df = df[df["Salary"] <= 150000]
df = df[df["Salary"] >= 10000]
df = df[df['Country'] != 'Other']


# In[17]:


fig, ax = plt.subplots(1,1, figsize=(12, 7))
df.boxplot('Salary', 'Country', ax=ax)
plt.suptitle('Salary (US$) v Country')
plt.title('')
plt.ylabel('Salary')
plt.xticks(rotation=90)
plt.show()


# In[18]:


df["YearsCodePro"].unique()


# In[19]:


def clean_exp(x):
    if x ==  'More than 50 years':
        return 50
    if x == 'Less than 1 year':
        return 0.5
    return float(x)

df['YearsCodePro'] = df['YearsCodePro'].apply(clean_exp)


# In[20]:


df["EdLevel"].unique()


# In[21]:


def clean_ed(x):
    if 'Bachelor’s degree' in x:
        return 'Undergrad degree'
    if 'Master’s degree' in x:
        return 'Master’s degree'
    if 'Professional degree' in x or 'Other doctoral' in x:
        return 'Post grad'
    return 'Less than a Bachelors'

df['EdLevel'] = df['EdLevel'].apply(clean_ed)


# In[22]:


df["EdLevel"].unique()


# In[23]:


from sklearn.preprocessing import LabelEncoder
le_education = LabelEncoder()
df['EdLevel'] = le_education.fit_transform(df['EdLevel'])
df["EdLevel"].unique()
#le.classes_


# In[24]:


le_country = LabelEncoder()
df['Country'] = le_country.fit_transform(df['Country'])
df["Country"].unique()


# In[25]:


X = df.drop("Salary", axis=1)
y = df["Salary"]


# In[26]:


from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(X, y.values)


# In[27]:


y_pred = linear_reg.predict(X)


# In[28]:


from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
error = np.sqrt(mean_squared_error(y, y_pred))


# In[29]:


error


# In[30]:


from sklearn.tree import DecisionTreeRegressor
dec_tree_reg = DecisionTreeRegressor(random_state=0)
dec_tree_reg.fit(X, y.values)


# In[31]:


y_pred = dec_tree_reg.predict(X)


# In[32]:


error = np.sqrt(mean_squared_error(y, y_pred))
print("${:,.02f}".format(error))


# In[33]:


from sklearn.ensemble import RandomForestRegressor
random_forest_reg = RandomForestRegressor(random_state=0)
random_forest_reg.fit(X, y.values)


# In[34]:


y_pred = random_forest_reg.predict(X)


# In[35]:


error = np.sqrt(mean_squared_error(y, y_pred))
print("${:,.02f}".format(error))


# In[36]:


from sklearn.model_selection import GridSearchCV

max_depth = [None, 2,4,6,8,10,12]
parameters = {"max_depth": max_depth}

regressor = DecisionTreeRegressor(random_state=0)
gs = GridSearchCV(regressor, parameters, scoring='neg_mean_squared_error')
gs.fit(X, y.values)


# In[37]:


plt.figure(figsize=(10,6))
sns.heatmap(df.corr(),yticklabels=False,annot=True, cmap='GnBu_r')


# In[38]:


regressor = gs.best_estimator_

regressor.fit(X, y.values)
y_pred = regressor.predict(X)
error = np.sqrt(mean_squared_error(y, y_pred))
print("${:,.02f}".format(error))


# In[53]:


X




# In[45]:


X = np.array([["United States of America", 'Master’s degree', 15 ]])


# In[46]:


X


# In[47]:


X[:, 0] = le_country.transform(X[:,0])
X[:, 1] = le_education.transform(X[:,1])
X = X.astype(float)
X


# In[48]:


y_pred = regressor.predict(X)
y_pred


# In[49]:


import pickle


# In[50]:


data = {"model": regressor, "le_country": le_country, "le_education": le_education}
with open('saved_steps.pkl', 'wb') as file:
    pickle.dump(data, file)


# In[51]:


with open('saved_steps.pkl', 'rb') as file:
    data = pickle.load(file)

regressor_loaded = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]


# In[54]:


y_pred = regressor_loaded.predict(X)
y_pred


# In[ ]:




