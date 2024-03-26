#!/usr/bin/env python
# coding: utf-8

# ## Real Estate -Price Predicator

# In[1]:


import pandas as pd


# In[2]:


housing = pd.read_csv("BostonHousing.csv")


# In[3]:


housing.head()


# In[4]:


housing.info()


# In[5]:


housing['chas'].value_counts()


# In[6]:


housing.describe()


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')




# In[8]:


#For plotting histrogram
# import matplotlib.pyplot as plt
# housing.hist(bins=50,figsize=(20,15))


# ## Train-Test Spitting

# In[9]:


import numpy as np
def split_train_test(data,test_ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    print(shuffled)
    test_set_size = int(len(data)*test_ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]  # 80-20 ratio
    return data.iloc[train_indices],data.iloc[test_indices]


# In[10]:


# train_set, test_set = split_train_test(housing,0.2)


# In[11]:


# print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n")


# In[12]:


from sklearn.model_selection import train_test_split
train_set,test_set = train_test_split(housing,test_size=0.2,random_state = 42)
print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n")


# In[13]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits =1,test_size =0.2,random_state = 42)
for train_index,test_index in split.split(housing,housing['chas']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[14]:


strat_test_set['chas'].value_counts()


# In[15]:


strat_train_set['chas'].value_counts()


# In[16]:


housing =strat_train_set.copy()


# # Looking for Correlations

# In[17]:


corr_matrix = housing.corr()
corr_matrix['medv'].sort_values(ascending=False)


# In[18]:


from pandas.plotting import scatter_matrix
attributes = ["medv","rm","zn","lstat"]
scatter_matrix(housing[attributes],figsize =(12,8))


# In[19]:


housing.plot(kind="scatter",x = "rm",y="medv",alpha=0.8)


# ## Trying out attribute combination

# In[20]:


housing["taxrm"] = housing['tax']/housing['rm'] 


# In[21]:


housing.head()


# In[22]:


corr_matrix = housing.corr()
corr_matrix['medv'].sort_values(ascending=False)


# In[23]:


housing.plot(kind="scatter",x="taxrm",y="medv",alpha =0.8)


# In[24]:


housing = strat_train_set.drop("medv",axis =1)
housing_labels =strat_train_set["medv"].copy()


# ## Missing Attributes

# In[25]:


# To take care missing attributes we can use the following ideas:
#     1. Get rid of the missing data points
#     2.Get rid of the whole attribute
#     3.Set the value to some value(0.mean or median)


# In[26]:


a=housing.dropna(subset=["rm"]) #option 1
a.shape


# In[27]:


housing.drop("rm",axis=1).shape #option 2
#Note: there is no rm column now and original housing dataframe is unchanged


# In[28]:


median = housing["rm"].median()  #compute median for option 3


# In[29]:


housing["rm"].fillna(median)   #option 3
# Note:original housing dataframe is unchanged


# In[30]:


housing.shape


# In[31]:


housing.describe() #before we started imputing


# In[32]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = "median")
imputer.fit(housing)


# In[33]:


imputer.statistics_


# In[34]:


X = imputer.transform(housing)


# In[35]:


housing_tr = pd.DataFrame(X,columns = housing.columns)


# In[36]:


housing_tr.describe()


# ## Scikit-learn Design

# Primarily, three types of objects
# 1. Estimators - It estimates some parameter based on a dataset. Eg:imputer.
#                - It has a fit method and tranform method.
#                - Fit method- Fits the dataset and calcualte internal parameter.
# 
# 2. Transformers - Takes input and returns output based on learning from fit().It also has a convenience function called fit_transform() which fits and then transforms.
# 3. Predictors - LinearRegression model is an example of predictor.fit() and predict() are two common functions.It also gives score() function which will evaluate the predictions.

# ## Feature Scaling

# Primarily,two types of feature scaling methods:
# 1. Min-max scaling (Normalization):
#     (value-min)/(max-min).
#     Sklearn provides a class called MinMaxScaler for this.
# 2. Standardization:
#     (value-mean)/std.
#     Sklearn provides a class called standard scaler for this  (Better bcz if 2-3 values are wrong than also it would not effect the values).

# ## Creating a pipline

# In[37]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer',SimpleImputer(strategy = "median")),('std_scaler',StandardScaler(),)  # we can add more values
])


# In[38]:


housing_num_tr = my_pipeline.fit_transform(housing)


# In[39]:


housing_num_tr.shape  #this is a numpy array


# ## Selecting a desired model for Real Estates

# In[40]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor 

# model = LinearRegression()
# model = DecisionTreeRegressor()
model = RandomForestRegressor()
model.fit(housing_num_tr, housing_labels)

#We are not taking taxrm attribute


# In[41]:


some_data = housing.iloc[:5]


# In[42]:


some_labels = housing_labels.iloc[:5]


# In[43]:


prepared_data = my_pipeline.transform(some_data)


# In[44]:


model.predict(prepared_data)


# In[45]:


list(some_labels)


# ## Evaluating the model

# In[46]:


from sklearn.metrics import mean_squared_error
housing_predications = model.predict(housing_num_tr)
lin_mse = mean_squared_error(housing_labels, housing_predications)
lin_rmse = np.sqrt(lin_mse)


# In[47]:


lin_mse     ##overfitting may happen


# In[48]:


lin_rmse


# ## Using better evaluation technique - Cross Validation

# In[49]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model,housing_num_tr,housing_labels,scoring = "neg_mean_squared_error",cv = 10)
rmse_scores = np.sqrt(-scores)


# In[50]:


rmse_scores


# In[51]:


def print_scores(scores):
    print("Scores:" ,scores)
    print("Mean: ",scores.mean())
    print("Standered devaition: ", scores.std())


# In[52]:


print_scores(rmse_scores)


# In[ ]:




