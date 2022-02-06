# LAPTOP PRICE

from random import random
import numpy as np
import pandas as pd
from seaborn.palettes import color_palette
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
import matplotlib.pyplot as plt
import seaborn as sn

import os
path = "C:/Users/HP/OneDrive/Documents/Python Anaconda/Streamlit_Laptop_App"
os.chdir(path)
os.listdir()

df = pd.read_csv("laptop_data.csv"); df

# NAs
df.isnull().sum()
df[df['Price'].isna()]

# Duplicated
df.duplicated().sum()
df[df['Price'].duplicated()].sort_values('Price', ascending=False)

# Categorical and Numerical features # IMPORTANT
catvars = df.select_dtypes(include='object').columns; catvars
numvars = df.select_dtypes(include=['int32', 'int64', 'float32', 'float64']); numvars

def valuecounts(col):
    print(f'Valuecounts of the particular col {col} is: \n {df[col].value_counts()}')

for col in df.columns:
    valuecounts(col)
    print("-"*50)

# Visualisation price
sn.distplot(df['Price'], color='red');

# Visualisation count plots for the categorical variables # IMPORTANT
def drawplot(col):
    plt.figure(figsize=(15,7))
    sn.countplot(df[col], palette='plasma')
    plt.xticks(rotation='vertical')

toview = ['Company', 'TypeName', 'Ram', 'OpSys']
for col in toview:
    drawplot(col)

# Visualisation average price
plt.figure(figsize=(15,7))
sn.barplot(x=df['Company'], y=df['Price'])
plt.xticks(rotation = 'vertical');

# Visualisation various types of laptops
sn.countplot(df['TypeName'], palette='autumn')
plt.xticks(rotation='vertical');

# Visualisation type and variation in price
sn.barplot(x=df['TypeName'], y=df['Price'])
plt.xticks(rotation = 'vertical');

# Visualisation screen towards price
sn.scatterplot(x=df['Inches'], y=df['Price']);

### UNNAMED:0
df.drop(columns=['Unnamed: 0'], inplace=True); df.columns

### CZK
df['Price'] = df['Price']*0.3; df

### SCREEN RESOLUTION
df['ScreenResolution'].value_counts()

# Touch screen
df['TouchScreen'] = df['ScreenResolution'].apply(lambda element:1 if 'Touchscreen' in element else 0); df
# df['TouchScreen'] = np.where(df['ScreenResolution'].astype(str).str.contains('Touchscreen', na=False), 1, 0)
sn.countplot(df['TouchScreen'], palette='plasma');

# Mean comparison of laptops without and with touch screen
sn.barplot(df['TouchScreen'], df['Price'])
df[df['TouchScreen']==0]['Price'].mean()
df[df['TouchScreen']==1]['Price'].mean()

# IPS is have
df['IPS'] = df['ScreenResolution'].apply(lambda element: 1 if "IPS" in element else 0); df

# Mean comparison of laptops without and with IPS
sn.barplot(df['IPS'], df['Price'])
df[df['IPS']==0]['Price'].mean()
df[df['IPS']==1]['Price'].mean()

# Resolution EXTRACTING X resolution and Y resolution
splitdf = df['ScreenResolution'].str.split('x', n=1, expand=True); splitdf

df['X_res'] = splitdf[0]
df['Y_res'] = splitdf[1]

df['X_res'] = df['X_res'].str.replace(',','').str.findall(r'(\d+\.?\d+)').apply(lambda x:x[0])

df['X_res'] = df['X_res'].astype('int')
df['Y_res'] = df['Y_res'].astype('int')

# Correlation
plt.figure(figsize=(12, 8))
sn.heatmap(df.corr(), annot=True, fmt='.1g', center=0, linewidths=1, linecolor='black', cmap='plasma');

df.corr()['Price']

# ppi
df['PPI'] = (((df['X_res']**2+df['Y_res']**2))**0.5/df['Inches']).astype('float')
df.drop(columns=['ScreenResolution', 'Inches', 'X_res', 'Y_res'], inplace=True)

### WEIGHT
df['Weight'].value_counts()

df['Weight'] = df['Weight'].str.replace('kg', '')
# Converting from string -> float for the weight column
df['Weight'] = df['Weight'].astype('float32')

### CPU
df['Cpu'].value_counts()

df['CPU_name'] = df['Cpu'].apply(lambda text: " ".join(text.split()[:3]))
df.drop(columns=['Cpu'], inplace=True)

def processortype(text):
    if text == 'Intel Core i7' or text == 'Intel Core i5' or text == 'Intel Core i3':
        return text
    else:
        if text.split()[0] == 'Intel':
            return 'Other Intel Processor'
        else:
            return 'AMD Processor'

df['CPU_name'] = df['CPU_name'].apply(lambda text: processortype(text))

sn.countplot(df['CPU_name'], palette='ocean') # IMPORTANT
plt.xticks(rotation = 'vertical');

# Visualisation processor towards price
sn.barplot(df['CPU_name'], df['Price'])
plt.xticks(rotation = 'vertical'); # HONEYWELL

### RAM
df['Ram'].value_counts()

df['Ram'] = df['Ram'].str.replace('GB', "")
# Converting from string -> integer for ram column
df['Ram'] = df['Ram'].astype('int32')

sn.countplot(df['Ram'], palette='ocean')

# Visualisation ram towards price
sn.barplot(df['Ram'], df['Price'])

### MEMORY
df['Memory'].value_counts()

# Removing .0
df['Memory'] = df['Memory'].astype(str).replace('\.0', '', regex=True) # HONEYWELL
# Replace GB with ''
df['Memory'] = df['Memory'].str.replace('GB', '')
# Replace the TB with 000
df['Memory'] = df['Memory'].str.replace('TB', '000')
# Split the word accross the "+" character
newdf = df['Memory'].str.split("+", n=1, expand=True); newdf

# first
df['first'] = newdf[0]
df['first'] = df['first'].str.strip()

def applychanges(value):
    df['Layer1'+value] = df['first'].apply(lambda x:1 if value in x else 0)
     
listtoapply = ['HDD','SSD','Hybrid','FlashStorage']    
for value in listtoapply:
    applychanges(value)
# Remove all the characters just keep the numbers
df['first'] = df['first'].str.replace(r'\D', '')
df['first'].value_counts()

# second
df['second'] = newdf[1]
df['second'] = df['second'].str.strip()

def applychanges(value):
    df['Layer2'+value] = df['second'].apply(lambda x:1 if value in x else 0)
     
listtoapply = ['HDD','SSD','Hybrid','FlashStorage'] 
df['second'] = df['second'].fillna("0")
for value in listtoapply:
    applychanges(value)
# Remove all the characters just keep the numbers
df['second'] = df['second'].str.replace(r'\D', '')
df['second'].value_counts()

# to intigers
df['first'] = df['first'].astype('int')
df['second'] = df['second'].astype('int')

# multiplying the elements and storing the result in subsequent columns
df["HDD"]=(df["first"]*df["Layer1HDD"]+df["second"]*df["Layer2HDD"])
df["SSD"]=(df["first"]*df["Layer1SSD"]+df["second"]*df["Layer2SSD"])
df["Hybrid"]=(df["first"]*df["Layer1Hybrid"]+df["second"]*df["Layer2Hybrid"])
df["Flash_Storage"]=(df["first"]*df["Layer1FlashStorage"]+df["second"]*df["Layer2FlashStorage"])

# dropping of uncessary columns
df.drop(columns=['first', 'second', 'Layer1HDD', 'Layer1SSD', 'Layer1Hybrid',
       'Layer1FlashStorage', 'Layer2HDD', 'Layer2SSD', 'Layer2Hybrid',
       'Layer2FlashStorage'],inplace=True)

df.drop(columns=['Memory'], inplace=True)
df.drop(columns=['Hybrid', 'Flash_Storage'], inplace=True) # not important
df.corr()['Price']

df.columns

# GPU
df['Gpu'].value_counts()

df['Gpu brand'] = df['Gpu'].apply(lambda x : x.split()[0])
df.drop(columns=['Gpu'], axis=1, inplace=True)

sn.countplot(df['Gpu brand'], palette='plasma');

df = df[df['Gpu brand']!='ARM']
sn.countplot(df['Gpu brand'], palette='plasma');

sn.barplot(df['Gpu brand'], df['Price'], estimator=np.median) # mean

# OPERATION SYSTEM
df['OpSys'].value_counts()

sn.barplot(df['OpSys'], df['Price'])
plt.xticks(rotation='vertical')
plt.show()

df['OpSys'].unique().reshape(-1,1)

def setcategory(x):
    if x == "Windows 10" or x == "Windows 10 S" or x == "Windows 7":
        return 'Windows'
    elif x == 'Mac OS X' or x == 'macOS':
        return 'Mac'
    else:
        return 'Other'

df['OpSys'] = df['OpSys'].apply(lambda x: setcategory(x))

sn.countplot(df['OpSys'], palette='plasma');

sn.barplot(x=df['OpSys'], y=df['Price'])
plt.xticks(rotation='vertical');

# WEIGHT
sn.distplot(df['Weight']);

sn.scatterplot(df['Weight'], df['Price']);

# PRICE
sn.distplot(df['Price']); # left skewed

sn.displot(np.log(df['Price'])) # almost gaussian distribution

df.corr()['Price']
plt.figure(figsize=(12, 8))
sn.heatmap(df.corr(), annot=True, fmt='.1g', center=0, linewidths=1, linecolor='black', cmap='plasma');

# ---------------------------------------------------------------------------------------------------------------
# Model ---------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd

import os
path = "C:/Users/HP/OneDrive/Documents/Python Anaconda/Streamlit_Laptop_App"
os.chdir(path)
os.listdir()

df = pd.read_csv('df.csv'); df

X = df.drop(columns=['Price'], axis=1); X
y = np.log(df['Price']); y

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=2)
X_train.shape,X_test.shape

# --- do not use
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn import tree
# --- do not use

# LINEAR REGRESSION
X_train.dtypes
# we will apply one hot encoding on the columns with this indices [0,1,3,8,11]
# the remainder we keep as passthrough i.e no other col must get effected 
# except the ones undergoing the transformation!

# CTRL + SHIFT + Z
# CTRL + /

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
step1 = ColumnTransformer(transformers=[
    ('col_tnf', OneHotEncoder(sparse=False, drop="first"), [0,1,3,8,11])],
    remainder='passthrough')

from sklearn.linear_model import LinearRegression
step2 = LinearRegression()

from sklearn.pipeline import Pipeline
pipe = Pipeline([('step1', step1),
                 ('step2', step2)])

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test); y_pred

from sklearn import metrics
print('R2 score', metrics.r2_score(y_test, y_pred))
print('MAE:', metrics.mean_absolute_error(y_test, y_pred), '  Original MAE:', np.exp(metrics.mean_absolute_error(y_test, y_pred)))

# RIDGE REGRESSION
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
step1 = ColumnTransformer(transformers=[
    ('col_tnf', OneHotEncoder(sparse=False, drop="first"), [0,1,3,8,11])],
    remainder='passthrough')

from sklearn.linear_model import Ridge
step2 = Ridge(alpha=10)

from sklearn.pipeline import Pipeline
pipe = Pipeline([('step1', step1),
                 ('step2', step2)])

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test); y_pred

from sklearn import metrics
print('R2 score', metrics.r2_score(y_test, y_pred))
print('MAE:', metrics.mean_absolute_error(y_test, y_pred), '  Original MAE:', np.exp(metrics.mean_absolute_error(y_test, y_pred)))

# LASSO REGRESSION
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
step1 = ColumnTransformer(transformers=[
    ('col_tnf', OneHotEncoder(sparse=False, drop="first"), [0,1,3,8,11])],
    remainder='passthrough')

from sklearn.linear_model import Lasso
step2 = Lasso(alpha=0.001)

from sklearn.pipeline import Pipeline
pipe = Pipeline([('step1', step1),
                 ('step2', step2)])

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test); y_pred

from sklearn import metrics
print('R2 score', metrics.r2_score(y_test, y_pred))
print('MAE:', metrics.mean_absolute_error(y_test, y_pred), '  Original MAE:', np.exp(metrics.mean_absolute_error(y_test, y_pred)))

# DECISION TREE
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
step1 = ColumnTransformer(transformers=[
    ('col_tnf', OneHotEncoder(sparse=False, drop="first"), [0,1,3,8,11])],
    remainder='passthrough')

from sklearn.tree import DecisionTreeRegressor
step2 = DecisionTreeRegressor(max_depth = 8)

from sklearn.pipeline import Pipeline
pipe = Pipeline([('step1', step1),
                 ('step2', step2)])

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test); y_pred

from sklearn import metrics
print('R2 score', metrics.r2_score(y_test, y_pred))
print('MAE:', metrics.mean_absolute_error(y_test, y_pred), '  Original MAE:', np.exp(metrics.mean_absolute_error(y_test, y_pred)))

# RANDOM FOREST
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
step1 = ColumnTransformer(transformers=[
    ('col_tnf', OneHotEncoder(sparse=False, drop="first"), [0,1,3,8,11])],
    remainder='passthrough')

from sklearn.ensemble import RandomForestRegressor
step2 = RandomForestRegressor(n_estimators = 512, # every tree is a sample out of dataset
                              random_state = 3,
                              max_samples = 0.5,
                              max_features = 0.75,
                              max_depth = 15)

from sklearn.pipeline import Pipeline
pipe = Pipeline([('step1', step1),
                 ('step2', step2)])

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test); y_pred

from sklearn import metrics
print('R2 score', metrics.r2_score(y_test, y_pred))
print('MAE:', metrics.mean_absolute_error(y_test, y_pred), '  Original MAE:', np.exp(metrics.mean_absolute_error(y_test, y_pred)))

# Save df
df.to_csv('df.csv',index=None)

# Save model
import pickle
data = {"model" : pipe}
with open('model.pkl', 'wb') as file:
    pickle.dump(data, file)

# ---------------------------------------------------------------------------------------------------------------
# Tunning ---------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd

import os
path = "C:/Users/HP/OneDrive/Documents/Python Anaconda/Streamlit_Laptop_App"
os.chdir(path)
os.listdir()

df = pd.read_csv('df.csv'); df

X = df.drop(columns=['Price'], axis=1); X
y = np.log(df['Price']); y

# Load model
import pickle
with open('model.pkl', 'rb') as file:
    data = pickle.load(file)
model = data["model"]; model

# Comparison
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=2); X_train.shape,X_test.shape
y_pred = model.predict(X_test)
from sklearn import metrics
print(metrics.r2_score(y_test, y_pred))

# TUNNING starting 
mapper = {i:value for i, value in enumerate(X.columns)}; mapper

indexlist = [0,1,3,8,11]
transformlist = []
for key,value in mapper.items():
    if key in indexlist:
        transformlist.append(value)
        
transformlist

X = pd.get_dummies(X, columns=transformlist, drop_first=True); X.shape

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=2)
X_train.shape,X_test.shape

# Ccp alpha
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn import tree

reg = DecisionTreeRegressor(random_state=0)
reg.fit(X_train,y_train)
plt.figure(figsize=(16,9))
tree.plot_tree(reg,filled=True,feature_names=X.columns)

path = reg.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas; ccp_alphas

alphalist = []
for alpha in ccp_alphas:
    reg = DecisionTreeRegressor(random_state=0,ccp_alpha=alpha)
    reg.fit(X_train,y_train)
    alphalist.append(reg)

train_score = [reg.score(X_train, y_train) for reg in alphalist]
test_score = [reg.score(X_test, y_test) for reg in alphalist]

plt.figure(figsize=(16,9))
plt.plot(ccp_alphas, train_score, marker = 'o',
         label = 'training', color = 'blue')
plt.plot(ccp_alphas, test_score, marker = '+',
         label = 'testing', color = 'red', drawstyle = 'steps-post')
plt.xlabel('ccp alpha')
plt.ylabel('Accuracy')
plt.legend()
plt.show() # [0.0025 --> 0.0075]

reg = DecisionTreeRegressor(random_state=0, ccp_alpha=0.0085)
reg.fit(X_train, y_train)
plt.figure(figsize=(16,9))
tree.plot_tree(reg, filled=True, feature_names=X.columns)

from sklearn.ensemble import RandomForestRegressor
params =  { 
    'RandomForest':{ # item
        'model' : RandomForestRegressor(),
        'params':{
            'n_estimators':[int(x) for x in np.linspace(100,1200,10)],
            'criterion':["mse", "mae"],
            'max_depth':[int(x) for x in np.linspace(1,30,5)],
            'max_features':['auto','sqrt','log2'],
            'ccp_alpha':[x for x in np.linspace(0.0025,0.0125,5)], # based on viz
            'min_samples_split':[2,5,10,14],
            'min_samples_leaf':[2,5,10,14],
        }
    },
    'Decision Tree':{ # item
        'model':DecisionTreeRegressor(),
        'params':{
            'criterion':["mse", "mae"],
            'max_depth':[int(x) for x in np.linspace(1,30,5)],
            'max_features':['auto','sqrt','log2'],
            'ccp_alpha':[x for x in np.linspace(0.0025,0.0125,5)], # based on viz
            'min_samples_split':[2,5,10,14],
            'min_samples_leaf':[2,5,10,14],
        }
    }
}

from sklearn.model_selection import RandomizedSearchCV
scores = []
for modelname, mp in params.items():
    clf = RandomizedSearchCV(mp['model'],
                             param_distributions=mp['params'], cv = 5,
                             n_iter = 10, scoring = 'neg_mean_squared_error', verbose=2)
    clf.fit(X_train, y_train)
    scores.append({
        'model_name':modelname,
        'best_score':clf.best_score_,
        'best_estimator':clf.best_estimator_
    })

print(scores[1]) # DT
print(scores[0]) # RF

model = RandomForestRegressor(
                           criterion='mae',
                           max_depth=15,
                           max_features='log2',
                           ccp_alpha=0.0025,
                           min_samples_split=14, 
                           min_samples_leaf=2,
                           n_estimators=1077)

model.fit(X_train,y_train)
y_pred = model.predict(X_test)

from sklearn import metrics
print(metrics.r2_score(y_test, y_pred))

# ---------------------------------------------------------------------------------------------------------------
# Prediction on the whole dataset -------------------------------------------------------------------------------

import numpy as np
import pandas as pd

import os
path = "C:/Users/HP/OneDrive/Documents/Python Anaconda/Streamlit_Laptop_App"
os.chdir(path)
os.listdir()

df = pd.read_csv('df.csv'); df

X = df.drop(columns=['Price'], axis=1); X
y = np.log(df['Price']); y

# Load model
import pickle
with open('model.pkl', 'rb') as file:
    data = pickle.load(file)
model = data["model"]; model

predicted = []
testtrain = np.array(X)
for i in range(len(testtrain)):
    predicted.append(model.predict([testtrain[i]]))

predicted

ans = [np.exp(predicted[i][0]) for i in range(len(predicted))]

df['Predicted Price'] = np.array(ans); df

import seaborn as sn
import matplotlib.pyplot as plt
sn.distplot(df['Price'], hist = False, color = 'red', label = 'Actual')
sn.distplot(df['Predicted Price'], hist = False, color = 'blue', label = 'Predicted')
plt.legend()
plt.show()

# ---------------------------------------------------------------------------------------------------------------
# Tunning II. ---------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd

import os
path = "C:/Users/HP/OneDrive/Documents/Python Anaconda/Streamlit_Laptop_App"
os.chdir(path)
os.listdir()

df = pd.read_csv('df.csv'); df

# Get all numeric data
df_numeric = df._get_numeric_data(); df_numeric

# Isolate the categorical variables
df_categorical = df.select_dtypes(exclude="number"); df_categorical

# Transform into dummies
df_categorical = pd.get_dummies(df_categorical, drop_first=True); df_categorical

# Joining numerical and categorical datasets
df_final = pd.concat([df_numeric, df_categorical], axis=1); df_final

X = df_final.drop(columns=['Price'], axis=1); X
y = np.log(df_final['Price']); y

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=2)
X_train.shape,X_test.shape

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators = 100, # every tree is a sample out of dataset
                              random_state = 3,
                              max_samples = 0.5,
                              max_features = 0.75,
                              max_depth = 15)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn import metrics
print('R2 score', metrics.r2_score(y_test, y_pred))
print('MAE:', metrics.mean_absolute_error(y_test, y_pred), '  Original MAE:', np.exp(metrics.mean_absolute_error(y_test, y_pred)))

predicted = []
testtrain = np.array(X)
for i in range(len(testtrain)):
    predicted.append(model.predict([testtrain[i]]))

predicted

ans = [np.exp(predicted[i][0]) for i in range(len(predicted))]

df['Predicted Price'] = np.array(ans); df

import seaborn as sn
import matplotlib.pyplot as plt
sn.distplot(df['Price'], hist = False, color = 'red', label = 'Actual')
sn.distplot(df['Predicted Price'], hist = False, color = 'blue', label = 'Predicted')
plt.legend()
plt.show()
