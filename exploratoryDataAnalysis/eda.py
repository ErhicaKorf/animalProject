#%%
# Imports
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
from sklearn.linear_model import LinearRegression


# %%
# Reading in the data
animal_data = pd.read_csv('C:/Users/user-pc/Documents/Projects/Personal/animalProject/animalProject/animal-data-1.csv')  

# %%
# Plots
# get top 10 most frequent names
n = 10
l_top = animal_data['speciesname'].value_counts()[:n].index.tolist()
df_tops = pd.DataFrame(columns=animal_data.columns)
for top in l_top:
    df_top = animal_data[animal_data['speciesname']==top]
    df_tops = pd.concat([df_tops,df_top])
sns.countplot(y="speciesname", data=df_tops)

# %%
# Create the datetime to months
l_months = []
for row in range(0,len(animal_data)):
    l_months.append(datetime.strptime(re.split("( )",animal_data['intakedate'][row])[0], "%Y-%m-%d").month)
    
animal_data['month'] = l_months

# %%
# Encoding the categorical values
df_copy = animal_data.copy()
#%%
df_copy["speciesname"] = df_copy["speciesname"].astype('category')
df_copy["speciesname_cat"] = df_copy["speciesname"].cat.codes
df_copy.head()
# %%
# Create the datetime to months
l_months = []
for row in range(0,len(df_copy)):
    l_months.append(datetime.strptime(re.split("( )",df_copy['movementdate'][row])[0], "%Y-%m-%d").month)
    
df_copy['movementmonth'] = l_months
# %%
