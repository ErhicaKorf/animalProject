#%%
from sklearn import linear_model



X = df_copy[['breedname_cat', 'intakereason_cat','animal_cat','animalage_cat','speciesname_cat']]
y = df_copy['movementmonth']

regr = linear_model.LinearRegression()
regr.fit(X, y)

#%%
regr.predict([[284, 1853,260,1,13]])

#%%
model = LinearRegression()
model.fit(np.array(df_copy['breedname_cat']).reshape(-1,1), df_copy['movementmonth'])

# %%
r_sq = model.score(np.array(df_copy['breedname_cat']).reshape(-1,1), df_copy['movementmonth'])
print('coefficient of determination:', r_sq)






# %%
# Random forests
# Labels are the values we want to predict
labels = np.array(df_copy['movementmonth'])

# Convert to numpy array
features = np.array(df_copy[['breedname_cat', 'intakereason_cat','animal_cat','animalage_cat','speciesname_cat','month']])
feature_list = list(['breedname_cat', 'intakereason_cat','animal_cat','animalage_cat','speciesname_cat'])
# %%
# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)
# %%
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)
# %%
# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 50, random_state = 42)
# Train the model on training data
rf.fit(train_features, train_labels)

#%%
# Use the forest's predict method on the test data
predictions = rf.predict(test_features)
# Calculate the absolute errors
errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')# %%

# %%
# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')














#%%
###### Feature importance #######

importance = rf.feature_importances_
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()# %%

# %%
# Convert to numpy array
features = np.array(df_copy[['animal_cat','animalage_cat','month']])
feature_list = list(['animal_cat','animalage_cat','month'])
# %%
# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)
# %%
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)
# %%
# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 500, random_state = 42)
# Train the model on training data
rf.fit(train_features, train_labels)

#%%
# Use the forest's predict method on the test data
predictions = rf.predict(test_features)
# Calculate the absolute errors
errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')# %%

# %%
# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')
# %%
