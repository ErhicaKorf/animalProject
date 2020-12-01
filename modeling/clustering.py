#%%
from sklearn.cluster import KMeans
from numpy import unique
from numpy import where
from matplotlib import pyplot

#%%
from sklearn.datasets import make_classification
X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)



#%%
# define the model
model = KMeans(n_clusters=5)
# fit the model
model.fit(features)
# assign a cluster to each example
yhat = model.predict(features)

# retrieve unique clusters
clusters = unique(yhat)

#%%
# create scatter plot for samples from each cluster
for cluster in clusters:
	# get row indexes for samples with this cluster
	row_ix = where(yhat == cluster)
	# create scatter of these samples
	pyplot.scatter(features[row_ix, 0], features[row_ix, 1])
# %%
