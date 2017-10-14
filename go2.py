import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot') # Look Pretty

df1 = pd.read_csv('sa2.csv')
df1.dropna(axis = 0, how = 'any', inplace = True)
print df1.dtypes
def doKMeans(dataframe):
  df = pd.concat([dataframe.x, dataframe.y], axis = 1)
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.scatter(x = df.x, y = df.y, marker='.', alpha=0.3, s = 30)
  kmeans_model = KMeans(n_clusters = 3, init = 'random', n_init = 60, max_iter = 360, random_state = 43)
  labels = kmeans_model.fit_predict(df)
  centroids = kmeans_model.cluster_centers_
  ax.scatter(x = centroids[:,0], y = centroids[:,1], marker='x', c='red', alpha=0.7, linewidths=3, s = 120)
  print centroids
doKMeans(df1)
plt.title("disease")
plt.show()