import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib

df = pd.read_csv('sa_geo.csv')
print df
x = df.x
y = df.y
df = df[['x','y']]
matplotlib.style.use('ggplot')

df.plot.scatter(x='x',y='y')
plt.show()

def do_kmeans(df):
    from sklearn.cluster import KMeans
    fig = plt.figure()
    kmeans = KMeans(n_clusters = 3)
    kmeans.fit(df)
    labels = kmeans.predict(df)    
    ax = fig.add_subplots(111)
    centeroids = kmeans.predict(df)
    ax.scatter(df.x,df.y,marker = '.', alpha = 0.3)
do_kmeans(df)
plt.show()    