import matplotlib as mpl
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np 
import time
C = 1
kernel = 'linear'
iterations = 5000  

FAST_DRAW = True


def drawPlots(model, wintitle='Figure 1'):
  mpl.style.use('ggplot') # Look Pretty

  padding = 3
  resolution = 0.5
  max_2d_score = 0

  y_colors = ['green', 'red', 'blue', 'yellow', 'white', 'black', '#000ff0']
  my_cmap = mpl.colors.ListedColormap(['green', 'red', 'blue', 'yellow', 'white', 'black', '#000ff0'])
  colors = [y_colors[i] for i in y_train]
  num_columns = len(X_train.columns)

  fig = plt.figure()
  fig.canvas.set_window_title(wintitle)
  
  cnt = 0
  for col in range(num_columns):
    for row in range(num_columns):
      # Easy out
      if FAST_DRAW and col > row:
        cnt += 1
        continue

      ax = plt.subplot(num_columns, num_columns, cnt + 1)
      plt.xticks(())
      plt.yticks(())

      # Intersection:
      if col == row:
        plt.text(0.5, 0.5, X_train.columns[row], verticalalignment='center', horizontalalignment='center', fontsize=12)
        cnt += 1
        continue


      # Only select two features to display, then train the model
      X_train_bag = X_train.ix[:, [row,col]]
      X_test_bag = X_test.ix[:, [row,col]]
      model.fit(X_train_bag, y_train)

      # Create a mesh to plot in
      x_min, x_max = X_train_bag.ix[:, 0].min() - padding, X_train_bag.ix[:, 0].max() + padding
      y_min, y_max = X_train_bag.ix[:, 1].min() - padding, X_train_bag.ix[:, 1].max() + padding
      xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                           np.arange(y_min, y_max, resolution))

      # Plot Boundaries
      plt.xlim(xx.min(), xx.max())
      plt.ylim(yy.min(), yy.max())

      # Prepare the contour
      Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
      Z = Z.reshape(xx.shape)
      plt.contourf(xx, yy, Z, cmap=my_cmap, alpha=0.8)
      plt.scatter(X_train_bag.ix[:, 0], X_train_bag.ix[:, 1], c=colors, alpha=0.5)


      score = round(model.score(X_test_bag, y_test) * 100, 3)
      plt.text(0.5, 0, "Score: {0}".format(score), transform = ax.transAxes, horizontalalignment='center', fontsize=8)
      max_2d_score = score if score > max_2d_score else max_2d_score

      cnt += 1

  print "Max 2D Score: ", max_2d_score
  fig.set_tight_layout(True)


def benchmark(model, wintitle='Figure 1'):
  print '\n\n' + wintitle + ' Results'
  s = time.time()
  for i in range(iterations):    
    a = model.fit(X_train, y_train)
  print "{0} Iterations Training Time: ".format(iterations), time.time() - s


  s = time.time()
  for i in range(iterations):
    score = a.score(X_test, y_test)
  print "{0} Iterations Scoring Time: ".format(iterations), time.time() - s
  print "High-Dimensionality Score: ", round((score*100), 3)
X = pd.read_csv('sa.csv')
print X
y = X.Disease
X.drop('Disease', axis = 1, inplace = True)
print X.head(6)
y = y.map({'Asthma': 0, 'nostelgia': 1, 'typhoid': 2, 'TuberCulosis': 3, 'Viral':4,'maleria':5,'Joindis':6})
print y
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 7)
from sklearn.svm import SVC
svc = SVC(C = C, kernel = kernel)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)

benchmark(knn, 'KNeighbors')
drawPlots(knn, 'KNeighbors')

benchmark(svc, 'SVC')
drawPlots(svc, 'SVC')
df = pd.read_csv('sa_test.csv')

print df
df.drop('Disease',axis=1,inplace = True)
print df[df.index==0]
df_test = df.ix[:, [1,2]]
print svc.predict(df_test)

plt.show()
