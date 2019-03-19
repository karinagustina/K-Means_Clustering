#K-Means Clustering using Sklearn Datasets_Iris

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#================================================
#Load Data
#================================================

from sklearn.datasets import load_iris
iris = load_iris()
print(dir(iris))

#================================================
#Create DataFrame
#================================================

dfIris = pd.DataFrame(
    iris['data'],
    columns = ['sepal_length','sepal_width','petal_length','petal_width']
)
dfIris['target'] = iris['target']
dfIris['jenis'] = dfIris['target'].apply(
    lambda x: iris['target_names'][x]
)
print(dfIris.head())

#================================================
#Split Datasets
#================================================

dfSetosa = dfIris[dfIris['jenis'] == 'setosa']
dfVersicolor = dfIris[dfIris['jenis'] == 'versicolor']
dfVirginica = dfIris[dfIris['jenis'] == 'virginica']

print(dfSetosa)
print(dfVersicolor)
print(dfVirginica)

#================================================
#K-Means Clustering
#================================================

from sklearn.cluster import KMeans
model = KMeans(n_clusters = 3, random_state = 0)

#Training
model.fit(dfIris[['petal_length', 'petal_width']])

#Prediction
prediction = model.predict(dfIris[['petal_length', 'petal_width']])
print(prediction)
dfIris['prediction'] = prediction
print(dfIris)

#Split dataset: dfSetosaP, dfVersicolorP, dfVirginicaP
dfSetosaPredict = dfIris[dfIris['prediction'] == 0]
dfVersicolorPredict = dfIris[dfIris['prediction'] == 2]
dfVirginicaPredict = dfIris[dfIris['prediction'] == 1]

#Centroids
centroids = model.cluster_centers_
print(centroids)

#================================================
#Plot Original Data vs K-Means Prediction
#================================================

plt.figure('KMeans_Iris', figsize = (13, 6))

#Plot Petal Length vs Petal Width (Original)
plt.subplot(121)
plt.scatter(
    dfSetosa['petal_length'],
    dfSetosa['petal_width'],
    color = 'r'
)
plt.scatter(
    dfVersicolor['petal_length'],
    dfVersicolor['petal_width'],
    color = 'lightgreen'
)
plt.scatter(
    dfVirginica['petal_length'],
    dfVirginica['petal_width'],
    color = 'b'
)

#Plot Centroids (Original)
plt.scatter(
    centroids[:,0],
    centroids[:,1],
    marker = '*',
    color = 'y',
    s = 300
)

plt.legend(['Setosa', 'Versicolor', 'Virginica', 'Centroids'])
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('Original Data')
plt.grid(True)

#Plot Petal Length vs Petal Width (Prediction)
plt.subplot(122)
plt.scatter(
    dfSetosaPredict['petal_length'],
    dfSetosaPredict['petal_width'],
    color = 'r'
)
plt.scatter(
    dfVersicolorPredict['petal_length'],
    dfVersicolorPredict['petal_width'],
    color = 'lightgreen'
)
plt.scatter(
    dfVirginicaPredict['petal_length'],
    dfVirginicaPredict['petal_width'],
    color = 'b'
)

#Plot Centroids (Prediction)
plt.scatter(
    centroids[:,0],
    centroids[:,1],
    marker = '*',
    color = 'y',
    s = 300
)

plt.legend(['Setosa', 'Versicolor', 'Virginica', 'Centroids'])
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('K-Means Prediction')
plt.grid(True)


plt.show()