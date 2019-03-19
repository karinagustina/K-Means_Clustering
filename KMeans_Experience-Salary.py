#SKlearn Machine Learning for Clustering: K-Means Clustering
'''
K-means clustering is a method of vector quantization, originally from signal processing, that is popular for cluster analysis in data mining. 
K-means clustering aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean, serving as a prototype of the cluster.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#================================================
#Load Data
#================================================

df = pd.read_excel('experience-salary.xlsx')
print(df.head(2))
print(df)

#================================================
#Plot Data
#================================================

plt.figure('Employee Salary Based on Experience')
plt.scatter(df['Experience (Years)'], df['Salary (IDR Mn)'])
plt.xlabel('Experience (Years)')
plt.ylabel('Salary (IDR Mn)')
plt.grid(True)

plt.show()

#================================================
#K-Means
#================================================

from sklearn.cluster import KMeans
model = KMeans(n_clusters = 2)

#Training Model
model.fit(df[['Experience (Years)', 'Salary (IDR Mn)']])

#Prediction
print(df['Experience (Years)'].values)
prediction = model.predict(df[['Experience (Years)', 'Salary (IDR Mn)']])
print(prediction)

#Add Prediction Result to DataFrame
df['cluster'] = prediction
print(df)

#Split DataFrame Based on Its Cluster
df0 = df[df['cluster'] == 0]
print(df0)
df1 = df[df['cluster'] == 1]
print(df1)

#Plot df0 and df1
plt.scatter(df0['Experience (Years)'], df0['Salary (IDR Mn)'], marker = 'o', color = 'g')
plt.scatter(df1['Experience (Years)'], df1['Salary (IDR Mn)'], marker = 'o', color = 'y')
plt.ylabel('Salary (IDR Mn)')
plt.grid(True)

#Plot Centroid
print(model.cluster_centers_)
plt.scatter(
    model.cluster_centers_[:,0],
    model.cluster_centers_[:,1],
    marker = '*',
    color = 'r',
    s = 200
)

plt.show()