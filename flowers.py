from turtle import pd
import numpy as np
import pandas as pd
from Kmeans import kmeans
from EM import em

df = pd.read_csv("iris_data.csv")
y = df["species"].to_numpy().flatten()
df = df.drop(["species"], axis=1)
data = df.to_numpy()

print(data[0:5])
print(y[0:5])

SSE = [0,0]
silhouette = [0,0]
davies = [0,0]

for i in range(10):
  m1 = kmeans(3, 0.5).fit(data)
  m2 = em(3, 0.1).fit(data)

  SSE[0]+=m1.SSE()
  SSE[1]+=m2.SSE()

  silhouette[0]+=m1.silhouette()
  silhouette[1]+=m2.silhouette()

  davies[0]+=m1.davies()
  davies[1]+=m2.davies()

print(np.array(SSE)/10)
print(np.array(silhouette)/10)
print(np.array(davies)/10)