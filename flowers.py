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

m1 = kmeans(3, 0.5).fit(data)
m2 = em(3, 0.1).fit(data)

print(m1.SSE())
print(m2.SSE())
print(m1.silhouette())
print(m2.silhouette())
print(m1.davies())
print(m2.davies())