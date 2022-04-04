import numpy as np
import pandas as pd

class kmeans:
  def __init__(self, nmeans, tol):
    self.nmeans = nmeans
    self.tol = tol
  
  def fit(self, x, text=False):
    means = np.mean(x, axis=-2)
    stds = np.std(x, axis=-2)
    print(f"shape of means: {means.shape}: {means}")
    print(f"shape of stds: {stds.shape}: {stds}")
    centers = np.random.normal(loc=means, scale=stds/5, size=(self.nmeans,means.shape[0]))
    #centers[0] = np.array([0,0,0])
    print(f"shape of centers: {centers.shape}")
    centers_prev = np.zeros(centers.shape)
    while np.sum(np.abs(centers - centers_prev))/(centers.shape[0]*centers.shape[1])>self.tol:
      print(np.sum(np.sum(np.abs(centers - centers_prev))))
      centers_prev = np.copy(centers)
      print(f"Current centers: {centers}")
      chunks = list()
      for i in range(self.nmeans):
        chunks.append(list())
      for point in range(x.shape[0]):
        bd = 999999
        chunk = -1
        for i in range(centers.shape[0]):
          dist = np.linalg.norm(x[point]-centers[i])
          if dist<bd:
            chunk = i
            bd = dist
        chunks[chunk].append(x[point])
      for i in range(centers.shape[0]):
        #print(np.array(chunks[i]))
        #print(f"Means: {np.mean(np.array(chunks[i]), axis=-2)}")
        centers[i] = np.mean(np.array(chunks[i]), axis=-2)
      self.centers = centers
      print(centers)
      print(np.sum(np.abs(centers - centers_prev)))
    return self