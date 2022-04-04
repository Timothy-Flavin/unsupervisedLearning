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
    #centers = np.random.normal(loc=means, scale=stds/5, size=(self.nmeans,means.shape[0]))
    centers = x[np.random.randint(low=0, high=x.shape[0]-1, size=self.nmeans),]
    print(centers.shape)
    #input()
    #centers[0] = np.array([0,0,0])
    print(f"shape of centers: {centers.shape}")
    centers_prev = np.zeros(shape=centers.shape)

    
    while np.sum(np.abs(centers - centers_prev))/(centers.shape[0]*centers.shape[1])>self.tol:
      
      prog=0
      report_const = int(x.shape[0]/100)
      report = report_const
      print(report)
      print(np.sum(np.sum(np.abs(centers - centers_prev))))
      centers_prev = np.copy(centers)
      print(f"Current centers: {centers}")
      chunks = list()
      for i in range(self.nmeans):
        chunks.append(list())
      
      for point in range(x.shape[0]):
        if prog == report:
          print(f"Progress: {prog/x.shape[0]*100}%")
          report+=report_const
        prog+=1
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
        if len(chunks[i]) >= 2:
          centers[i] = np.mean(np.array(chunks[i]), axis=-2)
        else:
          print("Chunk was too smol")

      self.centers = centers
      print(centers)
      print(np.sum(np.abs(centers - centers_prev))/(centers.shape[0]*centers.shape[1]))
    return self