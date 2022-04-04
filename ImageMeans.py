import numpy as np
from PIL import Image
from Kmeans import kmeans

im = np.array(Image.open("images/cool.jpg"))

scene = im.reshape((im.shape[0]*im.shape[1],3))
print(scene.shape)

means = kmeans(8,20).fit(scene)

for i in range(scene.shape[0]):
  bestm = -1
  bestd = 9999
  for m in range(means.centers.shape[0]):
    tempd= np.linalg.norm(scene[i]-means.centers[m])
    if tempd < bestd:
      bestd = tempd
      bestm = m
  #print(means.centers.shape)
  #print(np.shape(scene[i]))
  for kill in range(means.centers.shape[1]):
    scene[i,kill] = means.centers[bestm,kill] 

scene = scene.reshape((im.shape[0],im.shape[1],3))
Image.fromarray(scene).save("images/cool8.jpg")