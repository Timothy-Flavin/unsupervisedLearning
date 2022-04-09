import numpy as np
from PIL import Image
from EM import em
from scipy.stats import multivariate_normal

im = np.array(Image.open("images/ds.jpg"))

centers = np.array([
  [95,192,240], #light blue
  [72,123,187], #mid blie
  [90,170,100], #mid green
  [111,90,117], #purple wall
  [15,12,24], #dark/black
  [21,75,134], # dark blue
  [230,230,230], #highlight
])

scene = im.reshape((im.shape[0]*im.shape[1],3))
print(scene.shape)

means = em(5,5).fit(scene, given_centers=None)

print("Saving picture")
for i in range(scene.shape[0]):
  bestm = -1
  bestd = 0
  for m in range(means.centers.shape[0]):
    #tempd= np.linalg.norm(scene[i]-means.centers[m])
    tempd = multivariate_normal.pdf(scene[i], mean=means.centers[m], cov=means.covs[m]) #* means.chunklens[m]
    if tempd > bestd:
      bestd = tempd
      bestm = m
  #print(means.centers.shape)
  #print(np.shape(scene[i]))
  for kill in range(means.centers.shape[1]):
    scene[i,kill] = means.centers[bestm,kill] 

scene = scene.reshape((im.shape[0],im.shape[1],3))
Image.fromarray(scene).save("images/ds128.png")