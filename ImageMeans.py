import numpy as np
from PIL import Image
from Kmeans import kmeans

images = [
  "finalImages/firemountain.jpg",
  "finalImages/jupiter.jpg",
  "finalImages/river.jpg",
  "finalImages/lonely-japanese-cherry.jpg",
]
dameans = [3,5,10]

for damean in dameans:
  for imname in images:


    im = np.array(Image.open(imname))

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

    means = kmeans(damean,5).fit(scene, given_centers=None)

    print("Saving picture")
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
    Image.fromarray(scene).save(imname[:-4]+f"{damean}.png")