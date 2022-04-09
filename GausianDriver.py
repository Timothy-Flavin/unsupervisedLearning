from Kmeans import kmeans
from EM import em
import numpy as np
import pandas as pd
fileStats = [
[3, 1000, 0.5,1,1],
[3, 1000, 1,1,1],
[3, 1000, 1.5,1,1],
[3, 1000, 2,1,1],
[5, 1000, 0.5,1,1],
[5, 1000, 1,1,1],
[5, 1000, 1.5,1,1],
[5, 1000, 2,1,1],
[10, 1000, 0.5,1,1],
[10, 1000, 1,1,1],
[10, 1000, 1.5,1,1],
[10, 1000, 2,1,1],

[5, 1000, 3,1,1],
[5, 1000, 3,2,2],
[5, 1000, 3,3,3],

[5, 1000, 1.25,0.75,2],
	
[5, 100, 1.25,0.75,0.75],
[5, 1000, 1.25,0.75,0.75],
[5, 5000, 1.25,0.75,0.75],
]  

files=[]
for i,fi in enumerate(fileStats):
  with open(f"./CreateRandomForKMeans/src/Filec{i+1}.txt") as f:
    files.append(f.readlines())

means=[]
stds=[]

for i,fi in enumerate(files):
  means.append([])
  stds.append([])
  for gens in range(fileStats[i][0]):
    #print(fi[gens])
    gen = fi[gens].split(",")
    means[i].append(float(gen[0]))
    stds[i].append(float(gen[1]))
  #print("Before")
  #print(files[0][0:5])
  files[i]=files[i][fileStats[i][0]:]
  #print("After")
  #print(files[0][0:5])
  
  for j,lj in enumerate(files[i]):
    files[i][j] = files[i][j].split(",")
    files[i][j][0] = float(files[i][j][0])
    files[i][j][1] = float(files[i][j][1])
    files[i][j][0] = means[i].index(files[i][j][0])
  #print("After After")
  #print(files[0][0:5])
print(files[1][0:5])

def exp1(file, fnum=0, nmeans=3, trials=50, method=kmeans):
  data = np.array(file)
  labels = data[:,0].astype(int)
  x = np.expand_dims(data[:,1], axis=-1)
  print(x.shape)
  acc_tot=0
  ce = None
  std = None
  for trial in range(50):
    model = method(nmeans,0.1).fit(x)
    predictions = None
    centers = None
    stds = None
    predictions, centers, stds = model.pred(x)
    clusts = np.zeros((nmeans,np.unique(labels).shape[0]))
    tots = np.zeros((nmeans))
    for i in range(predictions.shape[0]):
      clusts[predictions[i], labels[i]]+=1
      tots[predictions[i]]+=1
    if trial==1:
      ce = centers
      std = stds
      print("Printing stats from first trial")
      print(f"Means: {centers}")
      print(f"Stds: {stds}")
      print(f"Clusts: {clusts}")
      print(f"clusts max: {np.max(clusts,axis=1)}")
      print(f"tots: {tots}")
    accuracies = np.divide(np.max(clusts,axis=1), tots)
    acc = np.sum(np.max(clusts,axis=1)) / np.sum(tots)
    acc_tot+=acc
    if trial==1:
      print(f"Accuracy: {acc}")
      print(f"Accuracy per cluster: {accuracies}")
  acc_tot/=50
  print(f"Accuracy over 50 trials: {acc_tot}")
  
  return {"accuracy": acc, "means":ce, "stds":std }

exp = "em"

if exp=="mean":
  trialacc=[]
  mns = [2,3,6,8]
  #Expiriment 1
  for i in mns:
    trialacc.append(exp1(files[0],fnum=0,nmeans=i,trials=50,method=kmeans))

  for i,f in enumerate(files):
    #print(fileStats[i][0])
    #input()
    trialacc.append(exp1(f,fnum=i,nmeans=fileStats[i][0],trials=50,method=kmeans))
    pd.DataFrame(trialacc).to_csv("GausianKMeans.csv")

  pd.DataFrame(trialacc).to_csv("GausianKMeans.csv")

else:
  trialacc=[]
  mns = [2,3,6,8]
  #Expiriment 1
  for i in mns:
    trialacc.append(exp1(files[0],fnum=0,nmeans=i,trials=50,method=em))

  for i,f in enumerate(files):
    #print(fileStats[i][0])
    #input()
    trialacc.append(exp1(f,fnum=i,nmeans=fileStats[i][0],trials=50,method=em))
    pd.DataFrame(trialacc).to_csv("GausianEM.csv")

  pd.DataFrame(trialacc).to_csv("GausianEM.csv")