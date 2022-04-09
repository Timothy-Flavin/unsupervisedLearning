import numpy as np
import pandas as pd

class kmeans:
  def __init__(self, nmeans, tol):
    self.nmeans = nmeans
    self.tol = tol
  
  def fit(self, x, text=False, given_centers=None, verbose=False):
    means = np.mean(x, axis=-2)
    stds = np.std(x, axis=-2)
    if verbose:
      print(f"shape of means: {means.shape}: {means}")
      print(f"shape of stds: {stds.shape}: {stds}")
    centers = given_centers
    if given_centers is None:
      try:
        centers = x[np.random.randint(low=0, high=int(x.shape[0]-1), size=self.nmeans, dtype=int),].astype(float)
      except:
        print(f"{x.shape[0]-1}, {self.nmeans}")
    if verbose:
      print(centers.shape)
      print(f"shape of centers: {centers.shape}")
    centers_prev = np.zeros(shape=centers.shape)

    chunks = np.zeros((self.nmeans,x.shape[0],x.shape[1]))
    while np.sum(np.abs(centers - centers_prev))/(centers.shape[0]*centers.shape[1]) > self.tol:
      
      prog=0
      report_const = int(x.shape[0]/100)
      report = report_const
      if verbose:
        print(f"X shape: {x.shape}")
        print(f"Current centers: {centers}")
        print(f"Centers prev before copy: {centers_prev}")
        print(f"Centers diff copy: {centers - centers_prev}")
        print(f"Starting Dist: {np.sum(np.sum(np.abs(centers - centers_prev)))/(centers.shape[0]*centers.shape[1])}")
        
      centers_prev = np.copy(centers).astype(float)
      chunklens = np.zeros(shape=(self.nmeans), dtype=int)
      
      for point in range(x.shape[0]):
        if prog == report:
          if verbose:
            print(f"Progress: {prog/x.shape[0]*100}%")
          report+=report_const
        prog+=1
        bd = 999999
        chunk = -1
        for i in range(centers.shape[0]):
          dist = np.linalg.norm(x[point].astype(int)-centers[i], ord=2)
          
          if dist<bd:
            chunk = i
            bd = dist
        if chunk==-1:
          print("Something wrong")
          exit()
        np.copyto(dst=chunks[chunk,chunklens[chunk]], src=x[point])
        chunklens[chunk]+=1
      for i in range(centers.shape[0]):
        if chunklens[i] >= 2:
          centers[i] = np.mean(chunks[i,0:chunklens[i]], axis=0).astype(float)
        else:
          if verbose:
            print("Chunk was too smol")
      #input()
      self.centers = np.copy(centers)
      self.chunklens = chunklens
      self.chunks = chunks
    if verbose:
      print("Done finding means")
    return self
  
  def pred(self,x):
    preds = np.zeros(x.shape[0])
    stds = np.zeros(self.chunks.shape[0])
    for i in range(self.chunks.shape[0]):
      stds[i] = np.std(self.chunks[i])
    for i in range(x.shape[0]):
      bc = -1
      dist = 99999
      for c in range(self.centers.shape[0]):
        td = np.linalg.norm(x[i].astype(float)-self.centers[c], ord=2)
        if td<dist:
          dist=td
          bc=c
      preds[i] = bc
    return preds.astype(int), self.centers, stds

  def SSE_unseen(self, x):
    x = np.array(x)
    tot_error = 0
    # for each point find the center it belongs too and the distance
    # from that center
    for i in range(x.shape[0]):
      group=0
      dist = 9999.0
      for c in range(self.centers.shape[0]):
        td = np.linalg.norm(x[i].astype(int)-self.centers[c], ord=2)
        if td<dist:
          group = c
          dist =td
      tot_error+=pow(td,2)
    return tot_error
  
  def SSE(self):
    tot_error = 0
    # for each point find the center it belongs too and the distance
    # from that center
    for cluster in range(self.chunks.shape[0]):
      for i in range(self.chunklens[cluster]):
        td = np.linalg.norm(self.chunks[cluster,i].astype(float)-self.centers[cluster], ord=2)      
        tot_error+=pow(td,2)
    return tot_error

  def silhouette(self):
    sc = np.zeros(self.chunks.shape[0])
    tot_error = 0
    # for each point find the center it belongs too and the distance
    # from that center
    for cluster in range(self.chunks.shape[0]):      
      #Finds a(i)
      a = np.zeros((self.chunklens[cluster])) 
      b = np.copy(a)
      s = np.copy(b)
      for i in range(self.chunklens[cluster]):
        if self.chunklens[cluster]<=1:
          s[i]=0
          continue
        tempd = 0
        for j in range(self.chunklens[cluster]): 
          tempd += np.linalg.norm(self.chunks[cluster,i].astype(float)-self.chunks[cluster,j], ord=2)      
        
        tempd/=(self.chunklens[cluster]-1)
        #print(a)
        #print(i)
        a[i]=tempd

        bdist = 99999
        for bcluster in range(self.chunks.shape[0]):
          if bcluster==cluster:
            continue
          if self.chunklens[bcluster] <1:
            s[i]=0
            continue
          tb = 0
          for j in range(self.chunklens[bcluster]): 
            tb += np.linalg.norm(self.chunks[cluster,i].astype(float)-self.chunks[bcluster,j], ord=2)      
          tb/=self.chunklens[bcluster]
          if tb<bdist:
            bdist = tb
        b[i] = bdist
        s[i] = (b[i]-a[i]) / max(a[i],b[i])
      sc[cluster] = np.mean(s)
    sil_score = 0
    for i in range(sc.shape[0]):
      sil_score+=sc[i]*self.chunklens[i]
    sil_score/=np.sum(self.chunklens)
    return sil_score

  def davies(self):
    dbi = np.zeros((self.centers.shape[0]))
    nzeros = 0

    #find this cluster's Si
    for c1 in range(self.centers.shape[0]):
      if self.chunklens[c1]>0:
        #print("Sanity check")
        #print(self.chunks[c1,0:5])
        #print(self.centers[c1])
        #print(self.chunks[c1,0:5]-self.centers[c1])
        s1 = np.sum(np.power(np.sum(np.power(self.chunks[c1,0:self.chunklens[c1]]-self.centers[c1],2),axis=1),0.5))/self.chunklens[c1]
        #print(s1)
        r=-9999999
        for c2 in range(self.centers.shape[0]):
          if c1==c2 or self.chunklens[c2]<1:
            continue
          s2 = np.sum(np.power(np.sum(np.power(self.chunks[c2,0:self.chunklens[c2]]-self.centers[c2],2),axis=1),0.5))/self.chunklens[c2]
          mij = np.linalg.norm(self.centers[c1]-self.centers[c2],ord=2)
          if mij==0:
            mij=0.01
            print("mij was 0")
          tr = (s1+s2)/mij
          if tr>r:
            r=tr
        dbi[c1-nzeros]=r
      else:
        print("Chunk had 0 length so we skip it")
        nzeros+=1
    return np.sum(dbi)/(dbi.shape[0]-nzeros)
      


