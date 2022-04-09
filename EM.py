import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

class em:
  def __init__(self, ndists, tol):
    self.ndists = ndists
    self.tol = tol
  
  def fit(self, x, text=False, given_centers=None, verbose = False):
    means = np.mean(x, axis=-2)
    stds = np.std(x, axis=-2)
    if verbose:
      print(f"shape of means: {means.shape}: {means}")
      print(f"shape of stds: {stds.shape}: {stds}")
    centers = given_centers
    if given_centers is None:
      centers = x[np.random.randint(low=0, high=x.shape[0]-1, size=self.ndists, dtype=int),].astype(float)
    if verbose:
      print(centers.shape)
      print(f"shape of centers: {centers.shape}")
    centers_prev = np.zeros(shape=centers.shape)
    chunklens_prev = np.ones(shape=(self.ndists), dtype=int)

    #Set the std of each distribution to the overall std at first
    stdDevs = np.zeros(shape=(self.ndists, x.shape[-1], x.shape[-1]), dtype=float)
    for i in range(stdDevs.shape[0]):
      stdDevs[i]=np.diag(stds)
      
    
    while np.sum(np.abs(centers - centers_prev))/(centers.shape[0]*centers.shape[1])>self.tol:
      prog=0
      report_const = int(x.shape[0]/100)
      report = report_const
      if verbose:
        print(f"X shape: {x.shape}")
        print(f"Current centers: {centers}")
        print(f"Starting Distance: {np.sum(np.sum(np.abs(centers - centers_prev)))/(centers.shape[0]*centers.shape[1])}")
      centers_prev = np.copy(centers)
      
      chunks = np.zeros((self.ndists,x.shape[0],x.shape[1]))
      chunklens = np.zeros(shape=(self.ndists), dtype=int)
      
      for point in range(x.shape[0]):
        if prog == report:
          if verbose:
            print(f"Progress: {prog/x.shape[0]*100}%")
          report+=report_const
        prog+=1
        bd = 0
        chunk = -1
        for i in range(centers.shape[0]):
          #dist = np.linalg.norm(x[point]-centers[i])
          dist = multivariate_normal.pdf(x[point], mean=centers[i], cov=stdDevs[i]) #* chunklens_prev[i]
          if dist>bd:
            chunk = i
            bd = dist
        #print(f"Chunk: {chunk}, chunklens: {chunklens}")
        chunks[chunk,chunklens[chunk]] = x[point]
        chunklens[chunk]+=1
      for i in range(centers.shape[0]):
        if verbose:
          print(f"Updating distribution {i}")
        if chunklens[i] >= 2:
          centers[i] = np.mean(chunks[i,0:chunklens[i]], axis=-2)
          #print(f"Chunk shape: {np.shape(chunks[i,0:chunklens[i]])}")
          #print(f"Cov shape: {np.cov(chunks[i,0:chunklens[i]].T).shape} ")
          stdDevs[i] = np.cov(chunks[i,0:chunklens[i]].T)
        else:
          print("Chunk was too smol")
      chunklens_prev = np.copy(chunklens)
      self.chunklens = np.copy(chunklens)
      self.centers = centers
      self.covs = stdDevs
      self.chunks = chunks
      if verbose:
        print(centers)
        print(np.sum(np.abs(centers - centers_prev))/(centers.shape[0]*centers.shape[1]))
    return self

  def pred(self,x):
    preds = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
      bc = -1
      dist = -1
      for c in range(self.centers.shape[0]):
        td = multivariate_normal.pdf(x[i], mean=self.centers[c], cov=self.covs[c])
        if td>dist:
          dist=td
          bc=c
      preds[i] = bc
    return preds.astype(int), self.centers, np.power(self.covs,0.5).flatten()

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
        #td = multivariate_normal.pdf(x[i], mean=self.centers[c], cov=self.stdDevs[c])
        if td<dist:
          group = c
          dist = td
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
        s[i] = b[i]-a[i] / max(a[i],b[i])
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
       

