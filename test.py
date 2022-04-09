import numpy as np

x = np.array([
  [1,2,3],
  [2,3,4],
  [6,5,6],
  [0,2,4]
])
center = np.array([9/4.0, 3, 17.0/4])

print(f"x - center: {x-center}")
print(f"squared: {np.power(x-center,2)}")
print(f"summed: {np.sum(np.power(x-center,2),axis=1)}")
print(f"square rooted: {np.power(np.sum(np.power(x-center,2),axis=1),0.5)}")
print(f"Summed: {np.sum(np.power(np.sum(np.power(x-center,2),axis=1),0.5))}")
s1 = np.sum(np.power(np.sum(np.power(x-center,2),axis=1),0.5))/4
print(f"S1: {s1}")