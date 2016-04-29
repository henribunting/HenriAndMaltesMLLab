import numpy as np
import scipy.io
import matplotlib.pyplot as plt


#--- Part 1: Implementation ---
#Assignment 1
class PCA():
   def __init__(self, X):
      self.n, self.d = X.shape   #X row-wise points
      self.U, self.D = self.solveeigensystem(self.scatter(X))
   def project(self, X, m):
      Ureduce = self.U[0:m]     #row-wise eigenvectors
      Z = np.dot(Ureduce, self.center(X).T).T
      return Z
   def denoise(self, X, m):
      Ureduce = self.U[0:m]     #row-wise eigenvectors
      Xprime = self.decenter(np.dot(Ureduce.T, self.project(X, m).T).T)
      return Xprime
   def center(self, X):
      self.Xbar = (1.0/self.n) * (np.sum(X, axis=0)).T
      return X - self.Xbar
   def decenter(self, X):
      return X + self.Xbar
   def scatter(self, X):
      Xcentered = self.center(X)
      S = np.dot(Xcentered.T, Xcentered)
      return S
   def solveeigensystem(self, X):
      #TODO: use eigh / eigvalsh ?
      D, U = np.linalg.eigh(X)
      U = U.T[::-1]*-1              #eigen vectors are row-wise now
      D = D[::-1] / (self.n - 1)
      return U, D
'''
X = np.array([[ -2.133268233289599,   0.903819474847349,   2.217823388231679, -0.444779660856219,
               -0.661480010318842,  -0.163814281248453,  -0.608167714051449,  0.949391996219125],
             [ -1.273486742804804,  -1.270450725314960,  -2.873297536940942,   1.819616794091556,
               -2.617784834189455,   1.706200163080549,   0.196983250752276,   0.501491995499840],
             [ -0.935406638147949,   0.298594472836292,   1.520579082270122,  -1.390457671168661,
               -1.180253547776717,  -0.194988736923602,  -0.645052874385757,  -1.400566775105519]]).T
m = 2;
correct_Z = np.array([  [   -0.264248351888547, 1.29695602132309, 3.59711235194654, -2.45930603721054,
                           1.33335186376208, -1.82020953874395, -0.85747383354342, -0.82618247564525],
                       [   2.25344911712941, -0.601279409451719, -1.28967825406348, -0.45229125158068,
                           1.82830152899142, -1.04090644990666, 0.213476150303194, -0.911071431421484]]).T

correct_U = np.array([  [   0.365560300980795,  -0.796515907521792,  -0.481589114714573],
                       [   -0.855143149302632,  -0.491716059542403,   0.164150878733159],
                       [  0.367553887950606,  -0.351820587590931,   0.860886992351241]] ).T

correct_D = np.array(   [ 3.892593483673686,   1.801314737893267,   0.356275626798182 ])

correct_X_denoised = np.array([[-1.88406616, -1.35842791, -1.38087939],
                                [ 0.96048487, -1.28976527,  0.19729962],
                                [ 2.34965134, -2.91823143,  1.28492391],
                                [-0.53132686,  1.84911663, -1.23574621],
                                [-0.96141012, -2.51555289, -0.64409954],
                                [ 0.17114282,  1.59202918, -0.79375686],
                                [-0.47605492,  0.15195227, -0.88121723],
                                [ 0.43110399,  0.67815178, -0.47407698]])
#pca = PCA(X)
#print(pca.project(X, m=2))
#print(pca.scatter(X))
#print(pca.U)
#print(pca.D)
#print(pca.project(X,m))
#print(pca.denoise(X, m))

#plt.bar(pca.D, range(len(pca.D)))
#plt.savefig("1.0.png")
#plt.close()
'''


#Assignment 2
def dist_matrix(A):
   n = A.shape[0]
   D = np.empty((n,n))
   for i in range(n):
       D[i] = np.sqrt(np.square(A-A[i]).sum(1))
   return D

def k_nearest(X, k_nn):
   D = dist_matrix(X)
   nearest_neighbors = np.empty((X.shape[0], k_nn), dtype=int)
   for i in range(X.shape[0]):
      nearest_neighbors[i]=np.argsort(D[i])[1:k_nn+1]
   return nearest_neighbors

def gammaidx(X, k):
   distances = dist_matrix(X)
   nearestindicies = k_nearest(X, k)
   
   nearestdistances = np.zeros((len(X),k))
   for i in range(len(X)):
      nearestdistances[i] = distances[i][ nearestindicies[i] ]
   
   return (1.0/k) * np.sum(nearestdistances, axis=1)

'''
X = np.array([  [   0.5376671395461, -2.25884686100365, 0.318765239858981, -0.433592022305684, 3.57839693972576,
   -1.34988694015652, 0.725404224946106, 0.714742903826096, -0.124144348216312, 1.40903448980048,
    0.67149713360808, 0.717238651328838, 0.488893770311789, 0.726885133383238, 0.293871467096658,
    0.888395631757642, -1.06887045816803, -2.9442841619949, 0.325190539456198, 1.37029854009523],
[   1.83388501459509, 0.862173320368121, -1.30768829630527, 0.34262446653865, 2.76943702988488,
    3.03492346633185, -0.0630548731896562, -0.204966058299775, 1.48969760778546, 1.41719241342961,
    -1.20748692268504, 1.63023528916473, 1.03469300991786, -0.303440924786016, -0.787282803758638,
    -1.14707010696915, -0.809498694424876, 1.4383802928151, -0.754928319169703, -1.7115164188537]]).T

k = 3;

correct_gamma = np.array([ 0.606051220224367, 1.61505686776722, 0.480161964450438, 1.18975154873627,
                            2.93910520141032, 2.15531724762712, 0.393996268071324, 0.30516080506303,
                            0.787481421847747, 0.895402545799062, 0.385599174039363, 0.544395897115756,
                            0.73397995201338, 0.314642851266896, 0.376994725474732, 0.501091387197748,
                            1.3579045507961, 1.96372676400505, 0.389228251829715, 0.910065898315003])
print(gammaidx(X, k))
'''


#Assignment 3
def auc(y_true, y_val, plot=False):
   y_true_labeled = y_true > 0
   truetotal = np.sum(y_true_labeled, dtype=float)
   falsetotal = len(y_true_labeled) - truetotal
   
   y_val_sorted = np.sort(y_val)
   stepsize = min(y_val_sorted[1:] - y_val_sorted[:-1])
   
   tprs = []
   fprs = []
   for i in np.arange(min(y_val), max(y_val), stepsize):
      y_val_labeled = y_val >= i
   
      tp = np.sum(y_true_labeled & y_val_labeled)
      fp = np.sum((y_true_labeled==False) & y_val_labeled)
      #print("bias: "+str(i)+" tp: "+str(tp)+" fp: "+str(fp))
   
      tpr = tp / truetotal
      fpr = fp / falsetotal
      
      tprs.append(tpr)
      fprs.append(fpr)
      #print("bias: "+str(i)+" tpr: "+str(tpr)+" fpr: "+str(fpr))
   plt.plot(fprs, tprs)
   
   #print("tprs:"+str(tprs))
   #print("fprs:"+str(fprs))
   
   tprs.append(0)
   tprs.insert(0,1)
   fprs.append(0)
   fprs.insert(0,1)
   
   aucvalue = np.trapz(tprs[::-1], fprs[::-1])
   #print("aucvalue:"+str(aucvalue))
   return aucvalue

#print( auc(np.array([-1, -1, -1, +1, +1]), np.array([0.3, 0.4, 0.5, 0.6, 0.7]), False) )
#print( auc(np.array([-1, -1, -1, +1, +1, +1]), np.array([0.3, 0.4, 0.6, 0.5, 0.7, 0.8]), plot=True) )

#Assignment 4

def local_C(P, Neighbors):
   D = P-Neighbors
   C = np.tensordot(D,D.T,[1,0])
   return C

def w_matrix(X, k_nn):
   knn_index = k_nearest(X, k_nn)
   #print(knn_index.shape)
   W = np.zeros((X.shape[0],X.shape[0]))
   #print(k_nn)
   #print(X.shape)
   #print(W.shape)
   for i,P in enumerate(X):
      C = local_C(P, X[knn_index[i]])
      unconstrained_W = 1./np.sum(C, axis=0)
      constrained_W = unconstrained_W/np.sum(unconstrained_W)
      for kk,j in enumerate(knn_index[i]):
         #print("i:"+str(i)+" j:"+str(j)+" kk:"+str(kk))
         W[i][j] = constrained_W[kk]
   return W
   

def lle(X, m, tol, n_rule='knn', k=5, epsilon=1.):
   W = w_matrix(X, k)
   #print(W.shape)
   A = np.eye(*W.shape) - W
   M = (A.T).dot(A)
   Y = np.linalg.eigh(M)[1][1:m+1].T
   return Y



'''
n = 500
Xt = 10. * np.random.rand(n, 2);
X = np.append(Xt, 0.5 * np.random.randn(n, 8), 1);

# Rotate data randomly.
X = np.dot(X, self.randrot(10).T);

with self.subTest(n_rule='knn', k=30):
   Xp = imp.lle(X, 2, n_rule='knn', k=30, tol=1e-3)
   self.plot(Xt, Xp, 'knn')
with self.subTest(n_rule='eps-ball', epsilon=5.):
   Xp = imp.lle(X, 2, n_rule='eps-ball', epsilon=5., tol=1e-3)
   self.plot(Xt, Xp, 'eps-ball')
with self.subTest(n_rule='eps-ball', epsilon=0.5):
   with self.assertRaises(ValueError, msg='Graph should not be connected and raise ValueError.'):
       imp.lle(X, 2, n_rule='eps-ball', epsilon=0.5, tol=1e-3)
'''

#--- Part 2: Application ---
'''
#Assignment 5
usps = scipy.io.loadmat('usps.mat')
print(usps['data_patterns'].shape)

def assignment5_analyze_pcas(X, noisecoefficient):
   #print(X.shape[1])
   
   print("--- "+str(noisecoefficient)+" ---")
   pca = PCA( X + noisecoefficient*np.random.randn(X.shape[0], X.shape[1]) )
   print(pca.D[250:260])

   plt.bar(range(len(pca.D)), pca.D)
   plt.savefig("5.b.a_"+str(noisecoefficient)+".png")
   plt.close()

   plt.bar(range(25), pca.D[:25])
   plt.savefig("5.b.b_"+str(noisecoefficient)+".png")
   plt.close()
   
   plt.imshow(pca.U[:5].reshape(5*16,16), cmap='gray')
   plt.savefig("5.b.c_"+str(noisecoefficient)+".png")
   plt.close()
   #plt.bar(pca.D[:5], range(5))
   #plt.savefig("5.b.c_"+str(noisecoefficient)+".png")
   #plt.close()

   for i in range(10):
      Xprime = pca.denoise(X, 5*i)
      plt.imshow(Xprime[:10].reshape(10*16,16), cmap='gray')
      plt.savefig("5.c.Xprime_"+str(noisecoefficient)+"_"+str(5*i)+".png")
   plt.imshow(X[:10].reshape(10*16,16), cmap='gray')
   plt.savefig("5.c.X_"+str(noisecoefficient)+".png")
   plt.close()

   
assignment5_analyze_pcas(usps['data_patterns'].T, 0)
assignment5_analyze_pcas(usps['data_patterns'].T, .5)
assignment5_analyze_pcas(usps['data_patterns'].T, 3)

X = usps['data_patterns'].T
noisecoefficient = 7
X[2] += noisecoefficient*np.random.randn(X.shape[1])
X[4] += noisecoefficient*np.random.randn(X.shape[1])
X[6] += noisecoefficient*np.random.randn(X.shape[1])
X[8] += noisecoefficient*np.random.randn(X.shape[1])
X[10] += noisecoefficient*np.random.randn(X.shape[1])

pca = PCA( X )

plt.bar(range(len(pca.D)), pca.D)
plt.savefig("5.b.a_23479.png")
plt.close()

plt.bar(range(25), pca.D[:25])
plt.savefig("5.b.b_23479.png")
plt.close()

for i in range(10):
   Xprime = pca.denoise(X, 5*i)
   plt.imshow(Xprime[:10].reshape(10*16,16), cmap='gray')
   plt.savefig("5.d.Xprime."+str(5*i)+".png")
plt.imshow(X[:10].reshape(10*16,16), cmap='gray')
plt.savefig("5.d.X.png")
plt.close()
'''

#Assignment 6
#Assignment 7
#Assignment 8
