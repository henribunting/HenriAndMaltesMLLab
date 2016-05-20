# coding: utf-8

#Malte Seimers and Henri Bunting

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas

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

#Assignment 2
def dist_matrix(A):
   n = A.shape[0]
   D = np.empty((n,n))
   for i in range(n):
       D[i] = np.sqrt(np.square(A-A[i]).sum(1))
   return D

def k_nearest(X, k):
    D = dist_matrix(X)
    nearest_neighbors = np.argsort(D, axis=1)[:,1:k+1]
    return nearest_neighbors

def gammaidx(X, k):
   distances = dist_matrix(X)
   nearestindicies = k_nearest(X, k)
   
   nearestdistances = np.zeros((len(X),k))
   for i in range(len(X)):
      nearestdistances[i] = distances[i][ nearestindicies[i] ]
   
   return (1.0/k) * np.sum(nearestdistances, axis=1)


#Assignment 3
def auc(y_true, y_val, plot=False):
   y_true_labeled = y_true > 0
   truetotal = np.sum(y_true_labeled, dtype=float)
   falsetotal = len(y_true_labeled) - truetotal
   
   y_val_sorted = np.sort(y_val)

   tprs = []
   fprs = []

   for i in y_val_sorted:
      y_val_labeled = y_val >= i
   
      tp = np.sum(y_true_labeled & y_val_labeled)
      fp = np.sum((y_true_labeled==False) & y_val_labeled)

      tpr = tp / truetotal
      fpr = fp / falsetotal
      
      tprs.append(tpr)
      fprs.append(fpr)

   tprs.append(0)
   tprs.insert(0,1)
   fprs.append(0)
   fprs.insert(0,1)
   
   aucvalue = np.trapz(tprs[::-1], fprs[::-1])

   if plot:
      plt.plot(fprs, tprs)
   
   return aucvalue

def auccumsum(y_true, y_val, plot=False):
   aucvalue = 0;
   zipped = zip(y_val, y_true)
   yv, yt = zip(*sorted(zipped, key=lambda x: x[0], reverse=True))
   yv = np.array(yv)
   yt = np.array(yt)
   
   ts = np.cumsum(yt==1) / np.sum(yt==1)
   fs = np.cumsum(yt==-1) / np.sum(yt==-1)
   
   aucvalue = (np.dot(ts[1:]-ts[:-1], fs[1:]))
   
   return 1.0 - aucvalue

#Assignment 4
def epsilon_ball(X, epsilon):
    D = dist_matrix(X)
    nearest_neighbors = []
    for i in range(len(X)):
        nearest_neighbors.append(np.nonzero(D[i]<=epsilon))
    return nearest_neighbors

def local_C(P, Neighbors):
    D = Neighbors-P
    C = np.tensordot(D,D.T,[1,0])
    return C

def w_matrix(X, tol, n_rule, k, epsilon):
    
    if n_rule == 'eps-ball':
        knn_index = epsilon_ball(X, epsilon)
    else:
        knn_index = k_nearest(X, k)
    W = np.zeros((X.shape[0],X.shape[0]))
    for i,P in enumerate(X):
        C = local_C(P, X[knn_index[i]])
        C += np.eye(C.shape[0])*tol*np.trace(C)
        unconstrained_W = np.linalg.solve(C,np.ones(C.shape[0]))
        constrained_W = unconstrained_W/np.sum(unconstrained_W)
        for kk,j in enumerate(knn_index[i]):
            W[i][j] = constrained_W[kk]
    return W
   
def lle(X, m, tol, n_rule='knn', k=5, epsilon=1.):
    W = w_matrix(X, tol, n_rule, k, epsilon)
    A = np.eye(W.shape[0]) - W
    M = (A.T).dot(A)
    Y = np.linalg.eigh(M)[1][:,1:m+1]*.5*np.sqrt(X.shape[0])
    return Y

#--- Part 2: Application ---

usps = scipy.io.loadmat('usps.mat')
print(usps['data_patterns'].shape)

def assignment5_analyze_pcas(X, noisecoefficient):
   #print(X.shape[1])
   
   print("--- "+str(noisecoefficient)+" ---")
   pca = PCA( X + noisecoefficient*np.random.randn(X.shape[0], X.shape[1]) )
   print(pca.D[250:260])

   plt.bar(range(len(pca.D)), pca.D)
   plt.savefig("5_b_a_"+str(noisecoefficient)+".png")
   plt.close()

   plt.bar(range(25), pca.D[:25])
   plt.savefig("5_b_b_"+str(noisecoefficient)+".png")
   plt.close()
   
   plt.imshow(np.hstack((pca.U[:3].reshape(3*16,1*16), pca.U[4:7].reshape(3*16,1*16))), cmap='gray')
   plt.tick_params(
      axis='both',       # changes apply to the x-axis
      which='both',      # both major and minor ticks are affected
      bottom='off',      # ticks along the bottom edge are off
      top='off',         # ticks along the top edge are off
      left='off',
      right='off',
      labelleft='off',
      labelbottom='off') # labels along the bottom edge are off
   plt.savefig("5_b_c_"+str(noisecoefficient)+".png")
   plt.close()
   #plt.bar(pca.D[:5], range(5))
   #plt.savefig("5.b.c_"+str(noisecoefficient)+".png")
   #plt.close()
   f, axes = plt.subplots(1, 11, sharey=True)
   axes[0].tick_params(
      axis='both',       # changes apply to the x-axis
      which='both',      # both major and minor ticks are affected
      bottom='off',      # ticks along the bottom edge are off
      top='off',         # ticks along the top edge are off
      left='off',
      right='off',
      labelleft='off',
      labelbottom='off') # labels along the bottom edge are off
   axes[0].imshow(X[:10].reshape(10*16,16), cmap='gray')
   for i in range(10):
      Xprime = pca.denoise(X, 5*i)
      axes[i+1].imshow(Xprime[:10].reshape(10*16,16), cmap='gray')
      axes[i+1].tick_params(
         axis='both',       # changes apply to the x-axis
         which='both',      # both major and minor ticks are affected
         bottom='off',      # ticks along the bottom edge are off
         top='off',         # ticks along the top edge are off
         left='off',
         right='off',
         labelleft='off',
         labelbottom='off') # labels along the bottom edge are off
   plt.savefig("5_c_Xprime_"+str(noisecoefficient)+"_"+str(5*i)+"_allinone.png")
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
plt.savefig("5_b_a_23479.png")
plt.close()

plt.bar(range(25), pca.D[:25])
plt.savefig("5_b_b_23479.png")
plt.close()

   
plt.imshow(np.hstack((pca.U[:3].reshape(3*16,1*16), pca.U[4:7].reshape(3*16,1*16))), cmap='gray')
plt.tick_params(
   axis='both',       # changes apply to the x-axis
   which='both',      # both major and minor ticks are affected
   bottom='off',      # ticks along the bottom edge are off
   top='off',         # ticks along the top edge are off
   left='off',
   right='off',
   labelleft='off',
   labelbottom='off') # labels along the bottom edge are off
plt.savefig("5_b_c_23479.png")
plt.close()

f, axes = plt.subplots(1, 11, sharey=True)
axes[0].tick_params(
   axis='both',       # changes apply to the x-axis
   which='both',      # both major and minor ticks are affected
   bottom='off',      # ticks along the bottom edge are off
   top='off',         # ticks along the top edge are off
   left='off',
   right='off',
   labelleft='off',
   labelbottom='off') # labels along the bottom edge are off
axes[0].imshow(X[:10].reshape(10*16,16), cmap='gray')
for i in range(10):
   Xprime = pca.denoise(X, 5*i)
   axes[i+1].imshow(Xprime[:10].reshape(10*16,16), cmap='gray')
   axes[i+1].tick_params(
      axis='both',       # changes apply to the x-axis
      which='both',      # both major and minor ticks are affected
      bottom='off',      # ticks along the bottom edge are off
      top='off',         # ticks along the top edge are off
      left='off',
      right='off',
      labelleft='off',
      labelbottom='off') # labels along the bottom edge are off
plt.savefig("5_c_Xprime_23479_"+str(5*i)+"_allinone.png")
plt.close()

#Assignment 6

data = np.load("banana.npz")

inliersx = data['data'][0][data['label'][0]==1]
inliersy = data['data'][1][data['label'][0]==1]
outliersx = data['data'][0][data['label'][0]==-1]
outliersy = data['data'][1][data['label'][0]==-1]

aucs = []
for outlierpercent in [.01,.05,.10,.25]:
   #print(" -------"+str(outlierpercent)+"------- ")
   numberofoutliers = np.floor(outlierpercent*len(inliersx))
   
   #fullx = np.append(inliersx, outliersx)
   #fully = np.append(inliersy, outliersy)
   #fulllabels = np.append(data['label'][0][data['label'][0]==1], data['label'][0])
   #fullxy = np.append([fullx], [fully], axis=0)
   
   aucas = []
   aucbs = []
   auccs = []
   for trial in range(100):
      #1. Choose a random set of outliers from the negative class of the respective size (depending on the outlier rate).
      trialoutlierindicies = np.random.random_integers(0,len(outliersx)-1, (1, numberofoutliers))
      trialoutliersx = outliersx[trialoutlierindicies]
      trialoutliersy = outliersy[trialoutlierindicies]
      
      #2. Add the outliers to the positive class
      trialx = np.append(inliersx, trialoutliersx)
      trialy = np.append(inliersy, trialoutliersy)
      triallabels = np.append(data['label'][0][data['label'][0]==1], data['label'][0][trialoutlierindicies])
      trialxy = np.append([trialx], [trialy], axis=0)
      #print(triallabels.shape)
      #print(trialxy.shape)
      
      #gammatime = time.time()
      # compute (a) the γ-index with k = 3
      resulta = gammaidx(trialxy.T, 3)
      #(b) the γ-index with k = 10
      resultb = gammaidx(trialxy.T, 10)
      # and (c) the distance to the mean for each data point.
      resultc = np.sqrt(np.square(trialxy - np.mean(trialxy, axis=1).reshape((2,1))).sum(0))      
      
      #auctime = time.time()
      #3. Compute the AUC (area under the ROC) for each method.
      auca = 1.0 - auccumsum(triallabels, resulta)
      aucb = 1.0 - auccumsum(triallabels, resultb)
      aucc = 1.0 - auccumsum(triallabels, resultc)
      #endtime = time.time()
      aucas.append(auca)
      aucbs.append(aucb)
      auccs.append(aucc)
      #print(str(auctime - gammatime)+" - "+str(endtime - auctime))
      '''
      if trial%10 == 0:
         print(str(outlierpercent) + " - "+str(int(trial/10)))
      '''
   aucs.append(aucas)
   aucs.append(aucbs)
   aucs.append(auccs)
   plt.boxplot([aucas,aucbs,auccs])
   plt.xticks([1, 2, 3], [str(int(outlierpercent*100))+'% gamma, k=3', str(int(outlierpercent*100))+'% gamma, k=10', str(int(outlierpercent*100))+'% mean dist'])
   plt.savefig("6."+str(outlierpercent)+".png")
   plt.close()

#ax.set_xticklabels()
plt.boxplot(aucs)
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 
            ['1% - gamma, k=3', '1% - gamma, k=10', '1% - mean dist',
             '5% - gamma, k=3', '5% - gamma, k=10', '5% - mean dist',
             '10% - gamma, k=3', '10% - gamma, k=10', '10% - mean dist',
             '25% - gamma, k=3', '25% - gamma, k=10', '25% - mean dist'], 
             rotation=37, horizontalalignment='right')
plt.margins(0.2)                 # Pad margins so that markers don't get clipped by the axes
plt.subplots_adjust(bottom=0.30) # Tweak spacing to prevent clipping of tick-labels
plt.savefig("6.png")
plt.close()

#Assignment 7

fishbowl = np.load('fishbowl_dense.npz')['X'].T
swissroll = np.load('swissroll_data.npz')['x_noisefree'].T
ref_swissroll = np.load('swissroll_data.npz')['z'].T
flatroll = np.load('flatroll_data.npz')['Xflat'].T
ref_flatroll = np.load('flatroll_data.npz')['true_embedding'].T

fig1 = plt.figure()
ax = fig1.add_subplot(111, projection='3d')
ax.scatter(fishbowl.T[0],fishbowl.T[1],fishbowl.T[2], c=fishbowl.T[2])
fig1.savefig('7.a_0.png')

f, axarr = plt.subplots(2, 2)
index_pairs = [[0,0],[0,1],[1,0],[1,1]]
knn = 23
mean = 0.0025
for i in range(4):
    tol= mean*(i+1)
    fishbowl_2d = lle(fishbowl, 2, tol, n_rule='knn', epsilon=0.5, k=knn)
    axarr[index_pairs[i][0],index_pairs[i][1]].set_title('tol = '+str(tol))
    axarr[index_pairs[i][0],index_pairs[i][1]].scatter(fishbowl_2d.T[0], fishbowl_2d.T[1], c=fishbowl.T[2])
plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)
plt.savefig('7.a_1.png')
    
fig1 = plt.figure()
ax = fig1.add_subplot(111, projection='3d')
ax.scatter(swissroll.T[0],swissroll.T[1],swissroll.T[2], c=swissroll.T[2])
fig1.savefig('7.a_2.png')

f, axarr = plt.subplots(2, 2)
index_pairs = [[0,0],[0,1],[1,0],[1,1]]
knn = 13
mean = 0.000025
for i in range(4):
    tol= 10**(0)*mean*(i+1)
    fishbowl_2d = lle(swissroll, 2, tol, n_rule='knn', epsilon=0.5, k=knn)
    axarr[index_pairs[i][0],index_pairs[i][1]].set_title('tol = '+str(tol))
    axarr[index_pairs[i][0],index_pairs[i][1]].scatter(fishbowl_2d.T[0], fishbowl_2d.T[1], c=ref_swissroll[:,1])
f.title = 'k_'+str(knn)
plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)
plt.savefig('7.a_3.png')

fig1 = plt.figure()
ax = fig1.add_subplot(111, projection='3d')
ax.scatter(flatroll.T[0],flatroll.T[1], c=ref_flatroll)
fig1.savefig('7.a_4.png')

f, axarr = plt.subplots(2, 2)
index_pairs = [[0,0],[0,1],[1,0],[1,1]]
knn = 13
mean = 0.000025
for i in range(4):
    tol= 10**(2)*mean*(i+1)
    fishbowl_2d = lle(flatroll, 1, tol, n_rule='knn', epsilon=0.5, k=knn)
    axarr[index_pairs[i][0],index_pairs[i][1]].set_title('tol = '+str(tol))
    axarr[index_pairs[i][0],index_pairs[i][1]].scatter(fishbowl_2d, np.zeros_like(fishbowl_2d), c=ref_flatroll)
f.title = 'k_'+str(knn)
plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)
plt.savefig('7.a_5.png')

#Assignment 8

#1. Load the data set.
loadeddata = np.load("flatroll_data.npz")
#print(loadeddata['true_embedding'].shape)
print(loadeddata['Xflat'].T.shape)

#2. Add Gaussian noise with variance 0.2 and 1.8 to the data set (this results in 2 noisy data sets).
data0 = loadeddata['Xflat'].T
data02 = loadeddata['Xflat'].T + np.random.normal(0, .2, (loadeddata['Xflat'].T.shape))
data18 = loadeddata['Xflat'].T + np.random.normal(0, 1.8, (loadeddata['Xflat'].T.shape))

#3. Apply LLE on both data sets, where the neighborhood graph should be constructed using k-nn.
#For both noise levels, try to find 
#(a) a good value for k which unrolls the flat roll and  
data02lle = lle(data0, 2, .001, n_rule='knn', k=10, epsilon=1.)
#print(data02lle.shape)
dataframe = pandas.DataFrame(data02lle, columns=['x', 'y', 'z'])
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data02lle.T[0], data02lle.T[1], data02lle.T[2])
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

fig.savefig('8.a.png')

#(b) a value which is obviously too large.

#4. For each of the four combinations of low/high noise level good/too large k, plot the neighborhood graph and the resulting embedding.

