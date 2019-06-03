#EM

import numpy as np
from matplotlib import pyplot as plt
import scipy.stats
import sklearn.mixture
from matplotlib.patches import Ellipse

#Generate the data
N=200
Normal1 = np.random.multivariate_normal(mean = np.array([0,0]), cov = np.identity(2),size = N)
Normal2 = np.random.multivariate_normal(mean = np.array([1,-3]), cov = np.array([[3,2],[2,2]]),size = N)
Normal3 = np.random.multivariate_normal(mean = np.array([5,0]), cov = np.array([[2,-2],[-2,3]]),size = N)

X = np.vstack((Normal1,Normal2,Normal3))

#Plot the data
#With colour
plt.scatter(Normal1[:,0],Normal1[:,1])
plt.scatter(Normal2[:,0],Normal2[:,1])
plt.scatter(Normal3[:,0],Normal3[:,1])
_  =plt.legend(['Normal 1','Normal 2','Normal3'])
#Without colour
plt.scatter(Normal1[:,0],Normal1[:,1],c = 'g')
plt.scatter(Normal2[:,0],Normal2[:,1],c = 'g')
plt.scatter(Normal3[:,0],Normal3[:,1],c ='g')
_  =plt.legend(['Normal 1','Normal 2','Normal3'])

#Fit EM
gmm = sklearn.mixture.GaussianMixture(n_components=3)
gmm.fit(X)
gmm.means_
gmm.covariances_

#Create elipse corresponding to the mean and covariance.
def cov2elipse(covariance):
    eig_values, eig_vectors = np.linalg.eig(covariance)
    eig_vector = eig_vectors[np.where(eig_values == np.max(eig_values))]
    rotation_rad = np.arccos(np.dot(eig_vector,np.array([0,1])))
    rotation_ang = np.rad2deg(rotation_rad)+90
        
    return eig_values , rotation_ang

def create_elipse(mean,covariance,color):
    a,b = cov2elipse(covariance)

    e = Ellipse(xy=mean,
                    width=2*np.sqrt(a[1]*5.991), height=2*np.sqrt(a[0]*5.991),
                    angle=b-90)

    ax.add_artist(e)
    e.set_clip_box(ax.bbox)
    e.set_alpha(0.4)
    e.set_facecolor(np.array(color))

 #Plot Scatterplot covered by elipse
 fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})


plt.scatter(Normal1[:,0],Normal1[:,1])
plt.scatter(Normal2[:,0],Normal2[:,1])
plt.scatter(Normal3[:,0],Normal3[:,1])

create_elipse(gmm.means_[0],gmm.covariances_[0],np.array([1,0,0]))
create_elipse(gmm.means_[1],gmm.covariances_[1],np.array([0,1,0]))
create_elipse(gmm.means_[2],gmm.covariances_[2],np.array([0,0,1]))
_  =plt.legend(['Normal 1','Normal 2','Normal3'])


#Log Likelihood vs clusters fit:
import sklearn.mixture

log_likelihoods = []
for i in range(1,10): 
    avg_lb = 0
    for k in range(1,10):
        gmm = sklearn.mixture.GaussianMixture(n_components=int(i))
        gmm.fit(X)
        avg_lb+=gmm.lower_bound_
    avg_lb /=10
    log_likelihoods.append(avg_lb)
plt.figure(figsize=(8,8))
plt.title('Log Likelihood vs Clusters Fit',fontsize = 30)
plt.plot(range(1,10),log_likelihoods)
plt.axvline(x=3,color = [1,0,0])
plt.savefig('loglikelihoods_far.png')

#Animation using Bootstrap

%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from IPython.display import HTML

import numpy as np
from matplotlib import pyplot as plt
import scipy.stats
import sklearn.mixture
from matplotlib.patches import Ellipse

def bootstrap(X):
    N = X.shape[0]
    rows = np.random.choice(range(N),N)
    return X[rows,:]

def get_colors():
    #Do a bootstrap sample of X.
    gmm = sklearn.mixture.GaussianMixture(n_components=6)
    gmm.fit(bootstrap(X))
    return gmm.means_

# First set up the figure, the axis, and the plot element we want to animate
fig, ax = plt.subplots()
fig.suptitle('6 Components',fontsize = 20)

ax.set_xlim(( -4, 9))
ax.set_ylim((-7, 5))

scat = ax.scatter([], [], lw=2)

# initialization function: plot the background of each frame
def init():
    scat.set_offsets(np.empty(2))
    return (scat,)

# animation function. This is called sequentially
def animate(i):
    scat.set_offsets(get_colors())
    return (scat,)

# call the animator. blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=100, interval=20, blit=True)

HTML(anim.to_html5_video())

#EM imputation

N=500
Normal1 = np.random.multivariate_normal(mean = np.array([0,0]), cov = np.identity(2),size = N)
Normal2 = np.random.multivariate_normal(mean = np.array([1,-2]), cov = np.array([[2,1.7],[1.7,2]]),size = N)

X = np.vstack((Normal1,Normal2))
X_true = X.copy()

#Remove 1 dimension from 10 random observations
N,K=X.shape

missing = 10
rows = np.random.choice(range(N),size = missing,replace = False)
cols = np.random.choice(range(K),size = missing)
X[rows,cols]=None


X_true = X_true[np.where(np.isnan(X).any(axis=1))[0]]

#Em Algorithm for missing data imputation
class em_impute():
    def __init__(self,components = 2,iterations=10):
        self.components=components
        self.iterations = iterations
    
    def mean_impute(self):
        #impute missing data by column mean
        inds = np.where(np.isnan(self.X_m))

        #Place column means in the indices. Align the arrays using take
        self.X_im = self.X_m.copy()
        self.X_im[inds] = np.take(self.col_means, inds[1])
        
        
    def mle_impute(self):
    	#Imput
        inds = np.where(np.isnan(self.X_m))
        preds = self.gmm.predict(self.X_im)
        for i in range(len(self.X_m)):
            cluster = preds[i]
            if np.isnan(self.X_m[i,0]):
                #missing the first col entry
                self.X_im[i,0] = self.mean[cluster][0]+(self.cov[cluster][0,1]/self.cov[cluster][1,1])*(self.X_m[i,1] - self.mean[cluster][1])
            else:
                self.X_im[i,1] = self.mean[cluster][1]+(self.cov[cluster][0,1]/self.cov[cluster][0,0])*(self.X_m[i,0] - self.mean[cluster][0])
            
                
    def fit(self,X):
        self.X_o = X[np.where(~np.isnan(X).any(axis=1))[0]]
        self.col_means = np.nanmean(self.X_o, axis = 0)
        self.X_m = X[np.where(np.isnan(X).any(axis=1))[0]]
        self.missing = np.sum(np.isnan(self.X_m))
                
        self.mean_impute()

    for i in range(self.iterations):
        if i==0:
            self.gmm = sklearn.mixture.GaussianMixture(n_components=self.components,max_iter=1)
        else:
        	#Do one E and one M step after estimating the full dataset.
            self.gmm = sklearn.mixture.GaussianMixture(n_components=self.components,means_init=self.mean,precisions_init=np.linalg.inv(self.cov),max_iter=1)
        X = np.vstack((self.X_o,self.X_im))
        self.gmm.fit(X)
        self.mean = self.gmm.means_
        self.cov = self.gmm.covariances_
        self.mle_impute()

#Bootstrap and build EM model on Each bootstrap:
bootstrap_im = []
for i in range(100):
    print(i)
    N,M=X.shape
    X_bs = X[np.random.choice(range(N),replace = True,size = N)]
    gmm = em_impute(components=2)
    gmm.fit(X_bs)
    bootstrap_im.append(gmm.X_im)

#Plot the animation
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from IPython.display import HTML

# First set up the figure, the axis, and the plot element we want to animate.
#This is the background with the blue points being the underlying true missing points.
fig, ax = plt.subplots()
fig.suptitle(' of Missing Data',fontsize = 20)
plt.scatter(X_true[:,0],X_true[:,1])
ax.set_xlim(( -5, 5))
ax.set_ylim((-5, 5))

scat = ax.scatter([], [], lw=2)

# initialization function: plot the background of each frame
def init():
    scat.set_offsets(np.empty(2))
    return (scat,)
# animation function. This is called sequentially
def animate(i):
    x = bootstrap_im[i][:,0]
    y = bootstrap_im[i][:,1]
    scat.set_offsets(np.vstack((x,y)).T)
    return (scat,)

# call the animator. bli([1t=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=100, interval=100, blit=True)

HTML(anim.to_html5_video())