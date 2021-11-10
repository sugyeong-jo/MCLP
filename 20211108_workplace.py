#%%
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, Point
import random
import time
from scipy.spatial import distance_matrix
from mip import Model, xsum, maximize, BINARY
import pydeck as pdk

# %%
def generate_candidate_sites(points, M=10):
    M = len(points)
    index = random.sample(range(0, len(points)), k=M)
    # ix = [random.randint(0, 50) for i in range(M)]
    # iy = [random.randint(0, 50) for i in range(M)]
    pd_points = pd.DataFrame(points)
    lon = pd_points[0]
    lat = pd_points[1]
    sites = [] 
    for i in range(M):
        sites.append(Point(np.array(lon[i]), np.array(lat[i])))

    return np.array([(p.x, p.y) for p in sites])

def run(points, K, radius, M, w):
    """
    Solve maximum covering location problem
    Input:
        points: input points, Numpy array in shape of [N,2]
        K: the number of sites to select
        radius: the radius of circle
        M: the number of candidate sites, which will randomly generated inside
        the ConvexHull wrapped by the polygon
    Return:
        opt_sites: locations K optimal sites, Numpy array in shape of [K,2]
        f: the optimal value of the objective function
    """
    print('----- Configurations -----')
    print('  Number of points %g' % points.shape[0])
    print('  K %g' % K)
    print('  Radius %g' % radius)
    print('  M %g' % M)
    start = time.time()
    sites = generate_candidate_sites(points, M=10)
    J = sites.shape[0]
    I = points.shape[0]
    D = distance_matrix(points, sites)
    mask1 = D <= radius
    D[mask1] = 1
    D[~mask1] = 0

    # Build model
    m = Model("mclp")
    # Add variables
    x = [m.add_var(name="x%d" % j, var_type=BINARY) for j in range(J)]
    y = [m.add_var(name="y%d" % i, var_type=BINARY) for i in range(I)]

    m.objective = maximize(xsum(w[i]*y[i] for i in range(I)))
    m += xsum(x[j] for j in range(J)) == K

    for i in range(I):
        m += xsum(x[j] for j in np.where(D[i] == 1)[0]) >= y[i]

    m.max_gap = 0.05
    m.optimize(max_seconds=300)
    end = time.time()
    print('----- Output -----')
    print('  Running time : %s seconds' % float(end-start))
    print('  Optimal coverage points: %g' % m.objective_value)
    solution = []
    for i in range(J):
        if x[i].x ==1:
            solution.append(int(x[i].name[1:]))
    opt_sites = sites[solution]

    return opt_sites,m.objective_value
# %%
lon = np.arange(123, 124, 0.001)
lat = np.arange(34, 35, 0.001)

point=[]
for x_coord in lat:
    for y_coord in lon:
        point.append(Point(np.array(y_coord), np.array(x_coord)))
points = np.array([(p.x, p.y) for p in point])

w = [random.choice(range(0, 1000)) for i in range(len(points))]

#%%
radius = (1/88.74/1000)*500
K = 20
M = 500
selected_idx = random.sample(range(0, len(points)), k=M)
candidate_sites = points[selected_idx, :]
pd_cand_sites = pd.DataFrame(candidate_sites)
opt_sites_org, f = run(candidate_sites, K, radius, M, w)
# %%
selected_idx
# %%
#%%
# example of bayesian optimization for a 1d function from scratch
from math import sin
from math import pi
from numpy import arange
from numpy import vstack
from numpy import argmax
from numpy import asarray
from numpy.random import normal
from numpy.random import random
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from warnings import catch_warnings
from warnings import simplefilter
from matplotlib import pyplot
import random
#%%
# objective function
def objective(x, noise=0.1):
    noise = normal(loc=0, scale=noise)
	#y = (x**2 * sin(5 * pi * x)**6.0) 
    y = gpr.predict(np.array([x]), return_std=True)[0][0]+noise
	#y = -(x**4-x**2)+noise
	#print(f"'x': {x} | 'y': {y}")
    return y

 
# surrogate or approximation for the objective function
def surrogate(model, X):
	# catch any warning generated when making a prediction
	with catch_warnings():
		# ignore generated warnings
		simplefilter("ignore")
		return model.predict([X], return_std=True)
 
# probability of improvement acquisition function
def acquisition(X, Xsamples, model):
	# calculate the best surrogate score found so far
	best_list = []
	for X_elm in X:
		yhat, _ = surrogate(model, X_elm)
		best_list.append(yhat)
	best = max(best_list)
	# calculate mean and stdev via surrogate function
	# mu, std = surrogate(model, Xsamples)
	# mu = mu[:, 0]

	mu = []
	for sample in Xsamples:
		m, std = surrogate(model, sample)
		#print(m[0])
		mu.append(m[0])
	
	# calculate the probability of improvement
	probs = norm.cdf((mu - best) / (std+1E-9))
	return probs
 
# optimize the acquisition function
def opt_acquisition(X, y, model):
	# random search, generate random samples
	# Xsamples = random(100, 10)
	# Xsamples = Xsamples.reshape(len(Xsamples), 1)
	Xsamples = []
	for i in range(10):
		samples = []
		samples.append(100)
		samples.append(300)
		samples.append(45)
		samples.append(25)
		samples.append(random.randint(0, 50))
		samples.append(random.randint(0, 50))
		samples.append(-random.randint(0, 50))
		samples.append(-random.randint(0, 50))
		samples.append(random.randint(20, 50))
		samples.append(random.randint(50, 80))
		Xsamples.append(samples)
	Xsamples
	# calculate the acquisition function for each sample
	scores = acquisition(X, Xsamples, model)
	# locate the index of the largest scores
	ix = argmax(scores)

	return Xsamples[ix]
 
# plot real observations vs surrogate function
def plot(X, y, model):
	# scatter plot of inputs and real objective function
	pyplot.scatter(X, y)
	# line plot of surrogate function across domain
	Xsamples = asarray(arange(0, 1, 0.001))
	Xsamples = Xsamples.reshape(len(Xsamples), 1)
	ysamples, _ = surrogate(model, Xsamples)
	pyplot.plot(Xsamples, ysamples)
	# show the plot
	pyplot.show()
 

#%%

from sklearn.gaussian_process.kernels import RBF, ConstantKernel

def sample2selected_idx(sample):
    selected_idx = []
    for i in sample:
        selected_idx.append(cord2idx[i])
    return selected_idx


def run(points, K, radius, M, w):
    """
    Solve maximum covering location problem
    Input:
        points: input points, Numpy array in shape of [N,2]
        K: the number of sites to select
        radius: the radius of circle
        M: the number of candidate sites, which will randomly generated inside
        the ConvexHull wrapped by the polygon
    Return:
        opt_sites: locations K optimal sites, Numpy array in shape of [K,2]
        f: the optimal value of the objective function
    """
    print('----- Configurations -----')
    print('  Number of points %g' % points.shape[0])
    print('  K %g' % K)
    print('  Radius %g' % radius)
    print('  M %g' % M)
    start = time.time()
    sites = generate_candidate_sites(points, M=10)
    J = sites.shape[0]
    I = points.shape[0]
    D = distance_matrix(points, sites)
    mask1 = D <= radius
    D[mask1] = 1
    D[~mask1] = 0

    # Build model
    m = Model("mclp")
    # Add variables
    x = [m.add_var(name="x%d" % j, var_type=BINARY) for j in range(J)]
    y = [m.add_var(name="y%d" % i, var_type=BINARY) for i in range(I)]

    m.objective = maximize(xsum(w[i]*y[i] for i in range(I)))
    m += xsum(x[j] for j in range(J)) == K

    for i in range(I):
        m += xsum(x[j] for j in np.where(D[i] == 1)[0]) >= y[i]

    m.max_gap = 0.05
    m.optimize(max_seconds=300)
    end = time.time()
    print('----- Output -----')
    print('  Running time : %s seconds' % float(end-start))
    print('  Optimal coverage points: %g' % m.objective_value)
    solution = []
    for i in range(J):
        if x[i].x ==1:
            solution.append(int(x[i].name[1:]))
    opt_sites = sites[solution]

    return opt_sites,m.objective_value

def acquisition(X, Xsamples, model):
	# calculate the best surrogate score found so far
	best_list = []
	for X_elm in X:
		yhat, _ = surrogate(model, X_elm)
		best_list.append(yhat)
	best = max(best_list)
	# calculate mean and stdev via surrogate function
	# mu, std = surrogate(model, Xsamples)
	# mu = mu[:, 0]

	mu = []
	for sample in Xsamples:
		m, std = surrogate(model, sample)
		#print(m[0])
		mu.append(m[0])
	
	# calculate the probability of improvement
	probs = norm.cdf((mu - best) / (std+1E-9))
	return probs

def opt_acquisition(X, y, model):
	# random search, generate random samples
	# Xsamples = random(100, 10)
	# Xsamples = Xsamples.reshape(len(Xsamples), 1)
    Xsamples = []
    for _ in range(n_sample):
        samples = []
        for _ in range(M):
            samples.append((random.randint(0, M-1), random.randint(0, M-1)))
        Xsamples.append(sample2selected_idx(samples))
	# calculate the acquisition function for each sample
    scores = acquisition(X, Xsamples, model)

	# locate the index of the largest scores
    ix = argmax(scores)
    return Xsamples[ix]

def objective(x):
	#y = (x**2 * sin(5 * pi * x)**6.0) 
    opt_sites_org, f = run(x, K, radius, M, w)
	#y = -(x**4-x**2)+noise
	#print(f"'x': {x} | 'y': {y}")
    return f

def surrogate(model, X):
	# catch any warning generated when making a prediction
	with catch_warnings():
		# ignore generated warnings
		simplefilter("ignore")
		return model.predict([X], return_std=True)
# scores = acquisition(X, Xsamples, model)
# x = opt_acquisition(X, y, model)
# # sample the point
# actual = objective(points[x, :])

# # summarize the finding
# est, _ = surrogate(model, x)
# print(f'>x={x}, f()={est[0]}, actual={actual}')


#%%
radius = (1/88.74/1000)*500
K = 3
M = 8
n_sample = 500
n_iter = 100
X = []
y = []

lon = np.arange(123, 124, 0.1)
lat = np.arange(34, 35, 0.1)

point=[]
for x_coord in lat:
    for y_coord in lon:
        point.append(Point(np.array(y_coord), np.array(x_coord)))
points = np.array([(p.x, p.y) for p in point])

w = [random.choice(range(0, 1000)) for i in range(len(points))]


cord2idx = dict()
idx = 0
for i in range(len(lon)):
    for j in range(len(lat)):        
        cord2idx[(i, j)] = idx
        idx += 1

def sample2selected_idx(sample):
    selected_idx = []
    for i in sample:
        selected_idx.append(cord2idx[i])
    return selected_idx
#%%

Xsamples = []
for _ in range(0, n_sample):
    samples = []
    for _ in range(M):
        samples.append((random.randint(0, M-1), random.randint(0, M-1)))
    Xsamples.append(sample2selected_idx(samples))
Xsamples
#%%

for i in range(0, n_sample):
    X.append(Xsamples[i])
    y.append(objective((points[Xsamples[i], :])))

kernel = RBF() + ConstantKernel(constant_value=1)
model = GaussianProcessRegressor(kernel=kernel,  random_state=0)

# fit the model
model.fit(X, y)

#%%


# plot before hand
#plot(X, y, model)
est_list = []
# perform the optimization process
for i in range(n_iter):
	# select the next point to sample
	x = opt_acquisition(X, y, model)
	# sample the point
	actual = objective(points[x, :])
	# summarize the finding
	est, _ = surrogate(model, x)

	print(f'>f()={est[0]}, actual={actual}') # x={x}
	# add the data to the dataset
	X = vstack((X, np.array([x])))
	#print(y)
	y.append(actual) # vstack((y, actual))
	est_list.append(est)
    #print(X)
	# update the model
	model.fit(X, y)
 
# plot all samples and the final surrogate function
# plot(X, y, model)
# best result
ix = argmax(y)
#%%

print(f'Best Result: x={ix}, y={y[ix]}')
X[ix]


import matplotlib.pyplot as plt
plt.plot(y)
plt.ylabel('some numbers')
plt.show()



# %%

# %%
K = 3
M = 100
n_sample = 500
n_iter = 100
run(points, K, radius, M, w)
# %%
len(points)
# %%
X[ix]
# %%
X
# %%

# %%
run(points[X[ix]], K, radius, M, w)
# %%
