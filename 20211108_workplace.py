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

w = [random.choice(range(0, 100)) for i in range(len(points))]

#%%
radius = (1/88.74/1000)*500
K = 20
M = 500
selected_idx = random.sample(range(0, len(points)), k=M)
candidate_sites = points[selected_idx,:]
pd_cand_sites = pd.DataFrame(candidate_sites)
opt_sites_org,f = run(candidate_sites, K, radius, M, w)
# %%
selected_idx
# %%
