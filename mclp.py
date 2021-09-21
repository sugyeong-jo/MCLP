import os
import pandas as pd
import numpy as np
import random
import time

import re

from typing import Dict

import yaml
#import tqdm
import logging

from scipy.spatial import distance_matrix
from shapely.geometry import Polygon, Point
from mip import Model, xsum, maximize, BINARY


class MCLP:
    def __init__(self):
        self.data = dict()
        self.dataset = dict()

    def generate_candidate_sites(points, M=100):
        '''
        Generate M candidate sites with the convex hull of a point set
        Input:
            points: a Numpy array with shape of (N,2)
            M: the number of candidate sites to generate
        Return:
            sites: a Numpy array with shape of (M,2)
        '''
        hull = ConvexHull(points)
        polygon_points = points[hull.vertices]
        poly = Polygon(polygon_points)
        min_x, min_y, max_x, max_y = poly.bounds
        sites = []
        while len(sites) < M:
            random_point = Point(
                [random.uniform(min_x, max_x),
                random.uniform(min_y, max_y)]
                )
            if (random_point.within(poly)):
                sites.append(random_point)
        return np.array([(p.x, p.y) for p in sites])

    def generate_candidate_sites(df_result_fin, M=100):
        sites = []
        idx = np.random.choice(np.array(range(0, len(df_result_fin))), M)
        for i in range(len(idx)):
            random_point = Point(
                np.array(df_result_fin.iloc[idx]['coord_cent'])[i][0],
                np.array(df_result_fin.iloc[idx]['coord_cent'])[i][1]
                )
            sites.append(random_point)
        return np.array([(p.x, p.y) for p in sites])

    def generate_candidate_sites(df_result_fin, Weight, M=100):
        sites = []
        idx = df_result_fin.sort_values(by=Weight, ascending=False).iloc[1:M].index
        for i in range(len(idx)):
            random_point = Point(
                np.array(df_result_fin.loc[idx]['coord_cent'])[i][0],
                np.array(df_result_fin.loc[idx]['coord_cent'])[i][1]
                )
            sites.append(random_point)
        return np.array([(p.x, p.y) for p in sites])

    def run(points, K, radius, M, df_result_fin, w, Weight):
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
        sites = generate_candidate_sites(df_result_fin, Weight, M)
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
            m += xsum(x[j] for j in np.where(D[i] is 1)[0]) >= y[i]

        m.max_gap = 0.05
        m.optimize(max_seconds=300)
        end = time.time()
        print('----- Output -----')
        print('  Running time : %s seconds' % float(end-start))
        print('  Optimal coverage points: %g' % m.objective_value)

        solution = []
        for i in range(J):
            if x[i].x is 1:
                solution.append(int(x[i].name[1:]))
        opt_sites = sites[solution]
        return opt_sites, m.objective_value


