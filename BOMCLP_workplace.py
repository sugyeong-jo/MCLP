#%%
from numpy.random import randint
from numpy.random import rand
import random

import pandas as pd
#%%
# xy
xy_coord = []
xt_idx = []
for i in range(250*250):
    xt_idx.append(i)
for i in range(250):
    for j in range(250):
        xy_coord.append([i,j])
# %%
xy = pd.concat([pd.DataFrame(xy_coord), pd.DataFrame(xt_idx)], axis=1)
xy.columns = ['x', 'y', 'idx']
xy_dict = {}
for index, info in xy.iterrows():
    xy_dict[info['x'], info['y']] = info['idx']

# %%
x = random.sample(range(0, 250), 5)
y = random.sample(range(0, 250), 5)


for i in range(5):
    print(xy_dict[x[i],y[i]])
# %%
