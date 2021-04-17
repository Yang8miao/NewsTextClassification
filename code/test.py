import numpy as np
from sklearn.model_selection import KFold,StratifiedKFold

X=np.array([
    ['a', 2, 122.21, 4],
    ['b', 3, 132.12, 14],
    ['c', 31, 155.33, 24],
    ['d', 12, 143.93, 34],
    ['c', 32, 124.31, 44],
    ['a', 1, 151.11, 54],
    ['b', 11, 112.33, 64],
    ['b', 21, 137.82, 74]
])

y=np.array([1,1,0,0,1,1,0,0])
sfolder = StratifiedKFold(n_splits=3,random_state=0,shuffle=False)
floder = KFold(n_splits=3,random_state=0,shuffle=False)

for train_idx,val_idx in sfolder.split(X,y):
    print('Train: %s | Val: %s' % (train_idx, val_idx))
    print(" ")
    print("train_x: ",X[train_idx])
    print("train_y: ",y[train_idx])
    print("val_x:", X[val_idx])
    print("val_y:", y[val_idx])

    print("\n\n")

print("\n\n  --------  \n\n")

for train_idx, val_idx in floder.split(X,y):
    print('Train: %s | Val: %s' % (train_idx, val_idx))
    print(" ")