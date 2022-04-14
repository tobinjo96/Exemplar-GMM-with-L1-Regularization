import numpy as np
import EGMM
import pyreadr
import csv
from sklearn.metrics import adjusted_rand_score
from time import time

from scipy import stats

             
n = 10000
for dim in [2]:
  for K in [ 10]:
    for Omega in [0]:
      for v in range(1, 11):
        for dens_thresh in [0.2]:
          for dist_thresh in [0.985]:
            for bandwidth in [1]:
              try:
                Data = np.genfromtxt("/Data/Simulated/Train_n" + str(n) + "_K" + str(K) + "_d" + str(dim) + "_Omega" + str(Omega) + "_v" + str(v) + ".csv", delimiter = ",")
                X = Data[:, range(dim)]
                Y = Data[:, dim]
                t1 = time()
                X_rem, exemplars, Y_rem,eps = EGMM.estimate_modes(X, bandwidth, Y, dens_thresh = dens_thresh, dist_thresh = dist_thresh, n_jobs = None)
                nc_min = 2
                Ss = EGMM.estimate_Sigmas(X_rem, exemplars)
                
                n_cent = exemplars.shape[0]
                n_rem = X_rem.shape[0]
                R = np.ones((n_rem, n_cent))/n_cent
                
                pis, exemplars, Ss, ll_new, R, X_rem, Y_rem= EGMM.EM(X_rem, Y_rem, exemplars, Ss, R, eps)
                R, pis, exemplars, Ss = EGMM.refine_exemplars(X_rem, exemplars, Ss, 0)
                theta = 0
                step = 1
                while exemplars.shape[0] > nc_min:
                  prev_n_cent = exemplars.shape[0]
                  R_new, pis_new, exemplars_new, Ss_new, theta = EGMM.recursive_function(0, step, prev_n_cent, X_rem, exemplars, Ss, R)
                  pis, exemplars, Ss, ll_new, R, X_rem, Y_rem = EGMM.EM(X_rem,Y_rem, exemplars_new, Ss_new, R_new, eps)
                  R, pis, exemplars, Ss = EGMM.refine_exemplars(X_rem, exemplars, Ss, 0)
                  t2 = time()
                  with open('EGMM_Assess.csv', 'a') as fd:
                    writer = csv.writer(fd)
                    writer.writerow([X.shape[0], X.shape[1], K, Omega, v, bandwidth, dens_thresh, dist_thresh, theta, t2-t1, R.shape[1],adjusted_rand_score(Y_rem.astype(int), R.argmax(1).astype(int)), EGMM.compute_ICL(X_rem, Ss, R, ll_new)])
                  
                  step = theta/5
              
              except Exception as e:
                with open("EGMM_Assess.csv", 'a') as fd:
                    writer = csv.writer(fd)
                    writer.writerow([X.shape[0], X.shape[1], K, Omega, v, bandwidth, dens_thresh, dist_thresh, e])


  
  
  
