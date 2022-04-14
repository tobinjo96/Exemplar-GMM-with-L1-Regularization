import numpy as np
import multiprocessing as mp
import sharedmem
from itertools import islice
import math
import gc 
import utils
from scipy import stats
from scipy.special import logsumexp
import scipy.spatial.distance as distance
from random import sample
from sklearn.neighbors import KernelDensity, NearestNeighbors, kneighbors_graph, KDTree
from sklearn.metrics import adjusted_rand_score
from matplotlib.patches import Ellipse
from time import time

from random import sample
import matplotlib.pyplot as plt
import csv


def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))



def get_density_dists(X, bandwidth, n_jobs = None):
    if type(X) is not np.ndarray:
      raise ValueError("X must be an n x d numpy array.")
    
    k = np.round(np.log(X.shape[0])).astype(int)
    k = np.round(k*bandwidth).astype(int)
    n, d = X.shape
    if k > n:
      raise ValueError("k cannot be larger than n.")

    if n_jobs is None:
      n_jobs = mp.cpu_count() - 1
    
    #X_samp = X[sample(range(n), np.round(n/5).astype(int)), :]
    kdt = KDTree(X, metric='euclidean')
    distances, neighbors = kdt.query(X, k)
    bandwidth = distances[:, k-1].sum()/n
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(X)
    density = kde.score_samples(X)
    best_distance = np.empty((X.shape[0]))
    big_brother = np.empty((X.shape[0]))
    radius_diff = density[:, np.newaxis] - density[neighbors]
    rows, cols = np.where(radius_diff < 0)
    rows, unidx = np.unique(rows, return_index =  True)
    del radius_diff
    gc.collect()
    cols = cols[unidx]
    big_brother[rows] = neighbors[rows, cols]
    best_distance[rows] = distances[rows, cols]
    search_idx = list(np.setdiff1d(list(range(X.shape[0])), rows))
    for indx_chunk in utils.chunks(search_idx, 100):
      search_density = density[indx_chunk]
      GT_radius =  density > search_density[:, np.newaxis] 
      if any(np.sum(GT_radius, axis = 1) == 0):
        max_i = [i for i in range(GT_radius.shape[0]) if np.sum(GT_radius[i,:]) ==0]
        if len(max_i) > 1:
          for max_j in max_i[1:len(max_i)]:
            GT_radius[max_j, indx_chunk[max_i[0]]] = True
        max_i = max_i[0]
        big_brother[indx_chunk[max_i]] = indx_chunk[max_i]
        best_distance[indx_chunk[max_i]] = np.sqrt(((X - X[indx_chunk[max_i], :])**2).sum(1)).max()
        del indx_chunk[max_i]
        GT_radius = np.delete(GT_radius, max_i, 0)
      
      GT_distances = ([X[indx_chunk[i],np.newaxis], X[GT_radius[i,:],:]] for i in range(len(indx_chunk)))
      if (GT_radius.shape[0]>25):
        try:
          pool = mp.Pool(processes=n_jobs)              
          N = 25
          distances_bb = []
          i = 0
          while True:
            distance_comp = pool.map(utils.density_broad_search_star, islice(GT_distances, N))
            if distance_comp:
              distances_bb.append(distance_comp)
              i += 1
            else:
              break
          distances_bb = [dis_pair for dis_list in distances_bb for dis_pair in dis_list]
          argmin_distance = [np.argmin(l) for l in distances_bb]
          pool.terminate()
        except Exception as e:
          print("POOL ERROR: "+ e)
          pool.close()
          pool.terminate()
      else:
          distances_bb = list(map(utils.density_broad_search_star, list(GT_distances)))
          argmin_distance = [np.argmin(l) for l in distances_bb]
      
      for i in range(GT_radius.shape[0]):
        big_brother[indx_chunk[i]] = np.where(GT_radius[i,:] == 1)[0][argmin_distance[i]]
        best_distance[indx_chunk[i]] = distances_bb[i][argmin_distance[i]]
    
    return density, best_distance, big_brother, distances, neighbors

def estimate_modes(X, bandwidth = 0.1, Y = None, dens_thresh = 0.2, dist_thresh = 0.99, n_jobs = None):
  if n_jobs is None:
    n_jobs = mp.cpu_count() - 1
  density, best_distance, big_brother, distances, neighbors = get_density_dists(X, bandwidth, n_jobs)
  eps = distances[:, 1].sum()/(distances.shape[0]*X.shape[1]*10)
  density_inlier = density > np.quantile(density, dens_thresh)
  best_distance_inlier = best_distance > np.quantile(best_distance, dist_thresh)
  exemplar_idx = np.where(density_inlier*best_distance_inlier)[0]
  exemplars = X[exemplar_idx, :]
  X_rem = np.delete(X, exemplar_idx, axis = 0)
  if Y is not None:
    Y_rem = np.delete(Y, exemplar_idx, axis = 0)
    return X_rem, exemplars, Y_rem, eps
  else:
    return X_rem, exemplars

def estimate_modes_thresholds(X, bandwidth = 0.1, Y = None, dens_thresh = 0.2, dist_thresh = 0.99, n_jobs = None):
  if n_jobs is None:
    n_jobs = mp.cpu_count() - 1
  density, best_distance, big_brother, distances, neighbors = get_density_dists(X, bandwidth, n_jobs)
  eps = distances[:, 1].sum()/(distances.shape[0]*X.shape[1]*10)
  density_inlier = density > dens_thresh
  best_distance_inlier = best_distance > dist_thresh
  exemplar_idx = np.where(density_inlier*best_distance_inlier)[0]
  exemplars = X[exemplar_idx, :]
  X_rem = np.delete(X, exemplar_idx, axis = 0)
  if Y is not None:
    Y_rem = np.delete(Y, exemplar_idx, axis = 0)
    return X_rem, exemplars, Y_rem, eps
  else:
    return X_rem, exemplars


def estimate_Sigmas(X_rem, exemplars):
  n_cent = exemplars.shape[0]
  dim = X_rem.shape[1]
  n_rem = X_rem.shape[0]
  mean = X_rem.sum(0)/n_rem
  X_cent = X_rem - mean
  Sigma_hat = np.einsum('ij,ki->jk', X_cent, X_cent.T)/(n_rem-1)
  sigma = 1/(n_cent*dim)*np.trace(Sigma_hat) 
  Ss = np.stack([np.diag(np.ones(dim)*sigma) for _ in range(n_cent)])
  return Ss

# def estimate_theta(D, delta, SsLogDet):
#   n_exemplars = len(delta)
#   thetas = np.ones(n_exemplars)*np.nan
#   for i in range(n_exemplars):
#     P = D - D[:, i, np.newaxis] + (SsLogDet - SsLogDet[i])
#     Ranges = np.delete(P, i, 1)/(delta[i] - np.delete(delta, i, 0))
#     gtidx = np.where(np.delete(delta, i, 0) > delta[i])[0]
#     ltidx = np.where(np.delete(delta, i, 0) <= delta[i])[0]
#     if len(gtidx) > 0 and len(ltidx) > 0: 
#       gtmax = np.max(Ranges[:, gtidx])
#       ltmin = np.min(Ranges[:, ltidx])
#       if ltmin > gtmax: 
#         thetas[i] = gtmax
#     elif len(gtidx) > 0:
#       gtmax = np.min(Ranges[:, gtidx])
#       thetas[i] = np.max((0, gtmax))
#     elif len(ltidx) > 0:
#       ltmin = np.min(Ranges[:, ltidx])
#       thetas[i] = np.max((0, ltmin))
#   
#   return thetas[~np.isnan(thetas)].min()
# 
#   
def return_refined(R, exemplars, Ss):
  pis = R.sum(0)/R.shape[0]
  pis[pis < 0.00001] = 0
  R = R[:, pis != 0]
  new_exemplars = exemplars[pis!=0,:]
  new_Ss = Ss[pis!=0, :, :]
  pis = pis[pis != 0]
  return R, pis, new_exemplars, new_Ss

def refine_exemplars(X_rem, exemplars, Ss, theta, R = None):
  n_exemplars = exemplars.shape[0]
  D = np.zeros((X_rem.shape[0], n_exemplars))
  if R is None:
    R = np.ones((D.shape[0], n_exemplars))/n_exemplars
  
  dim = X_rem.shape[1]
  for j in range(n_exemplars):
    D[:,j,np.newaxis] = distance.cdist(X_rem, exemplars[j,:][np.newaxis], metric='mahalanobis', VI=np.linalg.inv(Ss[j, :, :]))
  
  pis = R.sum(0)/R.shape[0]
  delta = np.zeros(n_exemplars)
  MLD = np.zeros((n_exemplars, n_exemplars))
  for i in range(n_exemplars):
    MLD[i,:] = distance.cdist(exemplars, exemplars[i,:][np.newaxis], metric='mahalanobis', VI=np.linalg.inv(Ss[i, :, :])).T
    MLD[i, :] *= -1/2
    MLD[i, :] = 2*stats.norm.cdf(MLD[i, :])
  
  delta = np.array([np.delete(MLD[:, i], i).max() for i in range(n_exemplars)])
  SsLogDet = np.array([np.log(np.linalg.det(Ss[i])) for i in range(n_exemplars) ])/R.shape[0]
  Y = theta*delta
  Ob = D + SsLogDet + Y 
  s_min = Ob.argmin(1)
  R = np.zeros(R.shape)
  R[range(R.shape[0]), s_min] = 1
  return return_refined(R, exemplars, Ss)

def recursive_function(theta, step, prev_n_cent, X_rem, exemplars_in, Ss_in, R_in):
  R, pis, exemplars, Ss = refine_exemplars(X_rem, exemplars_in, Ss_in, theta, R_in)
  n_cent = exemplars.shape[0]
  if n_cent == prev_n_cent:
    theta += step
    return recursive_function(theta, step, prev_n_cent,X_rem, exemplars, Ss, R)
  elif n_cent < prev_n_cent -1:
    theta -= step
    step /= 2
    return recursive_function(theta, step, prev_n_cent,X_rem, exemplars_in, Ss_in, R_in)
  else:
    return R, pis, exemplars, Ss, theta


def update_R(Pdfs, X_rem, exemplars, Ss, pis):
  if Pdfs is None:
    n_cent = len(pis)
    Pdfs = np.stack([pis[i]*stats.multivariate_normal(exemplars[i, :], Ss[i, :, :]).pdf(X_rem) for i in range(n_cent)]).T
  
  R_new = Pdfs/Pdfs.sum(1)[:, np.newaxis]
  return R_new

def update_pis(R_new):
  return R_new.sum(0)/R_new.shape[0]

def update_Ss(X_rem, exemplars, Ss, R_new, eps):
  n = X_rem.shape[0]
  n_cent = R_new.shape[1]
  for j in range(n_cent):
    X_cent = X_rem - exemplars[j, :]
    Ss[j, :, :] = np.einsum('ij,ki->jk', R_new[:, j, np.newaxis]*X_cent, X_cent.T)/R_new[:, j].sum()
    Ss[j, :, :] += np.diag(eps*np.ones(Ss[j].shape[1]))
  return Ss

def remove_singular(exemplars, Ss, pis, R_new):
  n_cent = exemplars.shape[0]
  rm_list = []
  for i in range(n_cent):
    try:
      stats.multivariate_normal(exemplars[i, :], Ss[i, :, :]).pdf(exemplars[i, :])
    except Exception as e:
      rm_list.append(i)
  if len(rm_list) > 0:
    exemplars = np.delete(exemplars, rm_list, 0)
    Ss = np.delete(Ss, rm_list, 0)
    pis = np.delete(pis, rm_list, 0)
    R_new = np.delete(R_new, rm_list, 1)
  
  return exemplars, Ss, pis, R_new

def check_converge(ll_new, ll_old, iters, tol):
  return ll_new-ll_old < tol or iters > 1000

def loglikelihood(X_rem, R_new, exemplars, Ss, pis):
  n_cent = R_new.shape[1]
  Pdfs = np.stack([pis[j]*stats.multivariate_normal(exemplars[j, :], Ss[j, :, :]).pdf(X_rem) for j in range(n_cent)]).T
  tmp = Pdfs.sum(1)
  ll = np.log(tmp).sum() 
  return ll, Pdfs

def remove_nans(X_rem, Y_rem, R_new):
  idx = np.where(np.isnan(R_new))[0]
  inidx =np.array([i for i in range(X_rem.shape[0]) if i not in idx])
  X_rem = X_rem[inidx, :]
  R_new = R_new[inidx, :]
  Y_rem = Y_rem[inidx]
  return X_rem, Y_rem, R_new

def EM(X_rem, Y_rem, exemplars, Ss, R, eps):
  R_old = R
  pis = R.sum(0)/R.shape[0]
  ll_old = -1e32
  iters = 0
  tol = 1e-3
  Pdfs = None
  for _ in range(60):
  #while True:
    R_new = update_R(Pdfs, X_rem, exemplars, Ss, pis)
    X_rem, Y_rem, R_new = remove_nans(X_rem, Y_rem, R_new)
    pis = update_pis(R_new)
    Ss = update_Ss(X_rem, exemplars, Ss, R_new, eps)
    exemplars, Ss, pis, R_new = remove_singular(exemplars, Ss, pis, R_new)
    ll_new, Pdfs = loglikelihood(X_rem, R_new, exemplars, Ss, pis)
    if np.abs(ll_old - ll_new) < tol:
      break
    else:
      ll_old = ll_new
  R = R_new
  return pis, exemplars, Ss, ll_new, R, X_rem, Y_rem

def compute_ICL(X_rem, Ss, R, loglikelihood):
  dim = Ss.shape[1]
  n_exemplars = Ss.shape[0]
  n = X_rem.shape[0]
  n_param = n_exemplars*((dim*dim -dim)/2 + dim + 1) - 1
  C = np.zeros((n, n_exemplars))
  C[range(n), R.argmax(1)] = 1
  ent_z = np.log(R)
  ent_z[ent_z == -np.inf] = 0
  ICL = 2 * np.sum(loglikelihood) - n_param * np.log(n) + 2*np.sum(C * ent_z)
  return ICL


def main(X, bandwidth, Y = None, dens_thresh = 0.2, dist_thresh = 0.99, n_jobs = None, step = 1, Outfile = "Results.csv", **kwargs):
  t1 = time()
  if Y is None:
    X_rem, exemplars = estimate_modes(X, bandwidth, Y, dens_thresh, dist_thresh, n_jobs = None)
  else:
    X_rem, exemplars, Y_rem, eps = estimate_modes(X, bandwidth, Y, dens_thresh, dist_thresh, n_jobs = None)
    nc_min = 2
  
  Ss = estimate_Sigmas(X_rem, exemplars)
  n_cent = exemplars.shape[0]
  n_rem = X_rem.shape[0]
  R = np.ones((n_rem, n_cent))/n_cent
  pis, exemplars, Ss, ll_new, R, X_rem, Y_rem = EM(X_rem, Y_rem, exemplars, Ss, R, eps)
  R, pis, exemplars, Ss = refine_exemplars(X_rem, exemplars, Ss, 0)
  ICL_old = compute_ICL(X_rem, Ss, R, ll_new)
  model = [pis, exemplars, Ss]
  if Y_rem is not None:
    model.append(adjusted_rand_score(Y_rem.astype(int), R.argmax(1).astype(int)))
  try:
    while exemplars.shape[0] > nc_min:
      prev_n_cent = exemplars.shape[0]
      R_new, pis_new, exemplars_new, Ss_new, theta = recursive_function(0, step, prev_n_cent, X_rem, exemplars, Ss, R)
      pis, exemplars, Ss, ll_new, R, X_rem, Y_rem = EM(X_rem, Y_rem, exemplars_new, Ss_new, R_new, eps)
      t2 = time()
      ICL_new = compute_ICL(X_rem, Ss, R, ll_new)
      if ICL_new > ICL_old:
        model = [pis, exemplars, Ss]
        if Y_rem is not None:
          model.append(adjusted_rand_score(Y_rem.astype(int), R.argmax(1).astype(int)), ICL_new, exemplars.shape[0])
        
        ICL_old = ICL_new
      
      step = theta/5
    
    with open(Outfile, 'a') as fd:
      writer = csv.writer(fd)
      writer.writerow([X.shape[0], X.shape[1], kwargs.get("K"),kwargs.get("Omega"),kwargs.get("v"), bandwidth, dens_thresh, dist_thresh, model[5], ICL_old, model[3]])
  except Exception as e:
    with open(Outfile, 'a') as fd:
      writer = csv.writer(fd)
      writer.writerow([X.shape[0], X.shape[1], kwargs.get("K"),kwargs.get("Omega"),kwargs.get("v"), bandwidth, dens_thresh, dist_thresh, model[5], ICL_old, model[3]])
    
  
