#!/usr/bin/env python
"""
Segments the pcl and returns the clusters of potential obstacles
"""
import numpy as np
from pcl_xyz import PCLtoXYZ
from sklearn.cluster import DBSCAN
import pdb

def segmentPCL(pcl, xyz, width, height, depth, R_imu, cluster_dist_thresh, min_cluster_samples):

    # xyz, ~ = PCLtoXYZ(pcl)
    xyz = xyz[xyz[:,1] > width[0]]
    xyz = xyz[xyz[:,1] < width[1]]

    xyz = xyz[xyz[:,2] > height[0]]
    xyz = xyz[xyz[:,2] < height[1]]

    xyz = xyz[xyz[:,0] < depth[1]]

    xyz = np.matmul(R_imu,np.transpose(xyz)) # apply pitch
    xyz = np.transpose(xyz)

    plane_coeffs, grnd_pnts = planeRansac(xyz)

    if len(grnd_pnts) > 0:
        xyz_grnd = xyz[grnd_pnts,:]
         #all_idx = np.arange(0,len(xyz))

        idx_non_grnd = ~grnd_pnts
         ## visualize the gound plane in rviz
         # pcl_grnd = self.XYZtoPCL(xyz_grnd, pcl.header.stamp, pcl.header.frame_id)
         # self.pub_grnd.publish(pcl_grnd)
         ## obstacle detection on this
        xyz_non_grnd = xyz[idx_non_grnd,:]


        clusters = DBSCAN(eps = cluster_dist_thresh, min_samples = min_cluster_samples,
                           metric = 'euclidean', algorithm = 'kd_tree').fit(xyz_non_grnd)
        no_clusters = len(np.unique(clusters.labels_)) - 1

         ## store the idices for each cluster in a list
        cluster_idx = [] # indices of corresponding cluster
        cluster_size = [] # cluster size(number of points)
        cluster_dist = [] # dist of the cluster from the origin
        cluster_height = []
        cluster_depth = []
        cluster_width = []
        cluster_centroid = []

        for i in range (0,no_clusters):
             cluster_idx.append(np.argwhere(clusters.labels_ == i))
             cluster_size.append(len(cluster_idx[i]))
             cluster_dist.append(np.min(xyz_non_grnd[cluster_idx[i],0]))
             cluster_depth.append(np.abs(np.max(xyz_non_grnd[cluster_idx[i],0]) - cluster_dist[i]))
             cluster_height.append(np.abs(np.max(xyz_non_grnd[cluster_idx[i],2]) - np.min(xyz_non_grnd[cluster_idx[i],2])))
             cluster_width.append(np.abs(np.max(xyz_non_grnd[cluster_idx[i],1]) - np.min(xyz_non_grnd[cluster_idx[i],1])))
             cluster_centroid.append(np.mean(xyz_non_grnd[cluster_idx[i],:], axis = 0))

        obs_idx = cluster_idx[np.argmax(cluster_size)]
        xyz_obs = xyz_non_grnd[obs_idx[:,0],:]
        dst_center = np.mean(xyz_obs, axis = 0)
        xyz_obs = np.transpose(np.matmul(np.linalg.inv(R_imu), np.transpose(xyz_obs)))
    
    return xyz_obs, dst_center



def planeRansac(pnts):

    max_pnts = 200 # max points in the plane
    bestSupport = 0
    bestPlane = []
    bestStd = np.inf
    dist_thresh = 0.05 # 0.05
    inlier_idx = []
    N = 200
    idx_= pnts[:,2] < 0 # indices of the points with height less than zero
    idx_all = np.arange(0,len(pnts))
    idx_ransac = idx_all[idx_] #

    for i in range(0,N):
        #idx  = np.random.randint(0,len(pnts),3)
        idx = np.random.choice(idx_ransac, 3, replace = False)
        p1 = pnts[idx[0],:]
        p2 = pnts[idx[1],:]
        p3 = pnts[idx[2],:]
        p12 = p1-p2
        p13 = p1-p3
        plane_coeffs = np.cross(p12,p13)

        n_ = plane_coeffs/np.linalg.norm(plane_coeffs) # unit normal to the plane
        theta = np.arccos(n_[2])

        if theta < 0.1:
            dist = np.sum(n_*(pnts - p1),axis = 1)
            idx = np.abs(dist) <= dist_thresh
            s = pnts[idx,:]
            sd = np.std(s)

            if (len(s) > bestSupport) or (len(s) == bestSupport and sd < bestStd):
                bestSupport = len(s)
                bestPlane = plane_coeffs
                bestStd = sd
                inlier_idx = idx

    return bestPlane, inlier_idx
