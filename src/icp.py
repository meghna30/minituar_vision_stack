#!/usr/bin/env python

import rospy
import numpy as np
import rospkg
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tf
import time
import pcl
from open3d import *
import rosbag
import copy
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from sensor_msgs import point_cloud2
from geometry_msgs.msg import Point
import ros_numpy

# class ICP():
#     def __init__(self):
#         self.first_pcl = True
#         rospy.init_node("ICP")
#         #self.camera_params = rospy.wait_message('camera_params', CameraInfo)
#         self.source_pcl = []
#         self.dest_pcl = []
#         self.first_pcd = []
#         self.final_pcd = []
#         self.params = [0,0,0,0,0,0] # r,p,y,tx,ty,tz
#         self.pub_filter_pcl = rospy.Publisher('filter_Pointcloud', PointCloud2, queue_size = 10)
#         self.threshold = 0.02
#         self.trans_init = np.asarray([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0],[0.0,0.0,0.0,1.0]])
#         self.iter = -1
#         self.T = np.asarray([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]])
#         self.debug_pcl = []
#     def extract_pcl(self):
#         #plc = rospy.Subscriber('/zed/point_cloud/cloud_registered', PointCloud2, self.doICP)
#         #rospy.spin()
#         #loop through the bag file
#
#         bag = rosbag.Bag('/home/meghna/zed_box_1.bag')
#         for topic, msg, t in bag.read_messages(topics='/zed/point_cloud/cloud_registered'):
#             self.doICP(msg)
#         bag.close()

def doICP(src, dst):

        # convert pcl to numpy array
        #tic = time.time()
        #xyz = list(point_cloud2.read_points(pcloud, skip_nans=True, field_names=("x","y","z")))
        #xyz = np.asarray(xyz, np.float32)
        #toc = time.time() - tic
    # xyz  = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(pcloud,remove_nans=True)
    # xyz = xyz[xyz[:,0] < 1.0] # cut of ahead at 1m
    # xyz = xyz[xyz[:,1] > -0.5]
    # xyz = xyz[xyz[:,1] < 0.5]
        #toc = time.time()-tic
        # convert numpy to open3d pointcloud
    p_src = pcl.PointCloud(np.array(src,dtype=np.float32))
    p_dst = pcl.PointCloud(np.array(dst,dtype=np.float32))
    # voxel_filter = p.make_voxel_grid_filter()
    # voxel_filter.set_leaf_size(0.05,0.05,0.05)
    # p = voxel_filter.filter()
    pcd_src = PointCloud()
    pcd_src.points = Vector3dVector(src)
    # p = voxel_filter.filter()
    pcd_dst = PointCloud()
    pcd_dst.points = Vector3dVector(dst)

        #pcd = voxel_down_sample(pcd, voxel_size = 0.05)
        #draw_geometries([pcd])
        #pcd = voxel_down_sample(pcd, voxel_size = 0.05)

        # if self.first_pcl:
        #     self.source_pcl = p
        #     #self.debug_pcl = p
        #     self.first_pcl = False
        #     # for plotting
        #     #pcd = PointCloud()
        #     #pcd.points = Vector3dVector(xyz)
        #     #self.source_pcd = pcd
    # else:
    #         tic = time.time()
    #         self.iter += 1
    #         self.dest_pcl = p

    icp = p_src.make_IterativeClosestPoint()
    converged, transf , estimate, fitness = icp.icp(p_src, p_dst, max_iter = 1000)

            #icp_debug = self.debug_pcl.make_IterativeClosestPoint()
            #debug_c, debug_t, debug_est, debug_f = icp_debug.icp(self.debug_pcl, self.dest_pcl)



            #if self.iter%25 == 0:
            #pcd = PointCloud()
            #pcd.points = Vector3dVector(xyz)
            #self.final_pcd = pcd
    print(converged)
    return transf[0:3,0:3], transf[0:3,3]
    #debug_plot(pcd_src, pcd_dst, transf)
            #self.debug_plot(self.source_pcd, self.final_pcd, debug_t)

            #self.source_pcl = self.dest_pcl
    # toc = time.time() - tic
            #print('calc', transf)
            #print('debug', debug_t)


    # print("Convert and transform PCL time={:.1f}ms".format(toc*1000))

def debug_plot(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1,0,0])
    target_temp.paint_uniform_color([0,0,1])
    source_tmp.transform(transformation)
    draw_geometries([source_temp, target_temp])




# if __name__== '__main__':
#     P = ICP()
#     P.extract_pcl()
