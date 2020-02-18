#!/usr/bin/env python
"""
initlializes the object to be tracked and then tracks it
main file to be run
"""

import rospy
import numpy as np
import rospkg
import cv2
import ros_numpy
import message_filters
from cv_bridge import CvBridge, CvBridgeError

import tf
from pcl_xyz import *
from pcl_segmentation import *

from procrustes_ransac import ProcrustesRansac

import pdb

from sensor_msgs.msg import Image, CameraInfo, PointCloud2, Imu, PointField
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry
from sensor_msgs import point_cloud2
from geometry_msgs.msg import Pose

from sklearn.cluster import DBSCAN

class Tracker():

    def __init__(self):
        rospy.init_node("Tracker")
        self.listener = tf.TransformListener()
        ## stuff that needs to be published
        self.pub_img = rospy.Publisher('obs_image', Image, queue_size = 10)
        self.pub_pcl = rospy.Publisher('pcl', PointCloud2, queue_size = 10)
        self.pub_poke_mode = rospy.Publisher('/minitaur/poke', Bool, queue_size = 10)

        ##
        self.bridge = CvBridge()
        ###
        ## pointcloud segmentation YAML
        self.width = [-1.0,1.0]
        self.height = [-2,0.5]
        self.depth = [0,3]
        ## clustering               YAML
        self.cluster_dist_thresh = 0.03 # 0.05
        self.min_cluster_samples = 20


        self.dist_thresh = 0.3   ##

        self.obs_init = False ## flag to check if obs has been initialised
        self.track_mode = True  ## flag to chack if tracking mode is enabled.
        ## tf transformations
        self.R_leftCamOpt_to_leftCam =  tf.transformations.euler_matrix(1.57, -1.57, 0)
        self.R_I = tf.transformations.euler_matrix(0,0,0)

        self.R_odom_leftCamOpt = tf.transformations.euler_matrix(-1.57,-0.0,-1.57)
        self.T_odom_leftCamOpt = np.asarray([0.00, 0.030, 0.00,1])
        self.odom_to_leftCamOpt = self.R_odom_leftCamOpt
        self.odom_to_leftCamOpt[:,3] = self.T_odom_leftCamOpt

        self.obs_tracker = cv2.TrackerMIL_create()

        self.img_height = int(720*0.2)
        self.img_width = int(1280*0.2)

        self.orb = cv2.ORB_create()

        FLANN_INDEX_KDTREE = 0
        #FLANN_INDEX_LSH = 6
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        # index_params= dict(algorithm = FLANN_INDEX_LSH,
        #           table_number = 12, # 12
        #           key_size = 20,     # 20
        #           multi_probe_level = 2) #2
        search_params = dict(checks = 25) #50
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        self.min_matches = 3
        self.pub_poke_flag = False
        self.H_odom = np.zeros([4,4])
        self.poke_done = False


    def GetMsgs(self):

        self.camera_info = rospy.wait_for_message('/zed/zed_node/depth/camera_info', CameraInfo)
        rospy.loginfo('received camera info')
        imu_data = message_filters.Subscriber("/zed/zed_node/imu/data", Imu)
        rospy.loginfo('received imu data')
        pcl = message_filters.Subscriber("/zed/zed_node/point_cloud/cloud_registered", PointCloud2)
        rospy.loginfo('received pointcloud')
        depth_image = message_filters.Subscriber("/zed/zed_node/depth/depth_registered", Image)
        rospy.loginfo('received depth image')
        rgb_img = message_filters.Subscriber("/zed/zed_node/rgb/image_rect_color", Image)
        rospy.loginfo('received rgb image')
        odom = message_filters.Subscriber("/zed/zed_node/odom", Odometry)
        ts = message_filters.ApproximateTimeSynchronizer([pcl,rgb_img, odom, imu_data],100,0.1)
        rospy.loginfo('messages synchronized')
        ts.registerCallback(self.callback)
        rospy.spin()

    def poke_callback(self, data):
        self.poke_done = data.data

    def callback(self, pcl, rgb_img, odom, imu):
        rospy.Subscriber("/poke_done",Bool,self.poke_callback)


        self.stamp = pcl.header.stamp
        self.frame_id = pcl.header.frame_id

        T_odom = np.asarray([odom.pose.pose.position.x,
                              odom.pose.pose.position.y,
                              odom.pose.pose.position.z])

        R_odom = tf.transformations.quaternion_matrix(np.asarray([odom.pose.pose.orientation.x,
                                                                   odom.pose.pose.orientation.y,
                                                                   odom.pose.pose.orientation.z,
                                                                   odom.pose.pose.orientation.w]))
        self.H_odom = R_odom
        self.H_odom[0:3,3] = T_odom


        quat = imu.orientation
        imu_euler = tf.transformations.euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])


        R_imu = np.asarray([[np.cos(imu_euler[1]),0.0,np.sin(imu_euler[1])],
                        [0.0,1.0,0.0],
                        [-np.sin(imu_euler[1]),0.0, np.cos(imu_euler[1])]])

        if self.track_mode:
            self.pub_poke_mode.publish(Bool(data=False))
            if not self.obs_init:
            ## segment pcl and initialize object
                xyz, xyz_img = PCLtoXYZ(pcl, self.R_leftCamOpt_to_leftCam, self.img_height,self.img_width)
                xyz_obs, dst_center = segmentPCL(pcl,xyz, self.width, self.height, self.depth, R_imu, self.cluster_dist_thresh, self.min_cluster_samples)
                dst_center = dst_center[0]
                print(dst_center)
                ## xyz to image
                ## obs_pixels
                cv_rgb_img = self.bridge.imgmsg_to_cv2(rgb_img, "bgr8")
                pixel_coords_x, pixel_coords_y = self.XYZtoPixels(xyz_obs, self.R_leftCamOpt_to_leftCam)

            ## track this object if lose track of the object reinitialize
            ## initialize tracker
                b_box = (np.min(pixel_coords_y), np.min(pixel_coords_x), np.max(pixel_coords_y) - np.min(pixel_coords_y), np.max(pixel_coords_x) - np.min(pixel_coords_x))

                ## draw bounding box on image
                p1 = (int(b_box[0]), int(b_box[1]))
                p2 = (int(b_box[0] + b_box[2]), int(b_box[1] + b_box[3]))
                # cv2.rectangle(cv_rgb_img,p1,p2,(200,0,0))
                # # img_msg = self.publishImages(cv_rgb_img)
                # # self.pub_img.publish(img_msg)
                #
                ok = self.obs_tracker.init(cv_rgb_img, b_box)
                if ok:
                    self.obs_init = True ## obstacle initialised

            else:
                cv_rgb_img = self.bridge.imgmsg_to_cv2(rgb_img, "bgr8")
                ok, b_box = self.obs_tracker.update(cv_rgb_img)
                if ok:
                    p1 = (int(b_box[0]), int(b_box[1]))
                    p2 = (int(b_box[0] + b_box[2]), int(b_box[1] + b_box[3]))
                    ##
                    #cv2.rectangle(cv_rgb_img,p1,p2,(200,0,0))
                    #img_msg = self.publishImages(cv_rgb_img)
                    #self.pub_img.publish(img_msg)
                    ##
                    # distance from the camera center
                    x_pixels = np.arange(p1[1], p2[1])
                    y_pixels = np.arange(p1[0], p2[0])
                    xx_pixels,yy_pixels = np.meshgrid(x_pixels, y_pixels)
                    xx_pixels.reshape(-1,1)
                    yy_pixels.reshape(-1,1)
                    blah, xyz_img = PCLtoXYZ(pcl, self.R_leftCamOpt_to_leftCam, self.img_height,self.img_width)
                    xyz_obs_img = xyz_img[xx_pixels, yy_pixels,:]  ##
                    xyz_obs_bbox = xyz_obs_img.reshape([-1,3])
                    xyz_obs_bbox = xyz_obs_bbox[~np.isnan(xyz_obs_bbox).any(1)]
                    # msg_pcl = XYZtoPCL(xyz_obs_bbox,self.stamp, self.camera_info.header.frame_id)
                    # self.pub_pcl.publish(msg_pcl)

                    ## clustering ?
                    clusters = DBSCAN(eps = self.cluster_dist_thresh, min_samples = self.min_cluster_samples,
                                      metric = 'euclidean', algorithm = 'kd_tree').fit(xyz_obs_bbox)
                    no_clusters = len(np.unique(clusters.labels_)) - 1
                    cluster_size = []
                    cluster_idx = []
                    for i in range(0, no_clusters):
                        cluster_idx.append(np.argwhere(clusters.labels_ == i))
                        cluster_size.append(len(cluster_idx[i]))

                    obs_idx = cluster_idx[np.argmax(cluster_size)]
                    xyz_obs = xyz_obs_bbox[obs_idx[:,0],:]
                    msg_pcl = XYZtoPCL(xyz_obs,self.stamp, self.camera_info.header.frame_id)
                    self.pub_pcl.publish(msg_pcl)
                    dst_center = np.mean(xyz_obs, axis = 0)[2]
                    print(dst_center)


                if not ok:
                    self.obs_init = False


        #once within threshold distance initiate poking behavior
            if np.abs(dst_center) < self.dist_thresh:
            ## stop tracking start poking

                self.track_mode = False
                self.pub_poke_mode.publish(Bool(data=True))

                cv_rgb_img = self.bridge.imgmsg_to_cv2(rgb_img, "bgr8")
                tracking_rgb_img = np.zeros([self.img_height,self.img_width,3], dtype=np.uint8)

                pixel_coords_x, pixel_coords_y = self.XYZtoPixels(xyz_obs, self.R_I)
                tracking_rgb_img[pixel_coords_x,pixel_coords_y] = cv_rgb_img[pixel_coords_x,pixel_coords_y]
                self.src_img = tracking_rgb_img
                blah, self.xyz_img_src = PCLtoXYZ(pcl, self.R_leftCamOpt_to_leftCam, self.img_height,self.img_width)
                # img_msg = self.publishImages(self.src_img)
                # self.pub_img.publish(img_msg)
                print("done storing src informaation")



        # ## use the frame before the poke and right after the poke to calculate translation and rotation
        # ### afer done with the poking

        if self.track_mode == False and self.poke_done == True:

            cv_rgb_img = self.bridge.imgmsg_to_cv2(rgb_img, "bgr8")
            ok, b_box = self.obs_tracker.update(cv_rgb_img)

            p1 = (int(b_box[0]), int(b_box[1]))
            p2 = (int(b_box[0] + b_box[2]), int(b_box[1] + b_box[3]))

            x_pixels = np.arange(p1[1], p2[1])
            y_pixels = np.arange(p1[0], p2[0])
            xx_pixels,yy_pixels = np.meshgrid(x_pixels, y_pixels)
            xx_pixels.reshape(-1,1)
            yy_pixels.reshape(-1,1)
            blah, xyz_img = PCLtoXYZ(pcl, self.R_leftCamOpt_to_leftCam, self.img_height,self.img_width)
            xyz_obs_img = xyz_img[xx_pixels, yy_pixels,:]  ##
            xyz_obs_bbox = xyz_obs_img.reshape([-1,3])
            xyz_obs_bbox = xyz_obs_bbox[~np.isnan(xyz_obs_bbox).any(1)]
                        # msg_pcl = XYZtoPCL(xyz_obs_bbox,self.stamp, self.camera_info.header.frame_id)
                        # self.pub_pcl.publish(msg_pcl)

                        ## clustering ?
            clusters = DBSCAN(eps = self.cluster_dist_thresh, min_samples = self.min_cluster_samples,
                                          metric = 'euclidean', algorithm = 'kd_tree').fit(xyz_obs_bbox)
            no_clusters = len(np.unique(clusters.labels_)) - 1
            cluster_size = []
            cluster_idx = []
            for i in range(0, no_clusters):
                cluster_idx.append(np.argwhere(clusters.labels_ == i))
                cluster_size.append(len(cluster_idx[i]))

            obs_idx = cluster_idx[np.argmax(cluster_size)]
            xyz_obs = xyz_obs_bbox[obs_idx[:,0],:]

            tracking_rgb_img = np.zeros([self.img_height,self.img_width,3], dtype=np.uint8)

            pixel_coords_x, pixel_coords_y = self.XYZtoPixels(xyz_obs, self.R_I)   ## xyz_obs is in the left opt frame check XYZtoPixels
            tracking_rgb_img[pixel_coords_x,pixel_coords_y] = cv_rgb_img[pixel_coords_x, pixel_coords_y]
            self.dest_img = tracking_rgb_img
            blah, self.xyz_img_dst = PCLtoXYZ(pcl, self.R_leftCamOpt_to_leftCam, self.img_height,self.img_width)
            img_msg = self.publishImages(self.dest_img)
            self.pub_img.publish(img_msg)


            self.feature_matching()



    def feature_matching(self):
        kp1, des1 = self.orb.detectAndCompute(self.src_img, None)
        kp2, des2 = self.orb.detectAndCompute(self.dest_img, None)

        if len(kp1) > 4 and len(kp2) > 4:
            matches = self.flann.knnMatch(np.asarray(des1,np.float32),np.asarray(des2,np.float32),k= 2)
            good = []
            for m,n in matches:
                if m.distance < 0.75*n.distance:
                    good.append(m)

            if len(good) > self.min_matches:
                src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

                src_idx = np.reshape(np.round(src_pts),[-1,2]).astype(int)
                dst_idx = np.reshape(np.round(dst_pts),[-1,2]).astype(int)


                src_3d_pts = self.xyz_img_src[src_idx[:,1],src_idx[:,0],:]
                src_nan_idx = np.isnan(src_3d_pts)
                src_nan_idx = np.logical_or(src_nan_idx[:,0],src_nan_idx[:,1],src_nan_idx[:,2])

                dst_3d_pts = self.xyz_img_dst[dst_idx[:,1],dst_idx[:,0],:]
                dst_nan_idx = np.isnan(dst_3d_pts)
                dst_nan_idx = np.logical_or(dst_nan_idx[:,0],dst_nan_idx[:,1],dst_nan_idx[:,2])
                non_nan_idx = np.logical_or(src_nan_idx, dst_nan_idx)
                src_3d_pts = src_3d_pts[~non_nan_idx,:]
                dst_3d_pts = dst_3d_pts[~non_nan_idx,:]

                self.ExtractTransformation(src_3d_pts,dst_3d_pts)

    def ExtractTransformation(self, src_pnts, dst_pnts):

        src_pnts_ =  np.hstack([src_pnts,np.ones([len(src_pnts),1])])
        src_pnts_odom = np.transpose(np.matmul(self.H_odom,np.matmul(self.odom_to_leftCamOpt,np.transpose(src_pnts_))))[:,0:3]

        dst_pnts_ =  np.hstack([dst_pnts,np.ones([len(dst_pnts),1])])
        dst_pnts_odom = np.transpose(np.matmul(self.H_odom,np.matmul(self.odom_to_leftCamOpt,np.transpose(dst_pnts_))))[:,0:3]

        src_centroid = np.mean(src_pnts_odom, axis=0)
        dst_centroid = np.mean(dst_pnts_odom, axis=0)

        R , t = ProcrustesRansac(src_pnts_odom,dst_pnts_odom)
        t = src_centroid - dst_centroid

        pdb.set_trace()
        print(R, t)



    def publishImages(self, img):
        try:
            img_msg = self.bridge.cv2_to_imgmsg(img,encoding="bgr8")
        except CvBridgeError as e:
            print(e)
        img_msg.header.frame_id = self.camera_info.header.frame_id
        img_msg.header.stamp = self.stamp
        return img_msg

    def XYZtoPixels(self,xyz, R):

        K = np.resize(self.camera_info.K,[3,3])
        P = np.resize(self.camera_info.P,[3,4])
        #R =  tf.transformations.euler_matrix(1.57, -1.57, 0)

        xyz_ = np.hstack([xyz,np.ones([len(xyz),1])])
        xyz_t = np.matmul(R,np.transpose(xyz_))
        pixel_coords = np.matmul(P,xyz_t)
        pixel_coords_y = np.round(pixel_coords[0,:]/pixel_coords[2,:])
        pixel_coords_x = np.round(pixel_coords[1,:]/pixel_coords[2,:])

        pixel_coords_y = np.clip(pixel_coords_y.astype(int),0,self.img_width-1)
        pixel_coords_x = np.clip(pixel_coords_x.astype(int),0,self.img_height-1)

        return pixel_coords_x, pixel_coords_y




if __name__=="__main__":
    T = Tracker()
    T.GetMsgs()
