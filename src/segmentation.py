#!/usr/bin/env python
"""
1. Segment the PCL
2  Ground plane segmentaton - use the imu and distance from the ground
3. Remove the ground plane from the point cloud
4. Cluster the remaining depth cloud the detect obstacles
5. Project the 3d obstacles back to the the 2d image (convex hull)
"""
import rospy
import numpy as np
import rospkg
import cv2
#import ros_numpy
import ros_numpy
import message_filters
from cv_bridge import CvBridge, CvBridgeError
import pdb
import tf
import time
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, Imu, PointField
from sensor_msgs import point_cloud2

from sklearn.cluster import DBSCAN


class Segmentation():

    def __init__(self):
        rospy.init_node("Segmentation")
        self.listener = tf.TransformListener()
        self.pub_pcl = rospy.Publisher('pcl', PointCloud2, queue_size = 10)
        self.pub_grnd = rospy.Publisher('grnd_pnts',PointCloud2, queue_size = 10)
        self.pub_non_grnd = rospy.Publisher('non_grnd_pnts', PointCloud2, queue_size = 10)
        self.pub_obs = rospy.Publisher('obs', PointCloud2, queue_size = 10)
        self.pub_img = rospy.Publisher('obs_image', Image, queue_size = 10)
        self.pub_rgb_img = rospy.Publisher('obs_rgb_image', Image, queue_size = 10)
        self.pub_matches_img = rospy.Publisher('matches_image', Image, queue_size = 10)

        self.bridge = CvBridge()
        ## pointcloud segmentation
        self.width = [-0.5,0.5]
        self.height = [-2,5]
        self.depth = [0,2]
        ## clustering
        self.cluster_dist_thresh = 0.1
        self.min_cluster_samples = 50
        ## feature detection and matching
        self.do_matching = True
        self.first_detection = True
        self.min_matches = 4
        self.source_img = []
        self.dest_img = []
        self.img_height = 72*2
        self.img_width = 128*2
        ## rotation matrix from camera_left_optical_frame to camera_left_frame
        #(trans, rot) = self.listener.lookupTransform('/zed_left_camera_optical_frame','/zed_left_camera_frame', rospy.Time(0))
        self.R_leftCamOpt_to_leftCam =  tf.transformations.euler_matrix(1.57, -1.57, 0)
        self.R_odom_leftCamOpt = tf.transformations.euler_matrix(-1.57,-0.0,-1.57)
        self.T_odom_leftCamOpt = np.asarray([0.00, 0.030, 0.00,1])
        self.odom_to_leftCamOpt = self.R_odom_leftCamOpt
        self.odom_to_leftCamOpt[:,3] = self.T_odom_leftCamOpt
        self.H_leftCamOpt = np.zeros([4,4])
        self.H_leftCamOpt[3,3] = 1


        ## feature detection and matching
        self.orb = cv2.ORB_create()

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 25) #50
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    def ExtractPlane(self):
        self.camera_info = rospy.wait_for_message('/zed/zed_node/depth/camera_info', CameraInfo)
        rospy.loginfo('received camera info')
        imu_data = message_filters.Subscriber("/zed/zed_node/imu/data", Imu)
        rospy.loginfo('received imu data')
        pcl = message_filters.Subscriber("/zed/zed_node/point_cloud/cloud_registered", PointCloud2)
        rospy.loginfo('received pointcloud')
        depth_image = message_filters.Subscriber("zed/zed_node/depth/depth_registered", Image)
        rospy.loginfo('received depth image')
        rgb_img = message_filters.Subscriber("zed/zed_node/rgb/image_rect_color", Image)
        rospy.loginfo('received rgb image')
        ts = message_filters.ApproximateTimeSynchronizer([pcl,imu_data,rgb_img],10,0.1)
        rospy.loginfo('messages synchronized')
        ts.registerCallback(self.FitPlane)
        rospy.spin()

    def FitPlane(self,pcl,imu, rgb_img):


         xyz = self.PCLtoXYZ(pcl)

         xyz = xyz[xyz[:,1] > self.width[0]]
         xyz = xyz[xyz[:,1] < self.width[1]]

         xyz = xyz[xyz[:,2] > self.height[0]]
         xyz = xyz[xyz[:,2] < self.height[1]]

         xyz = xyz[xyz[:,0] < self.depth[1]]

         ## publisih the pcl and see whther it works
         # pcl_segmented = self.XYZtoPCL(xyz, pcl.header.stamp, pcl.header.frame_id)
         # self.pub_pcl.publish(pcl_segmented)

         ## plane fitting assuming pitch = 0
         quat = imu.orientation
         imu_euler = tf.transformations.euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])


         R_imu = np.asarray([[np.cos(imu_euler[1]),0.0,np.sin(imu_euler[1])],
                        [0.0,1.0,0.0],
                        [-np.sin(imu_euler[1]),0.0, np.cos(imu_euler[1])]])

         xyz = np.matmul(R_imu,np.transpose(xyz)) # apply pitch
         xyz = np.transpose(xyz)

         ## call ransac
         plane_coeffs, grnd_pnts = self.planeRansac(xyz,pcl)
         if len(grnd_pnts) > 0:
             xyz_grnd = xyz[grnd_pnts,:]
             #all_idx = np.arange(0,len(xyz))

             idx_non_grnd = ~grnd_pnts
             ## visualize the gound plane in rviz
             pcl_grnd = self.XYZtoPCL(xyz_grnd, pcl.header.stamp, pcl.header.frame_id)
             self.pub_grnd.publish(pcl_grnd)
             ## obstacle detection on this
             xyz_non_grnd = xyz[idx_non_grnd,:]

             #print(np.shape(xyz_non_grnd))
             ## visualize the non_grnd_pnts in rviz
             ##pcl_non_grnd = self.XYZtoPCL(xyz_non_grnd, pcl.header.stamp, pcl.header.frame_id)
             ##self.pub_non_grnd.publish(pcl_non_grnd)

             # lets do some clustering

             clusters = DBSCAN(eps = self.cluster_dist_thresh, min_samples = self.min_cluster_samples,
                               metric = 'euclidean', algorithm = 'kd_tree').fit(xyz_non_grnd)
             no_clusters = len(np.unique(clusters.labels_)) - 1

             ## store the idices for each cluster in a list
             self.cluster_idx = [] # indices of corresponding cluster
             self.cluster_size = [] # cluster size(number of points)
             self.cluster_dist = [] # dist of the cluster from the origin
             self.cluster_height = []
             self.cluster_depth = []
             self.cluster_width = []

             for i in range (0,no_clusters):
                 self.cluster_idx.append(np.argwhere(clusters.labels_ == i))
                 self.cluster_size.append(len(self.cluster_idx[i]))
                 self.cluster_dist.append(np.min(xyz_non_grnd[self.cluster_idx[i],0]))
                 self.cluster_depth.append(np.abs(np.max(xyz_non_grnd[self.cluster_idx[i],0]) - self.cluster_dist[i]))
                 self.cluster_height.append(np.abs(np.max(xyz_non_grnd[self.cluster_idx[i],2]) - np.min(xyz_non_grnd[self.cluster_idx[i],2])))
                 self.cluster_width.append(np.abs(np.max(xyz_non_grnd[self.cluster_idx[i],1]) - np.min(xyz_non_grnd[self.cluster_idx[i],1])))

             obs_idx = self.cluster_idx[np.argmax(self.cluster_size)]
             xyz_obs = xyz_non_grnd[obs_idx[:,0],:]
             xyz_obs = np.transpose(np.matmul(np.linalg.inv(R_imu), np.transpose(xyz_obs)))
             ## visualize obstacle cluster in rviz
             pcl_obs = self.XYZtoPCL(xyz_obs, pcl.header.stamp, pcl.header.frame_id)
             self.pub_obs.publish(pcl_obs)

             self.XYZtoImage(xyz_obs,pcl, rgb_img)

             if self.first_detection:
                 self.source_img = self.dest_img
                 self.xyz_img_src = self.xyz_img_dst
                 self.first_detection = False

             self.feature_matching()

    def planeRansac(self, pnts, pcl):

        max_pnts = 200 # max points in the plane
        bestSupport = 0
        bestPlane = []
        bestStd = np.inf
        dist_thresh = 0.05
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

    def XYZtoImage(self,xyz,pcl, rgb_img):

        cv_rgb_img = self.bridge.imgmsg_to_cv2(rgb_img, "bgr8")
        tracking_rgb_img = np.zeros([self.img_height,self.img_width,3], dtype=np.uint8)
        img = np.zeros([self.img_height,self.img_width], dtype=np.uint8)
        K = np.resize(self.camera_info.K,[3,3])
        P = np.resize(self.camera_info.P,[3,4])
        R =  tf.transformations.euler_matrix(1.57, -1.57, 0)


        xyz_ = np.hstack([xyz,np.ones([len(xyz),1])])
        xyz_t = np.matmul(self.R_leftCamOpt_to_leftCam,np.transpose(xyz_))
        pixel_coords = np.matmul(P,xyz_t)
        pixel_coords_y = np.round(pixel_coords[0,:]/pixel_coords[2,:])
        pixel_coords_x = np.round(pixel_coords[1,:]/pixel_coords[2,:])

        pixel_coords_y = np.clip(pixel_coords_y.astype(int),0,self.img_width-1)
        pixel_coords_x = np.clip(pixel_coords_x.astype(int),0,self.img_height-1)

        img[pixel_coords_x,pixel_coords_y] = 255

        tracking_rgb_img[pixel_coords_x,pixel_coords_y] = cv_rgb_img[pixel_coords_x, pixel_coords_y]
        self.dest_img = tracking_rgb_img
        cv_rgb_img[pixel_coords_x, pixel_coords_y] = [255,255,255]

        try:
            obs_img = self.bridge.cv2_to_imgmsg(img,encoding="mono8")
        except CvBridgeError as e:
            print(e)
        obs_img.header.frame_id = self.camera_info.header.frame_id
        obs_img.header.stamp = pcl.header.stamp
        self.pub_img.publish(obs_img)

        try:
            rgb_obs_img = self.bridge.cv2_to_imgmsg(tracking_rgb_img,encoding="bgr8")
        except CvBridgeError as e:
            print(e)
        rgb_obs_img.header.frame_id = rgb_img.header.frame_id
        rgb_obs_img.header.stamp = rgb_obs_img.header.stamp
        self.pub_rgb_img.publish(rgb_obs_img)




    def PCLtoXYZ(self, pcl):
        tic = time.time()
        xyz_img  = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(pcl,remove_nans=False)
        xyz = xyz_img.reshape([-1,3])
        R = self.R_leftCamOpt_to_leftCam[0:3,0:3]
        ## transform into camera left optical frame
        xyz_R = np.transpose(np.matmul(R,np.transpose(xyz)))
         # xyz in left camera optical frame

        self.xyz_img_dst = np.reshape(xyz_R,[self.img_height,self.img_width,3]) ##  xyz image in left camera optical frame

        xyz = xyz[~np.isnan(xyz).any(1)]

        return xyz

    def XYZtoPCL(self,xyz,stamp,frame_id):
        msg = PointCloud2()

        msg.header.stamp = stamp
        msg.header.frame_id = frame_id

        msg.height = 1
        msg.width = len(xyz)

        msg.fields = [PointField('x',0,PointField.FLOAT32,1),
                      PointField('y',4,PointField.FLOAT32,1),
                      PointField('z',8,PointField.FLOAT32,1)]

        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = 12*xyz.shape[0]
        msg.is_dense = int(np.isfinite(xyz).all())
        msg.data = np.asarray(xyz, np.float32).tostring()
        return msg



    def feature_matching(self):

        kp1, des1 = self.orb.detectAndCompute(self.source_img, None)
        kp2, des2 = self.orb.detectAndCompute(self.dest_img, None)

        if len(kp1) > 4 and len(kp2) > 4:
            # Match descriptors.
            matches = self.flann.knnMatch(np.asarray(des1,np.float32),np.asarray(des2,np.float32),k= 2)

            good = []
            for m,n in matches:
                if m.distance < 0.7*n.distance:
                    good.append(m)


            if len(good)> self.min_matches:
                src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)


                ## using 2d features for homography estimation
                src_idx = np.reshape(src_pts.astype(int),[-1,2])
                dst_idx = np.reshape(src_pts.astype(int),[-1,2])



                src_3d_pts = self.xyz_img_src[src_idx[:,1],src_idx[:,0],:]
                src_nan_idx = np.isnan(src_3d_pts)
                #src_3d_pts = src_3d_pts[~np.isnan(src_3d_pts).any(1)]
                dst_3d_pts = self.xyz_img_dst[dst_idx[:,1],dst_idx[:,0],:]
                dst_nan_idx = np.isnan(dst_3d_pts)
                #dst_3d_pts = dst_3d_pts[~np.isnan(dst_3d_pts).any(1)]
                non_nan_idx = np.logical_or(src_nan_idx,dst_nan_idx)
                src_3d_pts = src_3d_pts[~non_nan_idx.any(1)]
                dst_3d_pts = dst_3d_pts[~non_nan_idx.any(1)]

                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
                print("Transformation", H)
                rpy = tf.transformations.euler_from_matrix(H)
                rpy = np.asarray(rpy)*180/np.pi
                print("roll, pitch, yaw" ,rpy)
                ## draw_matches for debuggin


                H_3d = self.ExtractTransformation(src_3d_pts,dst_3d_pts)

                #self.DrawMatches(H_3d, good, kp1, kp2, mask)
            else:
                rospy.loginfo("good matches less than 4")
        else:
            rospy.loginfo("No matches found")

        self.source_img = self.dest_img
        self.xyz_img_src = self.xyz_img_dst

    def ExtractTransformation(self, src_pnts, dst_pnts):

        src_pnts_ =  np.hstack([src_pnts,np.ones([len(src_pnts),1])])
        src_pnts_odom = np.transpose(np.matmul(self.odom_to_leftCamOpt,np.transpose(src_pnts_)))[:,0:3]


        dst_pnts_ =  np.hstack([dst_pnts,np.ones([len(dst_pnts),1])])
        dst_pnts_odom = np.transpose(np.matmul(self.odom_to_leftCamOpt,np.transpose(dst_pnts_)))[:,0:3]

        src_centroid = np.mean(src_pnts_odom, axis=0)
        dst_centroid = np.mean(dst_pnts_odom, axis=0)

        src_centered = src_pnts_odom - src_centroid
        dst_centered = dst_pnts_odom - dst_centroid

        H = np.matmul(np.transpose(dst_centered), src_centered)

        U,S,Vt = np.linalg.svd(H)
        R = np.matmul(U,Vt)

        if np.linalg.det(R) < 0:
            Vt[2,:] *= -1
            R = U*Vt

        t = src_centroid - np.matmul(R,dst_centroid)

        #self.H_odom = self.odom_to_leftCamOpt*self.H_leftCamOpt
        rpy = tf.transformations.euler_from_matrix(R)
        rpy = np.asarray(rpy)*180/np.pi
        #print("rotation",self.H_odom[0:3,0:3])
        print("rpy", rpy)
        print("translation",t)
        return R

    def DrawMatches(self, H, good_matches, kp1, kp2, mask):
        """ draw matches for sanity check between the two images
        """
        img1 = self.source_img
        img2 = self.dest_img
        matchesMask = mask.ravel().tolist()
        pts = np.float32([ [0,0],[0,self.img_height-1],[self.img_width-1,self.img_height-1],[self.img_width-1,0] ]).reshape(-1,1,2)

        dst = cv2.perspectiveTransform(pts,H)
        img2 = cv2.polylines(img2, [np.int32(dst)], True, 255,3, cv2.LINE_AA)

        draw_params = dict(matchColor = (0,255,0), singlePointColor = None, flags = 2 )
        img3 = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, **draw_params)

        try:
            matches_img = self.bridge.cv2_to_imgmsg(img3,encoding="bgr8")
        except CvBridgeError as e:
            print(e)
        self.pub_matches_img.publish(matches_img)






if __name__=="__main__":
    obj = Segmentation()
    obj.ExtractPlane()
