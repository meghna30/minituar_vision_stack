## helper fuctions to convert xyz to pcl and pcl to xyz

import rospy
import numpy as np
import ros_numpy
from sensor_msgs.msg import PointCloud2, PointField
import pdb


def PCLtoXYZ(pcl, R_leftCamOpt_to_leftCam, img_height,img_width):

    xyz_img  = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(pcl,remove_nans=False)

    xyz = xyz_img.reshape([-1,3])
    R = R_leftCamOpt_to_leftCam[0:3,0:3]
    ## transform into camera left optical frame
    xyz_R = np.transpose(np.matmul(R,np.transpose(xyz)))
     # xyz in left camera optical frame

    xyz_img_cam_opt = np.reshape(xyz_R,[img_height,img_width,3]) ##  xyz image in left camera optical frame
    # test_pcl_image  = self.XYZtoPCL(self.xyz_img_dst, pcl.header.stamp, pcl.header.frame_id)
    # self.pub_pcl.publish(test_pcl_image)
    xyz = xyz[~np.isnan(xyz).any(1)]
    return xyz, xyz_img_cam_opt




def XYZtoPCL(xyz,stamp,frame_id):
    msg = PointCloud2()

    msg.header.stamp = stamp
    msg.header.frame_id = frame_id

    if len(xyz.shape) == 3:
        msg.height = xyz.shape[1]
        msg.width = xyz.shape[0]
    else:
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
