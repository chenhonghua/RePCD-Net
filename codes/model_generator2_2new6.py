import tensorflow as tf
import math
import os
import sys
import numpy as np
from scipy import spatial

from utils import tf_util2
from utils.pointnet_util import pointnet_sa_module
from tf_ops.grouping.tf_grouping import query_ball_point, group_point, knn_point
from tf_ops.sampling.tf_sampling import farthest_point_sample, gather_point


def placeholder_inputs(batch_size, num_center, num_point):
    pointclouds_cp = tf.placeholder(tf.float32, shape=(batch_size, num_center, 3))
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    pointclouds_gt = tf.placeholder(tf.float32, shape=(batch_size, 2*num_center, 3))
    pointclouds_gt2 = tf.placeholder(tf.float32, shape=(batch_size, num_point , 3))
    pointclouds_radius = tf.placeholder(tf.float32, shape=(batch_size))
    pointclouds_weight = tf.placeholder(tf.float32, shape=(batch_size, num_point))
    return pointclouds_cp,pointclouds_pl, pointclouds_gt,pointclouds_gt2,  pointclouds_radius, pointclouds_weight

def get_res_model(center_cloud,point_cloud, is_training, scope ,bradius=1.0, reuse=None,use_bn=False,use_ibn = False, bn_decay=None ,tran_feature=[] , ball_number=4, feature_tran_num=1):
    with tf.variable_scope(scope,reuse=reuse) as sc:
        batch_size = point_cloud.get_shape()[0].value
        num_center = center_cloud.get_shape()[1].value
        num_point = point_cloud.get_shape()[1].value
        l0_center_xyz=center_cloud[:,:,0:3]
        l0_xyz = point_cloud[:,:,0:3]
        new_points=None
        new_points_list = []
        # Layer
        nsample_list=[32,48,64,128]
        mlp_list = [32,64,128]
        radius_list=[0.1,0.2,0.4,0.8]
        if(ball_number==1):
            nsample_list=[128,]
            radius_list=[1.0,]
        num_patch = int(math.ceil(num_center / 1000.0))
        l0_center_xyz = l0_center_xyz
        l0_xyz = l0_xyz
        #print(num_patch,'0000000000000000000\n')
        for i in range(ball_number):
            radius=radius_list[i] * bradius
            nsample=nsample_list[i]
            for k in range(num_patch):
                patch_xyz=l0_center_xyz[:,k*1000:min((k+1)*1000,num_center),:]
                idx, pts_cnt = query_ball_point(radius, nsample, l0_xyz, patch_xyz)
                grouped_xyz = group_point(l0_xyz, idx)
                grouped_xyz -= tf.tile(tf.expand_dims(patch_xyz, 2), [1, 1, nsample, 1])
                grouped_points = grouped_xyz
                for j, num_out_channel in enumerate(mlp_list):
                    grouped_points = tf_util2.conv2d(grouped_points, num_out_channel, [1, 1],
                                                 padding='VALID', stride=[1, 1],
                                                 bn=use_bn, ibn=use_ibn, is_training=is_training,
                                                 scope='conv%d_%d' % (i,j),  bn_decay=bn_decay)

                if k == 0:
                    new_points_temp = tf.reduce_max(grouped_points, axis=[2])
                else:
                    new_points_temp = tf.concat([new_points_temp,tf.reduce_max(grouped_points, axis=[2])], axis=1)
            new_points = new_points_temp
            if(i==0):
                new_points_list = tf.expand_dims(new_points,axis=2)
                new_points_list2 = tf.expand_dims(new_points,axis=2)
            else:
                new_points_list = tf.concat([new_points_list,tf.expand_dims(new_points,axis=[2])],axis=2)
                new_points_list2 = tf.concat([tf.expand_dims(new_points,axis=[2]),new_points_list,],axis=2)
        next_tran_feature=[]
        feature = new_points_list
        feature2 = new_points_list2
        new_points_list = tf.expand_dims(new_points_list,axis=3)
        next_tran_feature = new_points_list

    with tf.variable_scope('generator3',reuse=tf.AUTO_REUSE) as sc:
        if tran_feature != []:
            new_points_list = tf.concat([tran_feature,new_points_list], axis= 3)
            next_tran_feature = new_points_list
            if (feature_tran_num>0) :
                for i in range(ball_number):
                    feature_temp = new_points_list[:,:,i,:,:]
                    feature_temp = tf_util2.seq2seq_with_attention(feature_temp, 128, scope='time_rnn'+str(i), bn=use_bn,
                                                 is_training=is_training, bn_decay=bn_decay)
                    if(i==0):
                        feature = tf.expand_dims(feature_temp, axis=2)
                        feature2 = tf.expand_dims(feature_temp, axis=2)
                    else:
                        feature = tf.concat([feature,tf.expand_dims(feature_temp, axis=2)],axis=2)
                        feature2 = tf.concat([tf.expand_dims(feature_temp, axis=2),feature2],axis=2)
        #feature_in = feature #b n k d
        #feature = tf.reshape(feature,[batch_size,num_center,1,ball_number*128])

    with tf.variable_scope(scope,reuse=reuse) as sc:
        if(ball_number>1):
            feature = tf_util2.rnn_encoder(feature, 128, scope='space_rnn_1', bn=use_bn,
                                                 is_training=is_training, bn_decay=bn_decay)
            feature2 = tf_util2.rnn_encoder(feature2, 128, scope='space_rnn_2', bn=use_bn,
                                                 is_training=is_training, bn_decay=bn_decay)
            feature = tf.concat([feature, feature2],axis=-1)

            #feature_out = feature #b n  d
            feature = tf.expand_dims(feature, axis=2) #bxNx1xD
        
        #get the xyz

        Resi = tf_util2.conv2d(feature, 64, [1, 1],
                              padding='VALID', stride=[1, 1],
                              bn=False, is_training=is_training,
                              scope='fc_layer1', bn_decay=bn_decay)

        Resi = tf_util2.conv2d(Resi, 3, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=False, is_training=is_training,
                             scope='fc_layer2', bn_decay=bn_decay,
                             activation_fn=None, weight_decay=0.0)  # B*N*1*3
        Resi = tf.squeeze(Resi, [2])  # B*N*3
        
        if not is_training:
            return Resi , next_tran_feature#, feature_in, feature_out
        Resxyz = tf.add(l0_center_xyz, Resi)
        output = Resxyz-tf.reduce_mean(Resxyz,axis=1,keep_dims=True)#move center
    return Resxyz,next_tran_feature
