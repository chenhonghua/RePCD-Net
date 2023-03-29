import os
import numpy as np
from scipy import spatial

import tensorflow as tf
from tf_ops.CD import tf_nndistance
from tf_ops.emd import tf_auctionmatch
from tf_ops.grouping.tf_grouping import query_ball_point, group_point,knn_point,knn_point_2
from tf_ops.sampling import tf_sampling


def pre_load_checkpoint(checkpoint_dir):
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:

        epoch_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
        return epoch_step,ckpt.model_checkpoint_path
    else:
        return 0,None


def get_repulsion_loss4(pred,pred2, nsample=20, radius=0.07):
    idx, pts_cnt = query_ball_point(radius, nsample, pred, pred2)

    tf.summary.histogram('smooth/unque_index', pts_cnt)

    grouped_pred = group_point(pred, idx)
    grouped_pred -= tf.expand_dims(pred2, 2)

    ##get the uniform loss
    h = 0.03
    dist_square = tf.reduce_sum(grouped_pred ** 2, axis=-1)
    dist_square, idx = tf.nn.top_k(-dist_square, 5)
    dist_square = -dist_square[:, :, 1:]  
    dist_square = tf.maximum(1e-12,dist_square)
    dist = tf.sqrt(dist_square)
    weight = tf.exp(-dist_square/h**2)
    uniform_loss = tf.reduce_mean(radius-dist*weight)
    return uniform_loss

def get_repulsion_loss(pred,pred2):
    dist_square,idx = knn_point_2(7,pred,pred2)
    dist_square = dist_square[:, :, 1:]
    dist_square = tf.abs(0.00363*2-dist_square)
    uniform_loss = tf.reduce_mean(dist_square)
    return uniform_loss


def get_emd_loss(pred, gt, radius):
    """ pred: BxNxC,
        label: BxN, """
    batch_size = pred.get_shape()[0].value
    matchl_out, matchr_out = tf_auctionmatch.auction_match(pred, gt)
    matched_out = tf_sampling.gather_point(gt, matchl_out) 
    dist = tf.reshape((pred - matched_out) ** 2, shape=(batch_size, -1))
    dist = tf.reduce_mean(dist, axis=1, keep_dims=True)
    dist_norm = dist / radius

    emd_loss = tf.reduce_mean(dist_norm)#loss
    return emd_loss,matchl_out

def get_cd_loss(pred, gt, radius):
    """ pred: BxNxC,
        label: BxN, """
    dists_forward, _, dists_backward, _ = tf_nndistance.nn_distance(gt, pred)
    #dists_forward is for each element in gt, the cloest distance^2 to this element
    CD_dist = 0.5*dists_forward + 0.5*dists_backward
    CD_dist = tf.reduce_mean(CD_dist, axis=1)
    CD_dist_norm = CD_dist/radius
    cd_loss = tf.reduce_mean(CD_dist_norm)
    return cd_loss,None

def get_cd_weight_loss(pred, gt, radius,weight):
    """ pred: BxNxC,
        label: BxN, """
    dists_forward, _, dists_backward, _ = tf_nndistance.nn_distance(gt, pred)
    #dists_forward is for each element in gt, the cloest distance^2 to this element
    CD_dist = 0.5*dists_forward + 0.5*dists_backward
    CD_dist = CD_dist * weight #/ 100.0
    CD_dist = tf.reduce_mean(CD_dist, axis=1)
    CD_dist_norm = CD_dist/radius
    cd_loss = tf.reduce_mean(CD_dist_norm)
    return cd_loss,None

def get_res_loss(res):
    """ pred: BxNxC,
        label: BxN, """

    Res = tf.reduce_sum(res**2, axis=2)
    Res_loss = tf.reduce_mean(Res)
    return Res_loss, None

def get_cf_loss(pred, gt, radius):
    """ pred: BxNxC,
        label: BxN, """    
    xyz1 = tf.expand_dims(gt,axis=1)
    xyz2 = tf.expand_dims(pred,axis=2)
    dist = tf.reduce_sum((xyz1-xyz2)**2, -1)
    min_dist = tf.reduce_min(dist,axis=-1)
    xyz3 = tf.expand_dims(pred,axis=1)
    xyz4 = tf.expand_dims(gt,axis=2)
    dist2 = tf.reduce_sum((xyz3-xyz4)**2, -1)
    min_dist2 = tf.reduce_min(dist2,axis=-1)
    cf_dist=0.9*min_dist+0.1*min_dist2
    C_dist = tf.reduce_mean(cf_dist, axis=-1) #(B)
    cf_loss = tf.reduce_mean(C_dist)
    
    return 100*cf_loss

def get_cd_direct_loss(pred, gt, input, radius):
    """ pred: BxNxC,
        label: BxN, """
    dists_forward, _, dists_backward, idx = tf_nndistance.nn_distance(gt, input)
    #dists_forward is for each element in gt, the cloest distance^2 to this element
    points = tf_sampling.gather_point(gt, idx)
    CD_dist = tf.reduce_sum((pred - points) ** 2,axis=-1)
    CD_dist = tf.reduce_mean(CD_dist, axis=1)
    CD_dist_norm = CD_dist/radius
    cd_loss = tf.reduce_mean(CD_dist_norm)
    return cd_loss,None

def get_cd_direct_weight_loss(pred, gt, input, radius,weight):
    """ pred: BxNxC,
        label: BxN, """
    dists_forward, _, dists_backward, idx = tf_nndistance.nn_distance(gt, input)
    #dists_forward is for each element in gt, the cloest distance^2 to this element
    points = tf_sampling.gather_point(gt, idx)
    CD_dist = tf.reduce_sum((pred - points) ** 2,axis=-1)
    CD_dist = CD_dist * weight
    CD_dist = tf.reduce_mean(CD_dist, axis=1)
    CD_dist_norm = CD_dist/radius
    cd_loss = tf.reduce_mean(CD_dist_norm)
    return cd_loss,None

def get_emd_direct_loss(pred, gt, input, radius):
    """ pred: BxNxC,
        label: BxN, """
    batch_size = pred.get_shape()[0].value
    matchl_out, matchr_out = tf_auctionmatch.auction_match(input, gt)
    matched_out = tf_sampling.gather_point(gt, matchl_out) 
    dist = tf.reshape((pred - matched_out) ** 2, shape=(batch_size, -1))
    dist = tf.reduce_mean(dist, axis=1, keep_dims=True)
    dist_norm = dist / radius

    emd_loss = tf.reduce_mean(dist_norm)#loss
    return emd_loss,matchl_out

if __name__ == '__main__':
    gt = tf.constant([[[1,0,0],[3,0,0],[4,0,0]]],tf.float32)
    pred = tf.constant([[[-10,1.1,0], [1,0, 0], [2,0, 0], [3,0,0]]],tf.float32)

    dists_forward, idx1, dists_backward, idx2 = tf_nndistance.nn_distance(gt, pred)

    with tf.Session() as sess:
        print (idx1.eval()) # for each element in gt, the idx of pred
        print (dists_forward.eval())
        print (idx2.eval()) # for each element in pred,
        print (dists_backward.eval())
