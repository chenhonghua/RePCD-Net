import os
import threading
import time

import queue
import h5py
import numpy as np
import tensorflow as tf

from tf_ops.grouping.tf_grouping import query_ball_point, group_point ,knn_point
from tf_ops.sampling import tf_sampling

def normalize_point_cloud(input):
    if len(input.shape)==2:
        axis = 0
    elif len(input.shape)==3:
        axis = 1
    centroid = np.mean(input, axis=axis, keepdims=True)
    input = input - centroid
    furthest_distance = np.amax(np.sqrt(np.sum(input ** 2, axis=-1)),axis=axis,keepdims=True)
    input = input / furthest_distance
    return input, centroid,furthest_distance

def load_obj_data(path):
    vertexs=[]
    faces=[]
    for line in open(path, "r"):
        if line.startswith('#'): continue
        values = line.split()
        if not values: continue
        if values[0] == 'v':
            v = [float(x) for x in values[1:4]]
            vertexs.append(v)
        elif values[0] == 'f':
            f = [float(x) for x in values[1:4]]
            faces.append(f)
    ver=np.array(vertexs)
    fa=np.array(faces)
    return ver,fa

def save_obj_data(path,vertexs,faces):
    if not os.path.exists(os.path.split(path)[0]):
        os.makedirs(os.path.split(path)[0])
    myfile = open(path, "w")
    vertex_num = vertexs.shape[0]
    faces_num = faces.shape[0]
    for j in range(vertex_num):
        print >> myfile, "v %f %f %f" % (vertexs[j,0],vertexs[j,1],vertexs[j,2])
    for j in range(faces_num):
        print >> myfile, "f %d %d %d" % (faces[j,0],faces[j,1],faces[j,2])
    myfile.close()

def save_xyz_data(path,vertexs,faces):
    if not os.path.exists(os.path.split(path)[0]):
        os.makedirs(os.path.split(path)[0])
    myfile = open(path, "w")
    vertex_num = vertexs.shape[0]
    faces_num = faces.shape[0]
    for j in range(vertex_num):
        if vertexs.shape[1]==3:
            print >> myfile, "%f %f %f" % (vertexs[j,0],vertexs[j,1],vertexs[j,2])
    else:
            if vertexs.shape[1]==6:
                print >> myfile, "%f %f %f" % (vertexs[j,0],vertexs[j,1],vertexs[j,2],vertexs[j,3],vertexs[j,4],vertexs[j,5])
    myfile.close()


def load_h5_data( num_center=500 ,num_point = 4000,h5_filename='../h5_data/train_4000.h5'):

    print("input h5 file is:", h5_filename)
    f = h5py.File(h5_filename)
    center=np.array(f['center'])[:,:num_center,:]
    input = np.array(f['noise'])[:,:num_point,:]
    gt = np.array(f['noise_gt'])[:,:num_center*2,:3]
    gt2	= np.array(f['noise_gt'])[:,:num_point,:3]
    weight = np.array(f['weight'])[:,:num_point]
    #assert len(center) == len(gt)

    input=np.concatenate([center,input],axis=1)

    print("Normalization the data")
    data_radius = np.ones(shape=(input.shape[0]))
    centroid = np.expand_dims(center[:, 0,0:3],axis=1)
    center[:, :, 0:3] =center[:, :,0:3] - centroid
    furthest_distance = np.amax(np.sqrt(np.sum(center[:, :,0:3] **2, axis=-1)), axis=1, keepdims=True)
    input[:, :, 0:3] = input[:, :, 0:3] - centroid
    input[:, :, 0:3] =input[:, :,0:3] / np.expand_dims(furthest_distance, axis=-1)
    gt[:, :, 0:3] = gt[:, :, 0:3] - centroid
    gt[:, :, 0:3] = gt[:, :, 0:3] / np.expand_dims(furthest_distance, axis=-1)
    gt2[:, :, 0:3] = gt2[:, :, 0:3] - centroid
    gt2[:, :, 0:3] = gt2[:, :, 0:3] / np.expand_dims(furthest_distance, axis=-1)
    print(len(input),len(input[0]),len(input[0][0]))
    print(len(gt),len(gt[0]),len(gt[0][0]))
    print("total %d samples" % (len(input)))
    return input, gt, gt2, data_radius, weight

def rotate_point_cloud_and_gt(batch_data,batch_gt=None):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    for k in range(batch_data.shape[0]):
        angles = np.random.uniform(size=(3)) * 2 * np.pi
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]), np.cos(angles[2]), 0],
                       [0, 0, 1]])
        rotation_matrix = np.dot(Rz, np.dot(Ry, Rx))

        # rotation_angle = np.random.uniform(size=(3)) * 2 * np.pi
        # cosval = np.cos(rotation_angle)
        # sinval = np.sin(rotation_angle)
        # rotation_matrix = np.array([[cosval, 0, sinval],
        #                             [0, 1, 0],
        #                             [-sinval, 0, cosval]])

        batch_data[k, ..., 0:3] = np.dot(batch_data[k, ..., 0:3].reshape((-1, 3)), rotation_matrix)
        if batch_data.shape[-1]>3:
            batch_data[k, ..., 3:] = np.dot(batch_data[k, ..., 3:].reshape((-1, 3)), rotation_matrix)

        if batch_gt is not None:
            batch_gt[k, ..., 0:3]   = np.dot(batch_gt[k, ..., 0:3].reshape((-1, 3)), rotation_matrix)
            if batch_gt.shape[-1] > 3:
                batch_gt[k, ..., 3:] = np.dot(batch_gt[k, ..., 3:].reshape((-1, 3)), rotation_matrix)

    return batch_data,batch_gt


def shift_point_cloud_and_gt(batch_data, batch_gt = None, shift_range=0.3):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, shifted batch of point clouds
    """
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B,3))
    for batch_index in range(B):
        batch_data[batch_index,:,0:3] += shifts[batch_index,0:3]

    if batch_gt is not None:
        for batch_index in range(B):
            batch_gt[batch_index, :, 0:3] += shifts[batch_index, 0:3]

    return batch_data,batch_gt


def random_scale_point_cloud_and_gt(batch_data, batch_gt = None, scale_low=0.5, scale_high=2):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.shape#batch_size,num_points,3/6
    scales = np.random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        batch_data[batch_index,:,0:3] *= scales[batch_index]

    if batch_gt is not None:
        for batch_index in range(B):
            batch_gt[batch_index, :, 0:3] *= scales[batch_index]

    return batch_data,batch_gt,scales


def rotate_perturbation_point_cloud(batch_data, angle_sigma=0.03, angle_clip=0.09):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    for k in xrange(batch_data.shape[0]):
        angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
        Rx = np.array([[1,0,0],
                       [0,np.cos(angles[0]),-np.sin(angles[0])],
                       [0,np.sin(angles[0]),np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                       [0,1,0],
                       [-np.sin(angles[1]),0,np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                       [np.sin(angles[2]),np.cos(angles[2]),0],
                       [0,0,1]])
        R = np.dot(Rz, np.dot(Ry,Rx))
        batch_data[k, ...,0:3] = np.dot(batch_data[k, ...,0:3].reshape((-1, 3)), R)
        if batch_data.shape[-1]>3:
            batch_data[k, ..., 3:] = np.dot(batch_data[k, ..., 3:].reshape((-1, 3)), R)

    return batch_data


def jitter_perturbation_point_cloud(batch_data, sigma=0.005, clip=0.02):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data[:,:,3:] = 0
    jittered_data += batch_data
    return jittered_data


def save_pl(path, pl):
    if not os.path.exists(os.path.split(path)[0]):
        os.makedirs(os.path.split(path)[0])
    myfile = open(path, "w")
    point_num = pl.shape[0]
    for j in range(point_num):
        if (len(pl[j])==3):
            myfile.write("%f %f %f\n" % (pl[j,0],pl[j,1],pl[j,2]))
        elif (len(pl[j])==6):
            myfile.write("%f %f %f %f %f %f\n" % (pl[j, 0], pl[j, 1], pl[j, 2],pl[j, 3],pl[j, 4],pl[j, 5]))
    myfile.close()
    if np.random.rand()>1.9:
        show3d.showpoints(pl[:, 0:3])


def nonuniform_sampling(num = 4096, sample_num = 1024):
    sample = set()
    loc = np.random.rand()*0.8+0.1
    while(len(sample)<sample_num):
        a = int(np.random.normal(loc=loc,scale=0.3)*num)
        if a<0 or a>=num:
            continue
        sample.add(a)
    return list(sample)


def pca_normalize(batch_center_data, batch_input_data, batch_data_gt,batch_data_gt2):
    x = batch_input_data
    rotated_data = np.zeros(x.shape, dtype=np.float32)
    #x = x - np.mean(x, axis = -2, keepdims = True)
    H = np.matmul(x.transpose(0,2,1), x)
    for k in range(x.shape[0]):
        eigenvectors, eigenvalues, _ = np.linalg.svd(H[k])
        sort = eigenvalues.argsort()
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]
        normal = eigenvectors[:,0]
        
        shape_pc = x[k, ...]
        batch_center_data_k = batch_center_data[k, ...]
        batch_input_data_k = batch_input_data[k, ...]
        batch_data_gt_k = batch_data_gt[k, ...]
        batch_data_gt2_k = batch_data_gt2[k, ...]

        q_angle = np.arctan(normal[0]/normal[2])
        q_cos = np.cos(q_angle)
        q_sin = np.sin(q_angle)    
        rotation_matrix = np.array([[q_cos, 0, q_sin],
                                    [0, 1, 0],
                                    [-q_sin, 0, q_cos]])
        shape_pc = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        batch_center_data_k = np.dot(batch_center_data_k.reshape((-1, 3)), rotation_matrix)
        batch_input_data_k = np.dot(batch_input_data_k.reshape((-1, 3)), rotation_matrix)
        batch_data_gt_k = np.dot(batch_data_gt_k.reshape((-1, 3)), rotation_matrix)
        batch_data_gt2_k = np.dot(batch_data_gt2_k.reshape((-1, 3)), rotation_matrix)


        w_angle = np.arctan(normal[1]/np.sqrt(normal[0]*normal[0]+normal[2]*normal[2]))
        w_cos = np.cos(w_angle)
        w_sin = np.sin(w_angle)   
        rotation_matrix = np.array([[1, 0, 0],
                                    [0, w_cos, w_sin],
                                    [0, -w_sin, w_cos]])

        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        batch_center_data[k, ...] = np.dot(batch_center_data_k.reshape((-1, 3)), rotation_matrix)
        batch_input_data[k, ...] = np.dot(batch_input_data_k.reshape((-1, 3)), rotation_matrix)
        batch_data_gt[k, ...] = np.dot(batch_data_gt_k.reshape((-1, 3)), rotation_matrix)
        batch_data_gt2[k, ...] = np.dot(batch_data_gt2_k.reshape((-1, 3)), rotation_matrix)
    return batch_center_data, batch_input_data, batch_data_gt,batch_data_gt2


class Fetcher(threading.Thread):
    def __init__(self,input,gt_data, gt2_data,radius_data,weight,batch_size,num_center,num_point, IS_PCA = False):
        super(Fetcher,self).__init__()
        self.queue = queue.Queue(50)
        self.stopped = False
        self.center_data = None
        self.input_data = None
        self.input=input
        self.gt_data = gt_data
        self.gt2_data = gt2_data
        self.radius_data = radius_data
        self.weight = weight
        self.batch_size = batch_size
        self.num_center= num_center 
        self.num_point = num_point
        self.IS_PCA = IS_PCA
        self.sample_cnt = self.input.shape[0]
        self.num_batches = self.sample_cnt//self.batch_size
        self.seed=1
        print("NUM_BATCH is %s"%(self.num_batches))

    def run(self):
        
        while not self.stopped:
            idx = np.arange(0,self.sample_cnt)
            np.random.shuffle(idx)
            self.input = self.input[idx, ...]
            self.gt_data = self.gt_data[idx, ...]
            self.gt2_data = self.gt2_data[idx, ...]
            self.radius_data = self.radius_data[idx, ...]
            self.weight = self.weight[idx, ...]

            self.center_data=self.input[:,0:self.num_center,:]
            self.input_data=self.input[:,self.num_center:,:]

            for batch_idx in range(self.num_batches):
                if self.stopped:
                    return None
                start_idx = batch_idx * self.batch_size 
                end_idx = (batch_idx + 1) * self.batch_size
                batch_center_data = self.center_data[start_idx:end_idx, :, :].copy()#[b,:,3]
                batch_input_data = self.input_data[start_idx:end_idx, :, :].copy()#[b,:,3]
                batch_data_gt = self.gt_data[start_idx:end_idx, :, :].copy()#[b,:,3]
                batch_data_gt2 = self.gt2_data[start_idx:end_idx,:, :].copy()#[b,:,3]
                radius = self.radius_data[start_idx:end_idx].copy()  #[b]
                batch_weight = self.weight[start_idx:end_idx].copy()  #[b,N]
                if(self.IS_PCA == True):
                    batch_center_data,batch_input_data,batch_data_gt,batch_data_gt2 = pca_normalize(batch_center_data,batch_input_data,batch_data_gt,batch_data_gt2)

                self.queue.put((batch_center_data,batch_input_data,batch_data_gt, batch_data_gt2, radius, batch_weight))
            print(self.center_data.shape,self.input_data.shape,self.gt_data.shape,self.gt2_data.shape,self.radius_data.shape,self.weight.shape)
        return None
    def fetch(self):
        if self.stopped:
            return None
        return self.queue.get()

    def shutdown(self):
        self.stopped = True
        print ("Shutdown .....")
        while not self.queue.empty():
            self.queue.get()
        print ("Remove all queue data")

if __name__ == '__main__':
    print('data')

