import argparse
import os
import socket
import time
from glob import glob
from scipy import spatial
import data_provider
import model_generator2_2new6 as MODEL_GEN
import model_utils
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tqdm import tqdm
from utils import pc_util
import math
from tf_ops.grouping.tf_grouping import group_point

parser = argparse.ArgumentParser()
parser.add_argument('--phase', default='train', help='train or test [default: train]')
parser.add_argument('--gpu', default='0', help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='../model/generator2_new6', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=500,help='Point Number [2000/4000] [default: 2000]')
parser.add_argument('--num_center', type=int, default=500,help='center Point Number of patch[500/1000] [default: 500]')
parser.add_argument('--max_epoch', type=int, default=100, help='Epoch to run [default: 500]')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 32]')
parser.add_argument('--ball_number', type=int, default=4)
parser.add_argument('--filter_times', type=int, default=3)
parser.add_argument('--learning_rate', type=float, default=0.000001)
parser.add_argument('--feature_tran_num', type=int, default=1)
parser.add_argument('--sm_idx', type=int, default=-1)

ASSIGN_MODEL_PATH=None
USE_DATA_NORM = False
USE_REPULSION_LOSS = True

FLAGS = parser.parse_args()
PHASE = FLAGS.phase
GPU_INDEX = FLAGS.gpu
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
NUM_CENTER = FLAGS.num_center
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
MODEL_DIR = FLAGS.log_dir
FILTER_TIMES = FLAGS.filter_times
BALL_NUMBER=FLAGS.ball_number
FEATURE_TRAN_NUM=FLAGS.feature_tran_num

print(socket.gethostname())
print(FLAGS)
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_INDEX

def log_string(out_str):
    global LOG_FOUT
    LOG_FOUT.write(out_str)
    LOG_FOUT.flush()

def train(assign_model_path=None):
    is_training = True
    bn_decay = 0.95
    step = tf.Variable(0,trainable=False)
    #learning_rate = BASE_LEARNING_RATE
    learning_rate = tf.train.exponential_decay(BASE_LEARNING_RATE, step, 1350, 1, staircase=True)
    tf.summary.scalar('bn_decay', bn_decay)
    tf.summary.scalar('learning_rate', learning_rate)

    # get placeholder
    pointclouds_cp, pointclouds_pl, pointclouds_gt, pointclouds_gt2, pointclouds_radius, pointclouds_weight = MODEL_GEN.placeholder_inputs(BATCH_SIZE,  NUM_CENTER,NUM_POINT)
    #create the generator model
    #pred,_ = MODEL_GEN.get_res_model(pointclouds_pl, is_training, scope='generator',bradius=pointclouds_radius,
    #                                                      reuse=None, use_bn=False,use_ibn=False,
    #                                                      bn_decay=bn_decay,ball_number=BALL_NUMBER,filter_time=FILTER_TIMES)

    preds=[]
    for m in range(FILTER_TIMES):
        if m==0:
            preds.append(MODEL_GEN.get_res_model(pointclouds_cp, pointclouds_pl , is_training, scope='generator0',bradius=1.0 ,reuse=tf.AUTO_REUSE, use_bn=False, use_ibn=False, bn_decay=bn_decay, ball_number=BALL_NUMBER, feature_tran_num=FEATURE_TRAN_NUM))
        elif(m==1):
            preds.append(MODEL_GEN.get_res_model(preds[m-1][0],preds[m-1][0], is_training, scope='generator1',bradius=1.0,reuse=tf.AUTO_REUSE, use_bn=False,use_ibn=False, tran_feature=preds[m-1][1],bn_decay=bn_decay,ball_number=BALL_NUMBER, feature_tran_num=FEATURE_TRAN_NUM))
        elif(m==2):
            preds.append(MODEL_GEN.get_res_model(preds[m-1][0][:,:NUM_CENTER,:],preds[m-1][0], is_training, scope='generator2',bradius=1.0,reuse=tf.AUTO_REUSE, use_bn=False,use_ibn=False, tran_feature=preds[m-1][1][:,:NUM_CENTER,:,:,:], bn_decay=bn_decay, ball_number=BALL_NUMBER, feature_tran_num=FEATURE_TRAN_NUM))
    pred=preds[FILTER_TIMES-1][0]


    #get loss
    for m in range(FILTER_TIMES):
        if (m==0):
            gen_loss = model_utils.get_cd_direct_weight_loss(preds[m][0][:,:NUM_CENTER,:], pointclouds_gt,pointclouds_cp, pointclouds_radius,pointclouds_weight)[0]
        else:
            gen_loss += model_utils.get_cd_direct_weight_loss(preds[m][0][:,:NUM_CENTER,:], pointclouds_gt,pointclouds_cp, pointclouds_radius,pointclouds_weight)[0]
    #get repulsion loss
    if USE_REPULSION_LOSS:
        gen_repulsion_loss = model_utils.get_repulsion_loss(pred, pred)
        tf.summary.scalar('loss/gen_repulsion_loss', gen_repulsion_loss)
    else:
        gen_repulsion_loss =tf.Variable(0.0,trainable=False)

    #get total loss function
    pre_gen_loss = 100*gen_loss + gen_repulsion_loss  + tf.losses.get_regularization_loss()

    # create pre-generator ops
    gen_update_ops = [op for op in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if op.name.startswith("generator")]
    gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
    print(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
    print(tf.trainable_variables())
    print("*****************************************")
    print(tf.global_variables())
    with tf.control_dependencies(gen_update_ops):
        pre_gen_train = tf.train.AdamOptimizer(learning_rate,beta1=0.9).minimize(pre_gen_loss,var_list=gen_tvars,
                                                                                 colocate_gradients_with_ops=True,
                                                                                 global_step=step)

    tf.summary.scalar('loss/gen', gen_loss)
    tf.summary.scalar('loss/regularation', tf.losses.get_regularization_loss())
    tf.summary.scalar('loss/pre_gen_total', pre_gen_loss)
    pretrain_merged = tf.summary.merge_all()

    # Create a session
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    #config = tf.ConfigProto(gpu_options=gpu_options)
    
    config.allow_soft_placement = True
    config.log_device_placement = False
    with tf.Session(config=config) as sess:
        train_writer = tf.summary.FileWriter(os.path.join(MODEL_DIR, 'train'), sess.graph)
        
        init = tf.global_variables_initializer()
        sess.run(init)
        ops = {'pointclouds_cp': pointclouds_cp,
               'pointclouds_pl': pointclouds_pl,
               'pointclouds_gt': pointclouds_gt,
               'pointclouds_gt2': pointclouds_gt2,
               'pointclouds_radius': pointclouds_radius,
               'pointclouds_weight': pointclouds_weight,
               'pretrain_merged':pretrain_merged,
               'gen_loss': gen_loss,
               'gen_repulsion_loss':gen_repulsion_loss,
               'pre_gen_loss': pre_gen_loss,
               'pre_gen_train':pre_gen_train,
               'pred': pred,
               'step': step,
               'learning_rate':learning_rate,
               }
        #restore the model
        saver = tf.train.Saver(max_to_keep=300)
        restore_epoch, checkpoint_path = model_utils.pre_load_checkpoint(MODEL_DIR)

        global LOG_FOUT
        if restore_epoch==0: 
            LOG_FOUT = open(os.path.join(MODEL_DIR, 'log_train.txt'), 'w')
            LOG_FOUT.write(str(socket.gethostname()) + '\n')
            LOG_FOUT.write(str(FLAGS) + '\n')
        else:
            LOG_FOUT = open(os.path.join(MODEL_DIR, 'log_train.txt'), 'a')

            saver.restore(sess,checkpoint_path)

        ###assign the generator with another model file
        if assign_model_path is not None:
            print("Load pre-train model from %s"%(assign_model_path))
            assign_saver = tf.train.Saver(var_list=[var for var in tf.trainable_variables() if var.name.startswith("generator")])
            assign_saver.restore(sess, assign_model_path)

        input, gt_data,gt2_data,data_radius, weight = data_provider.load_h5_data(NUM_CENTER,NUM_POINT, h5_filename = '../h5_data/train_4000_normal_scale_label_weight_61_6.h5')

        fetchworker = data_provider.Fetcher(input,gt_data,gt2_data,data_radius,weight,BATCH_SIZE,NUM_CENTER,NUM_POINT)
        fetchworker.start()
        for epoch in tqdm(range(restore_epoch+1,MAX_EPOCH+1),ncols=55):
            log_string('**** EPOCH %03d ****\n' % (epoch)) 
            train_one_epoch(sess, ops, fetchworker, train_writer)
            if epoch % 1 == 0:
                saver.save(sess, os.path.join(MODEL_DIR, "model"), global_step=epoch)
        fetchworker.shutdown()

def train_one_epoch(sess, ops, fetchworker, train_writer):
    loss_sum = []
    repulsion_loss_sum=[]
    pre_loss_sum=[]
    fetch_time = 0
    for batch_idx in tqdm(range(fetchworker.num_batches)):
        start = time.time()
        batch_center_data,batch_input_data, batch_data_gt, batch_data_gt2,radius,batch_weight =fetchworker.fetch()
        #print(len(batch_center_data),len(batch_center_data[0]),len(batch_center_data[0][0]),)
        pred_temp=[]
        feed_dict = {ops['pointclouds_cp']: batch_center_data,
                     ops['pointclouds_pl']: batch_input_data,
                     ops['pointclouds_gt']: batch_data_gt,
                     ops['pointclouds_gt2']: batch_data_gt2,
                     ops['pointclouds_radius']: radius,
                     ops['pointclouds_weight']: batch_weight}
        summary,step, _,pred_val, gen_loss,gen_repulsion_loss,pre_gen_loss,learning_rate = sess.run( [ops['pretrain_merged'], ops['step'], ops['pre_gen_train'], ops['pred'], ops['gen_loss'],ops['gen_repulsion_loss'],ops['pre_gen_loss'],ops['learning_rate']], feed_dict=feed_dict)

        train_writer.add_summary(summary, step)
        loss_sum.append(gen_loss)
        repulsion_loss_sum.append(gen_repulsion_loss)
        pre_loss_sum.append(pre_gen_loss)

        end = time.time()
        fetch_time += end-start
        if step%20==0:
            log_string('step: %d time: %s loss: %.10f repulsion_loss: %.10f pre_gen_loss: %.10f\n' % (step, round(fetch_time,4), gen_loss,  gen_repulsion_loss, pre_gen_loss))
    loss_sum = np.asarray(loss_sum)
    repulsion_loss_sum = np.asarray(repulsion_loss_sum)
    pre_loss_sum = np.asarray(pre_loss_sum)
    log_string('**************** step: %d mean gen_loss: %.10f mean gen_repulsion_loss: %.10f mean pre_gen_loss: %.10f\n' % (step, loss_sum.mean(), repulsion_loss_sum.mean(), pre_loss_sum.mean()))
    print('train time: %s mean gen_loss: %.10f mean gen_repulsion_loss: %.10f mean pre_gen_loss: %.10f' % (round(fetch_time,4), loss_sum.mean(), repulsion_loss_sum.mean(), pre_loss_sum.mean()))
    print('step:%d , learning_rate:  %.10f' % (step,learning_rate))

def prediction_whole_model(data_folder=None,use_normal=False):
    data_folder = '../data/test_data/our_collected_data'
    phase = data_folder.split('/')[-2]+data_folder.split('/')[-1]
    save_path = os.path.join(MODEL_DIR, 'result/' + phase)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    knn=128
    _, restore_model_path = model_utils.pre_load_checkpoint(MODEL_DIR)
    print (restore_model_path)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    samples = glob(data_folder+"/*.xyz")
    samples.sort()
    total_time = 0
    #for i,item in enumerate(samples):
    if(FLAGS.sm_idx >=0 and FLAGS.sm_idx < len(samples)):
        print('idx: '+str(FLAGS.sm_idx)+'/'+str(len(samples)))
        item = samples[FLAGS.sm_idx]
        with tf.Session(config=config) as sess:
            input = np.loadtxt(item).astype(np.float32)
            input = np.expand_dims(input, axis=0)#[1,:,3]
            
            if not use_normal:
                input = input[:,:,0:3]
            print (item, input.shape) #name , (1,:,3)

            #center mormal
            data_radius = np.ones(shape=(len(input)))
            centroid = np.mean(input[:, :, 0:3], axis=1, keepdims=True)
            input[:, :, 0:3] = input[:, :, 0:3] - centroid
            furthest_distance = np.amax(np.sqrt(np.sum(input[:, :, 0:3] ** 2, axis=-1)), axis=1, keepdims=True)
            input[:, :, 0:3] = input[:, :, 0:3] / np.expand_dims(furthest_distance, axis=-1)
            start_time = time.time()
            num_point= input.shape[1]
            patch_size = 1000
            num_patch = int(math.ceil(num_point/1000.0))
            output=[]
            for f in tqdm(range(FILTER_TIMES)):
                Trans=[]
                input2=[]
                for m in range(3):
                    #feature_in =[]
                    #feature_out =[]
                    Trans_temp=[]
                    Resi=[]
                    if output == []:
                        if(num_point % patch_size==0):
                            input2=np.copy(input)
                        else:
                            input2 = np.concatenate([input,input[:, num_point % patch_size-patch_size:,:]],axis=1)
                    else:
                        if(num_point % patch_size==0):
                            input2 = np.copy(output)
                        else:
                            input2 = np.concatenate([output,output[:, num_point % patch_size-patch_size:,:]],axis=1)
                    nbrs = spatial.cKDTree(np.squeeze(input2))
                    dist,idxs = nbrs.query(np.squeeze(input2),k=501)

                    rad = np.mean(dist[:,500])  + np.max(dist[:,1]) * 300 * np.mean(dist[:,1])
                    input2 = input2 /rad

                    pointclouds_cipt = tf.placeholder(tf.float32, shape=(1, patch_size, 3))
                    pointclouds_ipt = tf.placeholder(tf.float32, shape=(1, num_point, 3))
                    if(Trans==[]):
                        pred=MODEL_GEN.get_res_model(pointclouds_cipt,pointclouds_ipt, is_training=False,  scope='generator'+str(m),bradius=1.0,
                                                              reuse=tf.AUTO_REUSE, use_bn=False,use_ibn=False,
                                                              bn_decay=0.95, ball_number=BALL_NUMBER, feature_tran_num=FEATURE_TRAN_NUM)
                        saver = tf.train.Saver(var_list=[var for var in tf.global_variables() if var.name.startswith("generator"+str(m))])
                        saver.restore(sess, restore_model_path)
                        for k in range(num_patch):
                            res,tra = sess.run(pred, feed_dict={pointclouds_cipt: input2[:,patch_size*k:patch_size*(k+1),:], pointclouds_ipt: input2[:,:num_point,:]})
                            if k == 0:
                                Resi=np.array(res).astype('float32')
                                Trans_temp=np.array(tra).astype('float32')
                                #feature_in = np.array(tempin).astype('float32')
                                #feature_out = np.array(tempout).astype('float32')
                            else:
                                Resi=np.concatenate([Resi,np.array(res).astype('float32')],axis=1)
                                Trans_temp=np.concatenate([Trans_temp,np.array(tra).astype('float32')],axis=1)
                                #feature_in = np.concatenate([feature_in,np.array(tempin).astype('float32')],axis=1)
                                #feature_out = np.concatenate([feature_out,np.array(tempout).astype('float32')],axis=1)
                        Trans = np.copy(Trans_temp)
                    else:
                        tran_feature_pt = tf.placeholder(tf.float32, shape=(1, patch_size,BALL_NUMBER,(m-1)%2+1,128))
                        pred=MODEL_GEN.get_res_model(pointclouds_cipt,pointclouds_ipt, is_training=False, scope='generator'+str((m-1)%2+1),bradius=1.0,
                                                              reuse=tf.AUTO_REUSE, use_bn=False,use_ibn=False,
                                                              bn_decay=0.95, ball_number=BALL_NUMBER, tran_feature=tran_feature_pt, feature_tran_num=FEATURE_TRAN_NUM)
                        saver = tf.train.Saver(var_list=[var for var in tf.global_variables() if var.name.startswith("generator")])
                        saver.restore(sess, restore_model_path)
                        for k in range(num_patch):
                            res,tra = sess.run(pred, feed_dict={pointclouds_cipt: input2[:,patch_size*k:patch_size*(k+1),:],pointclouds_ipt: input2[:,:num_point,:], tran_feature_pt:Trans[:,patch_size*k:patch_size*(k+1),:,(1-m)%2-1:,:]})
                            if k == 0:
                                Resi=np.array(res).astype('float32')
                                Trans_temp=np.array(tra).astype('float32')
                                #feature_in = np.array(tempin).astype('float32')
                                #feature_out = np.array(tempout).astype('float32')
                            else:
                                Resi=np.concatenate([Resi,np.array(res).astype('float32')],axis=1)
                                Trans_temp=np.concatenate([Trans_temp,np.array(tra).astype('float32')],axis=1)
                                #feature_in = np.concatenate([feature_in,np.array(tempin).astype('float32')],axis=1)
                                #feature_out = np.concatenate([feature_out,np.array(tempout).astype('float32')],axis=1)
                        Trans = np.copy(Trans_temp)
                        #a =  np.copy(Trans)
                        #print(a[1,:,0*1000:0*1000+5,101:106],'\n',a[1,:,1*1000:1*1000+5,100:105],'\n',a[1,:,2*1000:2*1000+5,100:105])

                    resknum = 20+int(num_point/2000)
                    idxs=np.expand_dims(idxs[:,1:resknum], axis=0)
                    rk=np.mean(dist[:,128])
                    dist=np.expand_dims(np.expand_dims(dist[:,1:resknum], axis=-1).repeat(3,axis=-1), axis=0)
                    zero=tf.zeros_like(dist)
                    one=tf.ones_like(dist)
                    dist=tf.to_float(tf.where(dist<rk,x=one,y=zero),name='ToFloat')
                    Resi_knn = tf.reduce_mean(tf.multiply(group_point(Resi, idxs),dist), axis=2)

                    #idxs=np.expand_dims(idxs, axis=0)
                    #Resi_knn = tf.reduce_mean(group_point(Resi, idxs[:,:,:20+int(input.shape[1]/2000)]), axis=2,keep_dims=False)
                    Resi = Resi-Resi_knn

                    output2 = tf.add(input2[:,:num_point,:], Resi[:,:num_point,:])#reduce res
                    output2 = output2-tf.reduce_mean(output2,axis=1,keepdims=True)#move center
                    output2 = output2 * rad
                    output=output2.eval(session=sess)

                    pred_pl= np.copy(output)
                    total_time +=time.time()-start_time

                    # back center mormal
                    pred_pl[:, :, 0:3] = pred_pl[:, :, 0:3] * np.expand_dims(furthest_distance, axis=-1)
                    pred_pl[:, :, 0:3] = pred_pl[:, :, 0:3] + centroid

                    ##--------------visualize predicted point cloud----------------------
                    path = os.path.join(save_path,item.split('/')[-1])

                data_provider.save_pl(path[:-4]+'_pred_iter_'+str(f+1)+'.xyz', pred_pl[0])  ######
            input[:, :, 0:3] = input[:, :, 0:3] * np.expand_dims(furthest_distance, axis=-1)
            input[:, :, 0:3] = input[:, :, 0:3] + centroid
            path = path[:-4]+'_input.xyz'
            data_provider.save_pl(path, input[0])
            print(total_time)

if __name__ == "__main__":
    np.random.seed(int(time.time()))
    tf.set_random_seed(int(time.time()))
    if PHASE=='train':
        # copy the codes
        assert not os.path.exists(os.path.join(MODEL_DIR, 'codes/'))
        os.makedirs(os.path.join(MODEL_DIR, 'codes/'))
        os.system('cp -r * %s' % (os.path.join(MODEL_DIR, 'codes/')))  # bkp of model def

        train(assign_model_path=ASSIGN_MODEL_PATH)
        LOG_FOUT.close()
    else:
        prediction_whole_model()
