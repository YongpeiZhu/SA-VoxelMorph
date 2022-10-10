"""
Test models for MICCAI 2018 submission of VoxelMorph.
"""

# py imports
import os
import sys
import glob

# third party
import tensorflow as tf
import scipy.io as sio
import numpy as np
import keras
from keras.backend.tensorflow_backend import set_session
from scipy.interpolate import interpn

# project
sys.path.append('drive/My Drive/aa/voxelmorph-master/ext/medipy-lib')
import medipy
import networks
# import util
from medipy.metrics import dice
import datagenerators
import nibabel as nib

# Test file and anatomical labels we want to evaluate
test_brain_file = open('drive/My Drive/aa/voxelmorph-master/data1/test_examples.txt')
test_brain_strings = test_brain_file.readlines()
test_brain_strings = [x.strip() for x in test_brain_strings]
n_batches = len(test_brain_strings)
# good_labels = sio.loadmat('../data1/labels.mat')['labels'][0]

# atlas files
# atlas = np.load('../data1/atlas_norm.npz')
atlas1 = nib.load('drive/My Drive/aa/voxelmorph-master/data1/5_reg_T1_brain.nii.gz')
atlas_vol = atlas1.get_data()[np.newaxis, ..., np.newaxis]
# atlas_vol = atlas['vol'][np.newaxis, ..., np.newaxis]
# atlas_seg = atlas['seg']
atlas_seg = nib.load('drive/My Drive/aa/voxelmorph-master/data1/5_reg_T1_brain.nii.gz').get_data()
good_labels=np.zeros_like(atlas_seg)
def test(gpu_id, model_dir, iter_num, 
         compute_type = 'GPU',  # GPU or CPU
         vol_size=(240,240,48),
         nf_enc=[16,32,32,32],
         nf_dec=[32,32,32,32,16,3],
         save_file=None):
    """
    test via segmetnation propagation
    works by iterating over some iamge files, registering them to atlas,
    propagating the warps, then computing Dice with atlas segmentations
    """  

    # GPU handling
    gpu = '/gpu:' + str(gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    set_session(tf.Session(config=config))

    # load weights of model
    with tf.device(gpu):
        # if testing miccai run, should be xy indexing.
        net = networks.miccai2018_net(vol_size, nf_enc, nf_dec, use_miccai_int=False, indexing='ij')  
        net.load_weights(os.path.join(model_dir, str(iter_num) + '.h5'))

        # compose diffeomorphic flow output model
        diff_net = keras.models.Model(net.inputs, net.get_layer('diffflow').output)

        # NN transfer model
        nn_trf_model = networks.nn_trf(vol_size, indexing='ij')

    # if CPU, prepare grid
    if compute_type == 'CPU':
        grid, xx, yy, zz = util.volshape2grid_3d(vol_size, nargout=4)
    
    # prepare a matrix of dice values
    dice_vals = np.zeros((len(good_labels), n_batches))
    dc=[]
    for k in range(n_batches):
        # get data
        vol_name, seg_name = test_brain_strings[k].split(",")
        X_vol, X_seg = datagenerators.load_example_by_name(vol_name, seg_name)

        mov_nii = nib.load('drive/My Drive/aa/voxelmorph-master/data1/5_reg_T1_brain.nii.gz')
        # predict transform
        with tf.device(gpu):
            pred = diff_net.predict([X_vol, atlas_vol])
            flow = pred[0, :, :, :, :]
            img = nib.Nifti1Image(flow, mov_nii.affine)
            # nib.save(img, '../data1/pred4-1.nii.gz')
            nib.save(img, os.path.join('drive/My Drive/aa/voxelmorph-master/data1','pred5'+str(k) + '.nii.gz'))

        # Warp segments with flow
        if compute_type == 'CPU':
            flow = pred[0, :, :, :, :]
            warp_seg = util.warp_seg(X_seg, flow, grid=grid, xx=xx, yy=yy, zz=zz)

        else:  # GPU
            warp_seg = nn_trf_model.predict([X_seg, pred])[0,...,0]
            img = nib.Nifti1Image(warp_seg, mov_nii.affine)
            # nib.save(img,'../data1/warp_seg4-1.nii.gz')
            nib.save(img, os.path.join('drive/My Drive/aa/voxelmorph-master/data1','warp_seg5'+str(k) + '.nii.gz'))
        
        # compute Volume Overlap (Dice)
        dice_vals[:, k] = dice(warp_seg, atlas_seg, labels=good_labels)
        print('%3d %5.3f %5.3f' % (k, np.mean(dice_vals[:, k]), np.mean(np.mean(dice_vals[:, :k+1]))))
        dc = np.append(dc, np.mean(dice_vals[:, k]))
        dc = np.array(dc)
        print(dc)
        print(np.mean(dc))
        if save_file is not None:
            sio.savemat(save_file, {'dice_vals': dice_vals, 'labels': good_labels})

if __name__ == "__main__":
    """
    assuming the model is model_dir/iter_num.h5
    python test_miccai2018.py gpu_id model_dir iter_num
    """
    test(sys.argv[1], sys.argv[2], sys.argv[3])
