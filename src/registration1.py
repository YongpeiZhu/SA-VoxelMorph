"""
Example script to register two volumes with VoxelMorph models

Please make sure to use trained models appropriately.
Let's say we have a model trained to register subject (moving) to atlas (fixed)
One could run:

python register.py --gpu 0 /path/to/test_vol.nii.gz /path/to/atlas_norm.nii.gz --out_img /path/to/out.nii.gz --model_file ../models/cvpr2018_vm2_cc.h5
"""

# py imports
import os
import sys
from argparse import ArgumentParser

# third party
import tensorflow as tf
import numpy as np
import keras
from keras.backend.tensorflow_backend import set_session
import nibabel as nib

# project
import networks, losses

sys.path.append('../ext/neuron')
import neuron.layers as nrn_layers


def register(gpu_id, moving, fixed, model_dir, iter_num, out_img, out_warp,
             vol_size=(160,192,224),nf_enc=[16,32,32,32],nf_dec=[32,32,32,32,16,3]):
    """
    register moving and fixed.
    """
    # assert model_file, "A model file is necessary"
    # assert out_img or out_warp, "output image or warp file needs to be specified"

    # GPU handling
    if gpu_id is not None:
        gpu = '/gpu:' + str(gpu_id)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        set_session(tf.Session(config=config))
    else:
        gpu = '/cpu:0'

    # load data
    mov_nii = nib.load(moving)
    mov = mov_nii.get_data()[np.newaxis, ...,np.newaxis]
    fix_nii = nib.load(fixed)
    fix = fix_nii.get_data()[np.newaxis, ...,np.newaxis]

    with tf.device(gpu):

        # load model
        # custom_objects = {'SpatialTransformer': nrn_layers.SpatialTransformer,
        #                   'VecInt': nrn_layers.VecInt,
        #                   'Sample': networks.Sample,
        #                   'Rescale': networks.RescaleDouble,
        #                   'Resize': networks.ResizeDouble,
        #                   'Negate': networks.Negate,
        #                   'recon_loss': losses.Miccai2018(0.02, 10).recon_loss,  # values shouldn't matter
        #                   'kl_loss': losses.Miccai2018(0.02, 10).kl_loss  # values shouldn't matter
        #                   }
        #
        # net = keras.models.load_model(model_file, custom_objects=custom_objects)
        net = networks.miccai2018_net(vol_size, nf_enc, nf_dec, use_miccai_int=False, indexing='ij')
        net.load_weights(os.path.join(model_dir, str(iter_num) + '.h5'))
        #
        # # compose diffeomorphic flow output model
        diff_net = keras.models.Model(net.inputs, net.get_layer('flow').output)
        nn_trf_model = networks.nn_trf(vol_size, indexing='ij')

        # register
        warp = diff_net.predict([mov, fix])
        # warp_nii = nib.load('/media/xichao/73876AB16BC245D7/voxelmorph/data1/atlas_normphi.nii.gz')
        # print(warp_nii.shape,'llll')
        # warp= warp_nii.get_data()[np.newaxis, ...,]
        # moved= nn_trf_model.predict([mov, warp])

    # output image
    # if out_img is not None:
    #     img = nib.Nifti1Image(moved[0, ..., 0], mov_nii.affine)
    #     nib.save(img, out_img)

    # output warp
    if out_warp is not None:
        img = nib.Nifti1Image(warp[0, ...], mov_nii.affine)
        nib.save(img, out_warp)


if __name__ == "__main__":
    parser = ArgumentParser()

    # positional arguments
    parser.add_argument("moving", type=str, default=None,
                        help="moving file name")
    parser.add_argument("fixed", type=str, default=None,
                        help="fixed file name")

    # optional arguments
    # parser.add_argument("--model_file", type=str,
    #                     dest="model_file", default='../models/cvpr2018_vm1_cc.h5',
    #                     help="models h5 file")
    parser.add_argument("--model_dir", type=str,
                        dest="model_dir", default='../models/',
                        help="models folder")
    parser.add_argument("--iter_num", type=int,
                        dest="iter_num", default=1500,
                        help="number of iterations")
    parser.add_argument("--gpu", type=int, default=None,
                        dest="gpu_id", help="gpu id number")
    parser.add_argument("--out_img", type=str, default=None,
                        dest="out_img", help="output image file name")
    parser.add_argument("--out_warp", type=str, default=None,
                        dest="out_warp", help="output warp file name")

    args = parser.parse_args()
    register(**vars(args))