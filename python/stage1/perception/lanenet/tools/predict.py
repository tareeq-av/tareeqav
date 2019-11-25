import os
import sys

sys.path.append('../../')
sys.path.append('../')
sys.path.append('./')

import cv2
import glog as log
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time

from config import global_config
from lanenet_model import lanenet
from lanenet_model import lanenet_postprocess

CFG = global_config.cfg
weight_path='./model/tusimple_lanenet_vgg/tusimple_lanenet_vgg.ckpt'


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='The image path or the src image save dir')
    parser.add_argument('--weights_path', type=str, help='The model tusimple_lanenet_vgg path')

    return parser.parse_args()


def args_str2bool(arg_value):
    """

    :param arg_value:
    :return:
    """
    if arg_value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True

    elif arg_value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def minmax_scale(input_arr):
    """

    :param input_arr:
    :return:
    """
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)

    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

    return output_arr



def infer(filename):
    """Video inferrence

    Args:
        filename:  video filename with "avi", "mp4" suffix
    """
    log.info('Start reading image and preprocessing')
    t_start = time.time()
    orig_h, orig_w = 720, 1280
    resized_h, resized_w = 256, 384

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('result.avi', fourcc, 30.0, (orig_w, orig_h))

    # use the trained model

    # read the video with height and width
    #start_frame_num = 1800
    start_frame_num = 0
    cap = cv2.VideoCapture(filename)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, orig_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, orig_h)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_num)

    filter_mask = None
    orig_win_name = 'Orig Img'
    raw_win_name = 'Raw Mask'
    cv2.namedWindow(orig_win_name, 0)

    net = lanenet.LaneNet(phase='test', net_flag='vgg')
    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='lanenet_model')
    postprocessor = lanenet_postprocess.LaneNetPostProcessor()
    saver = tf.train.Saver()
    # Set sess configuration
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    out = cv2.VideoWriter('result.avi', fourcc, 1.0, (resized_w, resized_h))

    while True:
        ret, orig_img = cap.read()
        if ret:
            # preprocess img
            frame_num = cap.get(cv2.CAP_PROP_POS_FRAMES)
            text = "TareeqAV--Frame #: " + str(frame_num)
            # img, temp_img = procNet.preprocess_img(orig_img, transform, \
            #                                       resized_h, resized_w)
            ############################################################
            image = orig_img
            image_vis = image

            image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
            image_resized=image
            image = image / 127.5 - 1.0
            log.info('Image load complete, cost time: {:.5f}s'.format(time.time() - t_start))


            with sess.as_default():

                saver.restore(sess=sess, save_path=weight_path)

                t_start = time.time()
                binary_seg_image, instance_seg_image = sess.run(
                    [binary_seg_ret, instance_seg_ret],
                    feed_dict={input_tensor: [image]}
                )
                t_cost = time.time() - t_start
                log.info('Single imgae inference cost time: {:.5f}s'.format(t_cost))

                postprocess_result = postprocessor.postprocess(
                    binary_seg_result=binary_seg_image[0],
                    instance_seg_result=instance_seg_image[0],
                    source_image=image_vis
                )


                mask_image = postprocess_result['mask_image']

                for i in range(CFG.TRAIN.EMBEDDING_FEATS_DIMS):
                    instance_seg_image[0][:, :, i] = minmax_scale(instance_seg_image[0][:, :, i])
                embedding_image = np.array(instance_seg_image[0], np.uint8)
                # binary_image = np.array(binary_seg_image[0], np.uint8)




            # plt.figure('mask_image')
            # plt.imshow(mask_image[:, :, (2, 1, 0)])
            # plt.figure('src_image')
            # plt.imshow(image_vis[:, :, (2, 1, 0)])
            # plt.figure('instance_image')
            # plt.imshow(embedding_image[:, :, (2, 1, 0)])
            # plt.figure('binary_image')
            # plt.imshow(binary_seg_image[0] * 255, cmap='gray')
            # plt.show()
            ###########################################################
            img_combined = np.zeros_like(embedding_image[:, :, (2, 1, 0)])
            img_combined[:, :, 0] =binary_seg_image[0]
            img_combined[:, :, 1] = binary_seg_image[0]
            img_combined[:, :, 2] = binary_seg_image[0]

            img_combined=cv2.multiply(img_combined, embedding_image[:, :, (2, 1, 0)])
            result = cv2.addWeighted(image_resized, 1, img_combined, 0.9, 0)

            cv2.imshow(orig_win_name, result)
            cv2.imshow("instance_image", embedding_image[:, :, (2, 1, 0)])
            cv2.imshow("binary_image", np.uint8(binary_seg_image[0]*255))
            cv2.imshow("combined_image", img_combined)
            out.write(img_combined)


            # out.write(mask_resized)
            k = cv2.waitKey(1)
            if k == 27:
                break
        else:
            break
    sess.close()
    cv2.destroyAllWindows()
    cap.release()
    out.release()

if __name__ == "__main__":
    infer(sys.argv[1])