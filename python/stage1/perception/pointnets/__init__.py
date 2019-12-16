import tensorflow as tf

from perception.pointnets.models import frustum_pointnets_v1 as model

def init_model(pointnet_model_file, batch_size=1, num_point=1024):
    """ Define model graph, load model parameters,
    create session and return session handle and tensors
    """
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(0)):
            (
            pointclouds_pl,
            one_hot_vec_pl,
            labels_pl,
            centers_pl,
            heading_class_label_pl,
            heading_residual_label_pl,
            size_class_label_pl,
            size_residual_label_pl
            ) = model.placeholder_inputs(batch_size, num_point)

            is_training_pl = tf.placeholder(tf.bool, shape=())

            end_points = model.get_model(
                    pointclouds_pl,
                    one_hot_vec_pl,
                    is_training_pl
                )

            loss = model.get_loss(
                    labels_pl,
                    centers_pl,
                    heading_class_label_pl,
                    heading_residual_label_pl,
                    size_class_label_pl,
                    size_residual_label_pl,
                    end_points
                )
            
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True

        sess = tf.Session(config=config)

        # Restore variables from disk.
        saver.restore(sess, pointnet_model_file)
        ops = {'pointclouds_pl': pointclouds_pl,
               'one_hot_vec_pl': one_hot_vec_pl,
               'labels_pl': labels_pl,
               'centers_pl': centers_pl,
               'heading_class_label_pl': heading_class_label_pl,
               'heading_residual_label_pl': heading_residual_label_pl,
               'size_class_label_pl': size_class_label_pl,
               'size_residual_label_pl': size_residual_label_pl,
               'is_training_pl': is_training_pl,
               'logits': end_points['mask_logits'],
               'center': end_points['center'],
               'end_points': end_points,
               'loss': loss}
        
        return sess, ops
