import os.path
import tensorflow as tf
import helper
import shutil
import warnings
from distutils.version import LooseVersion
import project_tests as tests

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()

    vgg_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    vgg_kp = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    vgg_l3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    vgg_l4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    vgg_l7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return vgg_input, vgg_kp, vgg_l3, vgg_l4, vgg_l7

tests.test_load_vgg(load_vgg, tf)

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes, is_training):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :param is_training: param for batch normalization
    :return: The Tensor for the last layer of output
    """

    k_init = tf.contrib.layers.xavier_initializer()
    k_reg = tf.contrib.layers.l2_regularizer(scale=1e-3)

    # #scaling of pool layers
    vgg_layer3_out = tf.multiply(vgg_layer3_out, .0001, name='vgg_layer3_out')
    vgg_layer4_out = tf.multiply(vgg_layer4_out, .01, name='vgg_layer4_out')

    enc_1x1_l3 = tf.layers.conv2d(vgg_layer3_out, 256, 1, padding='same', name='enc_1x1_l3', \
                                  kernel_initializer=k_init, kernel_regularizer=k_reg, activation=tf.nn.elu)
    enc_1x1_l3_bn = tf.layers.batch_normalization(enc_1x1_l3, name="enc_1x1_l3_bn", training=is_training)
    enc_1x1_l3_2 = tf.layers.conv2d(enc_1x1_l3_bn, num_classes, 1, padding='same', name='enc_1x1_l3_2', \
                                  kernel_initializer=k_init, kernel_regularizer=k_reg, activation=tf.nn.elu)
    enc_1x1_l3_2_bn = tf.layers.batch_normalization(enc_1x1_l3_2, name="enc_1x1_l3_2_bn", training=is_training)
    enc_1x1_l4 = tf.layers.conv2d(vgg_layer4_out, 512, 1, padding='same', name='enc_1x1_l4', \
                                  kernel_initializer=k_init, kernel_regularizer=k_reg, activation=tf.nn.elu)
    enc_1x1_l4_bn = tf.layers.batch_normalization(enc_1x1_l4, name="enc_1x1_l4_bn", training=is_training)
    enc_1x1_l4_2 = tf.layers.conv2d(enc_1x1_l4_bn, num_classes, 1, padding='same', name='enc_1x1_l4_2', \
                                    kernel_initializer=k_init, kernel_regularizer=k_reg, activation=tf.nn.elu)
    enc_1x1_l4_2_bn = tf.layers.batch_normalization(enc_1x1_l4_2, name="enc_1x1_l4_2_bn", training=is_training)
    enc_1x1_l7 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding='same', name='enc_1x1_l7', \
                                      kernel_initializer=k_init, kernel_regularizer=k_reg, activation=tf.nn.elu)
    enc_1x1_l7_bn = tf.layers.batch_normalization(enc_1x1_l7, name="enc_1x1_l7_bn", training=is_training)

    dec_l1 = tf.layers.conv2d_transpose(enc_1x1_l7_bn, num_classes, 4, 2, \
                                        padding='same', name='dec_l1', kernel_initializer=k_init, \
                                        kernel_regularizer=k_reg, activation=tf.nn.elu)
    dec_l1_bn = tf.layers.batch_normalization(dec_l1, name="dec_l1_bn", training=is_training)
    dec_l1_1x1 = tf.layers.conv2d(dec_l1_bn, num_classes, 1, padding='same', name='dec_l1_1x1', \
                                            kernel_initializer=k_init, kernel_regularizer=k_reg, activation=tf.nn.elu)
    dec_l1_1x1_bn = tf.layers.batch_normalization(dec_l1_1x1, name="dec_l1_1x1_bn", training=is_training)

    skip_connection1 = tf.add(dec_l1_1x1_bn, enc_1x1_l4_2_bn, name='skip_connection1')

    dec_l2 = tf.layers.conv2d_transpose(skip_connection1, num_classes, 4, 2, \
                                        padding='same', name='dec_l2', kernel_initializer=k_init, \
                                        kernel_regularizer=k_reg, activation=tf.nn.elu)
    dec_l2_bn = tf.layers.batch_normalization(dec_l2, name="dec_l2_bn", training=is_training)
    dec_l2_1x1 = tf.layers.conv2d(dec_l2_bn, num_classes, 1, padding='same', name='dec_l2_1x1', \
                                  kernel_initializer=k_init, kernel_regularizer=k_reg, activation=tf.nn.elu)
    dec_l2_1x1_bn = tf.layers.batch_normalization(dec_l2_1x1, name="dec_l2_1x1_bn", training=is_training)

    skip_connection2 = tf.add(dec_l2_1x1_bn, enc_1x1_l3_2_bn, name='skip_connection2')

    dec_l3 = tf.layers.conv2d_transpose(skip_connection2, num_classes, 16, 8, padding='same', name='dec_l3', \
                                        kernel_initializer=k_init, kernel_regularizer=k_reg, activation=tf.nn.elu)
    dec_l3_bn = tf.layers.batch_normalization(dec_l3, name="dec_l3_bn", training=is_training)
    dec_l3_1x1 = tf.layers.conv2d(dec_l3_bn, num_classes, 1, padding='same', name='dec_l3_1x1', \
                                  kernel_initializer=k_init, kernel_regularizer=k_reg, activation=tf.nn.elu)
    return dec_l3_1x1

tests.test_layers(layers)

def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logits_flatten = tf.reshape(nn_last_layer, [-1, num_classes])
    label_flatten = tf.reshape(correct_label, [-1, num_classes])
    cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits_flatten, labels=label_flatten)
    loss = tf.reduce_mean(cross_entropy_loss)
    loss += tf.losses.get_regularization_loss()

    # for tf.layers.batch_normalization, when training, the moving_mean and moving_variance need to be updated.
    # refer to https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    return logits_flatten, optimizer, loss

tests.test_optimize(optimize)

model_dir = './model'
def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, is_training, logits):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    :param is_training: param for batch normalization
    :param logits: TF Tensor for the logits
    :param test_img: test image to visualize current segmentation result
    """

    print("Training start")
    print("epochs: {}".format(epochs))
    # Create a saver object which will save all the variables
    test_img_name = 'um_000061.png'
    test_img = helper.imread_img('test_image', test_img_name)

    for epoch in range(epochs):
        for images, gt_images in get_batches_fn(batch_size):
            # print('input_image : {}'.format(input_image.shape))
            # print('images : {}'.format(images.shape))
            _, loss = sess.run([train_op, cross_entropy_loss],
                               feed_dict={input_image: images, correct_label: gt_images,
                                          keep_prob: 0.7, learning_rate: 1e-4, is_training: True})

        # To save training visualized image progress
        if epoch > 0:
            helper.visual_seg_progress(sess, epoch, keep_prob, input_image, logits, is_training, test_img)

        # save the model every 100 epoch
        if not (epoch + 1) % 25:
            saver = tf.train.Saver()
            saver.save(sess, model_dir + '/test_model', epoch + 1)
        print("epoch {} done.".format(epoch))
    print("Training finished.")

tests.test_train_nn(train_nn)

def run(mode='training', test_type='image', video_name=None):
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    type_dir = './data' if test_type == 'image' else './video'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    if mode == 'testing':
        pass
    else:
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        os.makedirs(model_dir)

    correct_label = tf.placeholder(tf.float32, shape=(None, image_shape[0], image_shape[1], 2))
    learning_rate = tf.placeholder(tf.float32)
    is_training_ph = tf.placeholder(tf.bool)

    epochs = 200
    batch_size = 8

    # training
    if mode == 'training':
        is_training = True
        with tf.Session() as sess:
            # Path to vgg model
            vgg_path = os.path.join(data_dir, 'vgg')
            # Create function to get batches
            get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

            # Build NN using load_vgg, layers, and optimize function
            input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
            output = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes, is_training)
            logits, train_op, cross_entropy_loss = optimize(output, correct_label, learning_rate, num_classes)

            # initialize tensorflow variables
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

            # Train NN using the train_nn function
            train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image, correct_label,
                     keep_prob, learning_rate, is_training_ph, logits)

            print("model saved")

    # testing
    else:
        is_training = False
        with tf.Session() as sess:
            # initialize tensorflow variables
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            # Path to vgg model
            vgg_path = os.path.join(data_dir, 'vgg')
            # Build NN using load_vgg, layers, and optimize function
            input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
            output = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes, is_training)
            logits, _, _ = optimize(output, correct_label, learning_rate, num_classes)

            # Create a saver object which will restore all the variables
            saver = tf.train.Saver()
            # Restore variables from disk.
            saver.restore(sess, "./test_model")

            # Save inference data using helper.save_inference_samples
            helper.save_inference_samples(runs_dir, type_dir, sess, image_shape, logits, \
                                          keep_prob, input_image, test_type, video_name, is_training_ph)


if __name__ == '__main__':
    mode = 'training' # 'testing | training'
    test_type = 'image' # 'image | video'
    video_name = 'challenge_video.mp4'

    run(mode, test_type, video_name)
