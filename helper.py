import re
import random
import numpy as np
import os.path
import scipy.misc
from scipy.ndimage import gaussian_filter
import shutil
import zipfile
import time
import cv2
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))


def gen_batch_function(data_folder, image_shape):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
        label_paths = {
            re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
            for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}
        background_color = np.array([255, 0, 0])

        random.shuffle(image_paths)
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_images = []
            for image_file in image_paths[batch_i:batch_i+batch_size]:
                gt_image_file = label_paths[os.path.basename(image_file)]

                image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
                gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)

                gt_bg = np.all(gt_image == background_color, axis=2)
                gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
                gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)

                images.append(image)
                gt_images.append(gt_image)

            yield np.array(images), np.array(gt_images)
    return get_batches_fn


def gen_test_output(sess, logits, keep_prob, image_pl, data_folder,
                    image_shape, type='image', video_name=None, is_training=False):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :param type: test data type(image file or video)
    :param video_name: video file's name
    :param is_training: param for batch normalization(False for testing)
    :return: Output for for each test image
    """

    if type == 'image':
        for image_file in glob(os.path.join(data_folder, 'image_2', '*.png')):
            image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)

            im_softmax = sess.run(
                [tf.nn.softmax(logits)],
                feed_dict={keep_prob: 1.0, image_pl: [image], is_training: False})
            im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
            segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1])
            segmentation = segmentation.reshape(image_shape[0], image_shape[1], 1)
            mask = np.dot(segmentation, np.array([[200, 0, 0, 127]]))
            mask = scipy.misc.toimage(mask, mode="RGBA")
            street_im = scipy.misc.toimage(image)
            street_im.paste(mask, box=None, mask=mask)
            yield os.path.basename(image_file), np.array(street_im)
    else:
        cap = cv2.VideoCapture(os.path.join(data_folder, video_name))
        while (cap.isOpened()):
            _, image = cap.read()
            if image is None:
                break
            height, width = image.shape[0], image.shape[1]
            image = scipy.misc.imresize(image, image_shape)
            im_softmax = sess.run(
                [tf.nn.softmax(logits)],
                {keep_prob: 1.0, image_pl: [image], is_training: False})

            im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
            segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1])
            #road_prob = im_softmax * segmentation
            segmentation = segmentation.reshape(image_shape[0], image_shape[1], 1)
            mask = np.dot(segmentation, np.array([[0, 0, 200, 127]]))
            #mask[:, :, 2] = np.int32(mask[:, :, 2] * (road_prob ** 2))
            mask = scipy.misc.toimage(mask, mode="RGBA")
            street_im = scipy.misc.toimage(image)
            street_im.paste(mask, box=None, mask=mask)
            image = scipy.misc.imresize(street_im, (height // 2, width // 2))
            yield np.array(image)


def save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob,
                           input_image, test_type='image', video_name=None, is_training=False):

    if not os.path.exists(data_dir):
        raise NameError('Please check the data file direction.')

    if test_type == 'image':
        print('image testing start')
        # Make folder for current run
        output_dir = os.path.join(runs_dir, str(time.time()))
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        # Run NN on test images and save them to HD
        print('Saving test images to: {}'.format(output_dir))
        image_outputs = gen_test_output(
            sess, logits, keep_prob, input_image, os.path.join(data_dir, 'data_road/testing'),
            image_shape, is_training=is_training)
        for name, image in image_outputs:
            scipy.misc.imsave(os.path.join(output_dir, name), image)
    else: # video data testing
        print('video testing start')
        image_outputs = gen_test_output(sess, logits, keep_prob, input_image,
                                        data_dir, image_shape, test_type, video_name, is_training)

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        vid = cv2.VideoWriter('result.avi', fourcc, 28.0, (640, 360))

        """ save result video """
        for frame in image_outputs:
            vid.write(frame)

        print('video saved')
        vid.release()


def imread_img(img_dir, name):
    img = scipy.misc.imread(os.path.join(img_dir, name))
    return img


def visual_seg_progress(sess, epoch, keep_prob1, input_image, logits, is_training, test_img):
    output_dir = './visual_img'
    image_shape = (160, 576)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    height, width = test_img.shape[0], test_img.shape[1]
    image = scipy.misc.imresize(test_img, image_shape)
    # save current epoch's segmentation result
    im_softmax = sess.run( [tf.nn.softmax(logits)], feed_dict={input_image: [image],
                 keep_prob1: 1.0, is_training: False})
    im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
    im_softmax[im_softmax < 0.5] = 0
    im_softmax = im_softmax.reshape(image_shape[0], image_shape[1], 1)
    im_softmax = np.dot(im_softmax, np.array([[200, 0, 0, 127]]))
    output = scipy.misc.toimage(im_softmax, mode="RGBA")
    image = scipy.misc.toimage(image)
    image.paste(output, box=None, mask=output)
    output = scipy.misc.imresize(image, (height, width))
    scipy.misc.imsave(os.path.join(output_dir,'result_' + str(epoch) + '.png'), output)

