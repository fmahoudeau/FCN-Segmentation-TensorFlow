# Copyright 2019 Florent Mahoudeau. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import sys, shutil, glob
import argparse
import os.path
from datetime import datetime
import numpy as np
import zipfile
import tensorflow as tf

from dataset import Dataset
from image_utils import (imread, imwrite, bytesread,
                         colors2labels, labels2colors,
                         pad, center_crop, apply_mask,
                         random_transform)

# Images and segmentation ground truth are in PNG format
TRAIN_IM_PATH = 'training/image_2'  # Location of images
TRAIN_GT_PATH = 'training/gt_image_2'  # Location of ground truth
VAL_IM_PATH = 'testing/image_2'  # Location of images

# Location for saving ground truth pixels' labels in sparse format
GT_SPARSE_PATH = 'training/gt_sparse_2'

TRAIN_SHARE = 0.8

parser = argparse.ArgumentParser()

parser.add_argument(
    '--data-dir', type=str, default='/tmp/kitty_road_data/',
    help='Directory where the data is located')


class KittyRoadDataset(Dataset):
    """Base class for building the FCN model."""

    def __init__(self, augmentation_params):
        super().__init__(augmentation_params)
        self.image_shape = (192, 640)  # Image is re-sized to be smaller with shape divisible by 32.
        self.n_classes = 2  # Excluding the ignore/void class
        self.class_labels = ['background', 'road', 'void']
        self.n_images = {
            'train': int(289 * TRAIN_SHARE),
            'test': 289 - int(289 * TRAIN_SHARE),
            'val': 290
        }
        self.cmap = [[255, 0, 0], [255, 0, 255], [0, 0, 0]]
        assert len(self.cmap) == (self.n_classes + 1), 'Invalid number of colors in cmap'

    def extract_dataset(self, data_dir):
        if not os.path.exists(os.path.join(data_dir, TRAIN_IM_PATH, 'um_000000.png')):
            print('Extracting zip...')
            zip_ref = zipfile.ZipFile(os.path.join(data_dir, 'data_road.zip'), 'r')
            zip_ref.extractall(data_dir)
            zip_ref.close()
        else:
            print('Zip already extracted')
        print('Finished extracting')

    def get_basenames(self, is_training, dataset_path):
        """
        Loads a list of images base names that have been labelled for semantic segmentation.
        For the training set, returns a list of tuples with the image and ground truth file names.
        For the validation set, returns only the list of image file names as labels are not available.

        :param is_training: Whether to return the training or validation file names.
        :param dataset_path: The root path of the dataset.
        :return: The sorted list of image base names.
        """
        if is_training:
            gt_road_basenames = [f.split(os.sep)[-1] for f in glob.glob(dataset_path + '/' + TRAIN_GT_PATH + '/*road*.png')]
            basenames = list(zip(sorted(os.listdir(os.path.join(dataset_path, TRAIN_IM_PATH))), sorted(gt_road_basenames)))
        else:
            basenames = sorted(os.listdir(os.path.join(dataset_path, VAL_IM_PATH)))
        return basenames

    def train_test_split(self, basenames, train_size):
        """Splits a dataset to create a test set"""
        np.random.seed(42)  # Enforce reproducibility
        np.random.shuffle(basenames)
        split_samples = int(train_size * len(basenames))
        train_basenames = basenames[:split_samples]
        test_basenames = basenames[split_samples:]
        print('Split', len(train_basenames), 'training samples and', len(test_basenames), 'test samples')
        assert len(basenames) == len(train_basenames)+len(test_basenames)
        return train_basenames, test_basenames

    def export_sparse_encoding(self, dataset_path):
        """
        Converts ground truth images to sparse labels and saves them to disk in PNG format.
        Ground truth images are only available for the training set.

        :param dataset_path: The root path of the dataset.

        :return: None
        """
        # Load the list of image base names
        basenames = self.get_basenames(is_training=True, dataset_path=dataset_path)

        gt_path = os.path.join(dataset_path, TRAIN_GT_PATH)
        gt_sparse_path = os.path.join(dataset_path, GT_SPARSE_PATH)

        # Create sparse labels folder
        if not os.path.exists(gt_sparse_path):
            print('Creating sparse labels folder')
            os.makedirs(gt_sparse_path)
        else:
            print('Sparse labels folder already exists')

        for _, basename in basenames:
            gt = imread(os.path.join(gt_path, basename))
            gt = colors2labels(gt, self.cmap, one_hot=False)
            gt = np.dstack([gt, np.copy(gt), np.copy(gt)])
            imwrite(os.path.join(gt_sparse_path, basename), gt)

    def export_tfrecord(self, basenames, dataset_path, tfrecord_filename):
        """Exports a semantic image segmentation dataset to TFRecords.

        :param basenames:

        :return: the list of image base names for either the training or validation image set
        """
        print('Loading dataset...')
        # Create folder for TF records
        tfrecords_path = os.path.join(dataset_path, 'training/TFRecords')
        if not os.path.exists(tfrecords_path):
            print('Creating TFRecords folder')
            os.makedirs(tfrecords_path)
        else:
            print('TFRecords folder already exists')

        im_set, gt_set, shape_set = [], [], []
        for basename in basenames:
            # Save image in raw bytes format
            im = bytesread(os.path.join(dataset_path, TRAIN_IM_PATH, basename[0]))
            # Save ground truth as a ndarray
            gt = imread(os.path.join(dataset_path, TRAIN_GT_PATH, basename[1]))
            shape_set.append(gt.shape)
            gt = colors2labels(gt, self.cmap)
            im_set.append(im)
            gt_set.append(gt)

        print('Saving to ' + tfrecord_filename)
        self._export(im_set, gt_set, shape_set, os.path.join(tfrecords_path, tfrecord_filename))

    def parse_record(self, record_serialized):
        """
        Parses a sample proto. Each sample proto contains the following fields (values are included as examples):
            height: 281
            width: 500
            channels: 3
            format: 'JPEG'
            filename: '2007_000032.JPEG'
            image_raw: <JPEG encoded string>
            label_raw: <Numpy array encoded string>

        :param record_serialized: scalar Tensor tf.string containing a serialized sample protocol buffer.
        :return:
            image: Tensor tf.uint8 containing the decoded JPEG file.
            labels: Tensor tf.int32 containing the image's pixels' labels.
            shape: list of float Tensors describing the image shape: [height, width, channels].
        """
        keys_to_features = {
            'height': tf.io.FixedLenFeature([1], tf.int64),
            'width': tf.io.FixedLenFeature([1], tf.int64),
            'depth': tf.io.FixedLenFeature([1], tf.int64),
            'image_raw': tf.io.FixedLenFeature([], tf.string),
            'label_raw': tf.io.FixedLenFeature([], tf.string)
        }
        parsed = tf.io.parse_single_example(serialized=record_serialized, features=keys_to_features)

        # Decode raw data
        height = tf.cast(parsed["height"][0], tf.int64)
        width = tf.cast(parsed["width"][0], tf.int64)
        depth = tf.cast(parsed["depth"][0], tf.int64)
        im = tf.expand_dims(tf.image.decode_png(parsed["image_raw"]), 0)
        gt = tf.io.decode_raw(parsed['label_raw'], tf.uint8)
        gt = tf.reshape(gt, [height, width])
        # Perform additional pre-processing
        gt = tf.expand_dims(gt, -1)  # Append channel axis
        gt = tf.expand_dims(gt, 0)  # Insert batch axis
        im = tf.image.resize(im, self.image_shape, method=tf.image.ResizeMethod.BILINEAR)  # Returns a float32 type
        im = tf.cast(im, tf.uint8)
        gt = tf.image.resize(gt, self.image_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # For future use
        # imgt = tf.stack([im, gt], 0)
        # imgt = tf.image.random_flip_left_right(imgt)
        # im, gt = tf.split(imgt, 2, 0)

        return tf.squeeze(im), tf.squeeze(gt), (height, width, depth)

    def load_dataset(self, is_training, data_dir, batch_size):
        """Returns a TFRecordDataset for the requested dataset."""
        data_path = os.path.join(data_dir, 'training/TFRecords',
                                 'segmentation_{}.tfrecords'.format('train' if is_training else 'test'))
        if not os.path.exists(data_path):
            raise ValueError('Check dataset path: {}'.format(data_path))
        dataset = tf.data.TFRecordDataset(data_path)

        # Prefetches a batch at a time to smooth out the time taken to load input
        # files for shuffling and processing.
        dataset = dataset.prefetch(buffer_size=batch_size)
        dataset = dataset.map(self.parse_record)

        if is_training:
            """
            WARNING:tensorflow:From /home/fanos/PycharmProjects/TF2FCN/kitty_road_dataset.py:221: 
            py_func (from tensorflow.python.ops.script_ops) is deprecated and will be removed in a future version.
            Instructions for updating:
            tf.py_func is deprecated in TF V2. Instead, there are two
                options available in V2.
                - tf.py_function takes a python function which manipulates tf eager
                tensors instead of numpy arrays. It's easy to convert a tf eager tensor to
                an ndarray (just call tensor.numpy()) but having access to eager tensors
                means `tf.py_function`s can use accelerators such as GPUs as well as
                being differentiable using a gradient tape.
                - tf.numpy_function maintains the semantics of the deprecated tf.py_func
                (it is not differentiable, and manipulates numpy arrays). It drops the
                stateful argument making all functions stateful.            
            """
            dataset = dataset.map(lambda im, gt, _: tuple(tf.compat.v1.numpy_function(self.transform_record,
                                                                                      [im, gt],
                                                                                      [im.dtype, tf.uint8])))
            dataset = dataset.shuffle(self.n_images['train'])
        # Remove the shape parameter. It is only needed at prediction time to reshape the logits before masking.
        else:
            dataset = dataset.map(lambda im, gt, _: (im, gt))

        return dataset.batch(batch_size)

    def predict_dataset(self, save_path, dataset_filepath, model, batch_size):
        """
        Predicts semantic labels for all images of the speficied dataset and saves results to disk.

        :param save_path: The target directory. A sub-directory will be created from the current date and time.
        :param dataset_filepath: The filename of the TFRecordDataset to use for prediction.
        :param model: An instance of FCN Model.
        :param batch_size: The number of images per batch.
        :return: None
        """
        if not os.path.exists(dataset_filepath):
            raise ValueError('File not found: {}'.format(dataset_filepath))

        sess = tf.compat.v1.get_default_session()

        # Make the folder to save the predictions
        output_path = os.path.join(save_path, datetime.now().isoformat().split('.')[0]).split(':')
        output_path = output_path[0] + ':' + output_path[1] + 'H' + output_path[2]
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        print('Saving predictions to ' + output_path)
        os.makedirs(output_path)

        # Load the dataset and make an iterator
        dataset = tf.data.TFRecordDataset(dataset_filepath)
        dataset = dataset.map(self.parse_record)
        dataset = dataset.batch(batch_size)
        iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
        next_sample = iterator.get_next()

        idx = 0  # The image name is it's index in the TFRecordDataset
        while True:
            try:
                im_batch, _, shape_batch = sess.run(next_sample)
                # Make an array from a tuple of 3 lists each with `batch_size` elements
                shape_batch = np.swapaxes(np.asarray(shape_batch), 0, 1)
                # Returns a 1-item list containing a numpy vector of length BATCH_SIZE * N_PIXELS * N_CLASSES
                im_softmax = sess.run([tf.nn.softmax(model.logits)], {model.keep_prob: 1.0,
                                                                      model.inputs: im_batch})[0]
                im_softmax = im_softmax.reshape((len(im_batch), np.prod(model.image_shape), self.n_classes+1))

                for i in range(len(im_batch)):
                    # Predict pixel class and expand with a channel dimension.
                    im_pred = np.argmax(im_softmax[i], axis=1).reshape(model.image_shape)
                    im_pred = labels2colors(im_pred, self.cmap)
                    im_masked = center_crop(apply_mask(im_batch[i], im_pred), shape_batch[i][:2])
                    imwrite(os.path.join(output_path, str(idx) + '.jpg'), im_masked)
                    idx += 1
            except tf.errors.OutOfRangeError:
                break


def main(_):
    """Export the Kitty Road segmentation dataset to TFRecords."""
    
    dataset = KittyRoadDataset(augmentation_params=None)

    if not os.path.exists(os.path.join(FLAGS.data_dir, 'data_road.zip')):
        raise ValueError('Dataset zip file not found: {}'.format(
            os.path.join(FLAGS.data_dir, 'data_road.zip')))
    dataset.extract_dataset(FLAGS.data_dir)

    dataset_path = os.path.join(FLAGS.data_dir, 'data_road')
    basenames = dataset.get_basenames(True, dataset_path)
    print('Found', len(basenames), 'training samples')

    train_basenames, test_basenames = dataset.train_test_split(basenames, TRAIN_SHARE)

    print('Exporting ground truth images to sparse labels...')
    dataset.export_sparse_encoding(dataset_path)

    # Export train and test datasets to TFRecords
    dataset.export_tfrecord(train_basenames, dataset_path, 'segmentation_train.tfrecords')
    dataset.export_tfrecord(test_basenames, dataset_path, 'segmentation_test.tfrecords')
    print('Finished exporting')


if __name__ == '__main__':
    FLAGS, unparsed = parser.parse_known_args()
    tf.compat.v1.app.run(argv=[sys.argv[0]] + unparsed)
