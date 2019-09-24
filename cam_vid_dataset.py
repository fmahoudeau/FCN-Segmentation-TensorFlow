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

import sys, shutil
import argparse
import os.path
from datetime import datetime
import numpy as np
import tensorflow as tf
import zipfile

from dataset import Dataset
from image_utils import (imread, imwrite, bytesread,
                         colors2labels, labels2colors,
                         pad, center_crop, apply_mask,
                         random_transform)


# Images and segmentation ground truth are in PNG format
TRAIN_IM_PATH = 'images_prepped_train'  # Location of images
TRAIN_GT_PATH = 'annotations_prepped_train'  # Location of ground truth
VAL_IM_PATH = 'images_prepped_test'  # Location of images
VAL_GT_PATH = 'annotations_prepped_test'  # Location of ground truth

parser = argparse.ArgumentParser()

parser.add_argument(
    '--data_dir', type=str, default='/tmp/cam_vid_data/',
    help='Directory where the data is located')


class CamVidDataset(Dataset):
    """Dataset class for CamVid."""

    def __init__(self, augmentation_params):
        super().__init__(augmentation_params)
        self.image_shape = (384, 480)  # Image is padded with (24, 0) to have a shape divisible by 32.
        self.n_classes = 11  # Excluding the ignore/void class
        self.class_labels = ['sky', 'building', 'pole', 'road', 'sidewalk', 'vegetation',
                             'sign/light', 'guard rail', 'car', 'pedestrian', 'bicycle']
        self.n_images = {
            'train': 367,
            'test': 0,
            'val': 101
        }
        self.cmap = np.asarray([
            (0,     0,   0),  # Sky
            (128,  64, 128),  # Building
            (244,  35, 232),  # Pole
            (70,   70,  70),  # Road
            (220, 220,   0),  # Sidewalk
            (107, 142,  35),  # Vegetation
            (152, 251, 152),  # Sign/light
            (70,  130, 180),  # Guard rail
            (220,  20,  60),  # Car
            (0,     0, 142),  # Pedestrian
            (119,  11,  32),  # Bicycle
            (0,    80, 100)]) # Ignore/void
        assert len(self.cmap) == (self.n_classes+1), 'Invalid number of colors in cmap'

    def extract_dataset(self, data_dir):
        if not os.path.exists(os.path.join(data_dir, TRAIN_IM_PATH, '0001TP_006690.png')):
            print('Extracting zip...')
            zip_ref = zipfile.ZipFile(os.path.join(data_dir, 'cam_vid_prepped.zip'), 'r')
            zip_ref.extractall(data_dir)
            zip_ref.close()
        else:
            print('Zip already extracted')
        print('Finished extracting')

    def get_basenames(self, is_training, dataset_path):
        """
        Obtains a list of images base names that have been labelled for semantic segmentation.

        :param is_training: Whether to return the training or validation file names.
        :param dataset_path: The root path of the dataset.
        :return: The sorted list of image base names.
        """
        if is_training:
            basenames = sorted(os.listdir(os.path.join(dataset_path, TRAIN_IM_PATH)))
        else:
            basenames = sorted(os.listdir(os.path.join(dataset_path, VAL_IM_PATH)))
        return basenames

    def export_tfrecord(self, is_training, dataset_path, tfrecord_filename):
        """Exports a semantic image segmentation dataset to TFRecords.

        :param basenames:

        :return: the list of image base names for either the training or validation image set
        """
        print('Loading dataset...')
        basenames = self.get_basenames(is_training, dataset_path)

        # Create folder for TF records
        tfrecords_path = os.path.join(dataset_path, 'TFRecords')
        if not os.path.exists(tfrecords_path):
            print('Creating TFRecords folder')
            os.makedirs(tfrecords_path)
        else:
            print('TFRecords folder already exists')

        im_set, gt_set, shape_set = [], [], []
        for basename in basenames:
            # Save image in raw bytes format
            im = bytesread(os.path.join(dataset_path, TRAIN_IM_PATH if is_training else VAL_IM_PATH, basename))
            # Save ground truth as a ndarray
            gt = imread(os.path.join(dataset_path, TRAIN_GT_PATH if is_training else VAL_GT_PATH, basename))
            shape_set.append(gt.shape)
            im_set.append(im)
            gt_set.append(gt[:, :, 0])

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
        im = tf.image.decode_png(parsed['image_raw'])
        gt = tf.io.decode_raw(parsed['label_raw'], tf.uint8)
        gt = tf.reshape(gt, [height, width])

        return tf.squeeze(im), tf.squeeze(gt), (height, width, depth)

    def load_dataset(self, is_training, data_dir, batch_size):
        """Returns a TFRecordDataset for the requested dataset."""
        data_path = os.path.join(data_dir, 'TFRecords',
                                 'segmentation_{}.tfrecords'.format('train' if is_training else 'val'))
        dataset = tf.data.TFRecordDataset(data_path)

        # Prefetches a batch at a time to smooth out the time taken to load input
        # files for shuffling and processing.
        dataset = dataset.prefetch(buffer_size=batch_size)
        dataset = dataset.map(self.parse_record)

        if is_training:
            dataset = dataset.map(lambda im, gt, _: tuple(tf.compat.v1.py_func(self.transform_record,
                                                                     [im, gt],
                                                                     [im.dtype, tf.uint8])))
            dataset = dataset.shuffle(self.n_images['train'])
        # Remove the shape parameter. It is only needed at prediction time to reshape the logits before masking.
        else:
            dataset = dataset.map(lambda im, gt, _: tuple(tf.compat.v1.py_func(self.pad_record,
                                                                     [im, gt],
                                                                     [im.dtype, tf.uint8])))

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
        dataset = dataset.map(
            lambda im, gt, shape: tuple(tf.compat.v1.py_func(self.pad_record, [im, gt, shape], [im.dtype, tf.uint8, tf.int64])))
        dataset = dataset.batch(batch_size)
        iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
        next_sample = iterator.get_next()

        idx = 0  # The image name is it's index in the TFRecordDataset
        while True:
            try:
                im_batch, _, shape_batch = sess.run(next_sample)

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
    """Export the CAM VID segmentation dataset to TFRecords."""

    dataset = CamVidDataset(augmentation_params=None)

    if not os.path.exists(os.path.join(FLAGS.data_dir, 'cam_vid_prepped.zip')):
        raise ValueError('Dataset zip file not found: {}'.format(
            os.path.join(FLAGS.data_dir, 'cam_vid_prepped.zip')))
    dataset.extract_dataset(FLAGS.data_dir)

    dataset_path = os.path.join(FLAGS.data_dir, 'cam_vid')
    train_basenames = dataset.get_basenames(True, dataset_path)
    print('Found', len(train_basenames), 'training samples')

    val_basenames = dataset.get_basenames(False, dataset_path)
    print('Found', len(val_basenames), 'validation samples')

    # Export train and validation datasets to TFRecords
    dataset.export_tfrecord(True, dataset_path, 'segmentation_train.tfrecords')
    dataset.export_tfrecord(False, dataset_path, 'segmentation_val.tfrecords')
    print('Finished exporting')


if __name__ == '__main__':
    FLAGS, unparsed = parser.parse_known_args()
    tf.compat.v1.app.run(argv=[sys.argv[0]] + unparsed)
