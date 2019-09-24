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
import matplotlib.pyplot as plt
import tensorflow as tf

from dataset import Dataset
from image_utils import (imread, imwrite, bytesread,
                         colors2labels, labels2colors,
                         pad, center_crop, apply_mask,
                         random_transform)


parser = argparse.ArgumentParser()

parser.add_argument(
    '--data_dir', type=str, default='/tmp/pascal_voc_data/',
    help='Directory where the data is located')


class PascalVOC2012Dataset(Dataset):
    """Dataset class for PASCAL VOC 2012."""

    def __init__(self, augmentation_params):
        super().__init__(augmentation_params)
        self.image_shape = (512, 512)  # Image is padded to obtain a shape divisible by 32.
        self.n_classes = 21  # Excluding the ignore/void class
        self.class_labels = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                             'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                             'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor', 'void']
        self.n_images = {
            'train': 1464,
            'val': 1449
        }
        self.cmap = self.color_map()
        assert len(self.cmap) == (self.n_classes + 1), 'Invalid number of colors in cmap'

    def _color_map(self, n_classes=256, normalized=False):
        """
        Builds the PASCAL VOC color map for the specified number of classes.

        :param n_classes: the number of classes in the colormap
        :param normalized: normalize pixel intensities, default is False
        :return: a list of RGB colors
        """
        def _bitget(byteval, idx):
            return (byteval & (1 << idx)) != 0

        dtype = 'float32' if normalized else 'uint8'
        cmap = np.zeros((n_classes, 3), dtype=dtype)
        for i in range(n_classes):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (_bitget(c, 0) << 7-j)
                g = g | (_bitget(c, 1) << 7-j)
                b = b | (_bitget(c, 2) << 7-j)
                c = c >> 3

            cmap[i] = np.array([r, g, b])

        cmap = cmap/255 if normalized else cmap
        return cmap

    def color_map_viz(self, class_labels):
        """
        Plots the PASCAL VOC color map using the specified class labels.
        The number of classes is inferred from the length of the `class_labels` parameter.

        :param class_labels: the list of class labels
        :return: None
        """
        n_classes = len(class_labels) - 1
        row_size = 50
        col_size = 500
        cmap = self._color_map()
        array = np.empty((row_size*(n_classes+1), col_size, cmap.shape[1]), dtype=cmap.dtype)
        for i in range(n_classes):
            array[i*row_size:i*row_size+row_size, :] = cmap[i]
        array[n_classes*row_size:n_classes*row_size+row_size, :] = cmap[-1]

        plt.imshow(array)
        plt.yticks([row_size*i+row_size/2 for i in range(n_classes+1)], class_labels)
        plt.xticks([])
        plt.show()

    def color_map(self):
        """
        Builds the standard 21 class PASCAL VOC color map, plus one additional
        void/ignore label appended to the end of the list.

        :return: A list of RGB values.
        """
        cmap = self._color_map()
        cmap = np.vstack([cmap[:self.n_classes], cmap[-1].reshape(1, 3)])
        return cmap

    def get_basenames(self, filename, dataset_path):
        """
        Obtains the list of image base names that have been labelled for semantic segmentation.
        Images are stored in JPEG format, and segmentation ground truth in PNG format.

        :param filename: The dataset name, either 'train', 'val' or 'test'.
        :param dataset_path: The root path of the dataset.
        :return: The list of image base names for either the training, validation, or test set.
        """
        assert filename in ('train', 'val', 'test')
        filename = os.path.join(dataset_path, 'ImageSets/Segmentation/', filename+'.txt')
        return [line.rstrip() for line in open(filename)]

    def export_sparse_encoding(self, filename, dataset_path):
        """
        Converts ground truth images to sparse labels and saves them to disk in PNG format.

        :param filename:
        :param dataset_path:
        :return: None
        """
        # Load the list of image base names
        basenames = self.get_basenames(filename, dataset_path)

        gt_path = os.path.join(dataset_path, 'SegmentationClass')
        gt_sparse_path = os.path.join(dataset_path, 'SegmentationSparseClass')

        # Create sparse labels folder
        if not os.path.exists(gt_sparse_path):
            print('Creating sparse labels folder')
            os.makedirs(gt_sparse_path)
        else:
            print('Sparse labels folder already exists')

        for basename in basenames:
            gt = imread(os.path.join(gt_path, basename + '.png'))
            gt = colors2labels(gt, self.cmap, one_hot=False)
            gt = np.dstack([gt, np.copy(gt), np.copy(gt)])
            imwrite(os.path.join(gt_sparse_path, basename + '.png'), gt)

    def export_tfrecord(self, filename, dataset_path, tfrecord_filename):
        """
        Exports a semantic image segmentation dataset to TFRecords.
        Images are stored in JPEG format, and segmentation ground truth in PNG format.

        :param filename: the text file name, either 'train.txt' or 'val.txt'

        :return: the list of image base names for either the training or validation image set
        """
        print('Loading images...')
        basenames = self.get_basenames(filename, dataset_path)

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
            im = bytesread(os.path.join(dataset_path, 'JPEGImages', basename + '.jpg'))
            # Save ground truth as a ndarray
            gt = imread(os.path.join(dataset_path, 'SegmentationClass', basename + '.png'))
            shape_set.append(gt.shape)
            gt = colors2labels(gt, self.cmap)
            im_set.append(im)
            gt_set.append(gt)

        print('Saving to ' + tfrecord_filename)
        self._export(im_set, gt_set, shape_set, os.path.join(tfrecords_path, tfrecord_filename))

    def _export(self, im_set, gt_set, shape_set, filename):
        def _int64_feature(value):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        with tf.io.TFRecordWriter(filename) as writer:
            for im, gt, shape in list(zip(im_set, gt_set, shape_set)):
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'height': _int64_feature(shape[0]),
                            'width': _int64_feature(shape[1]),
                            'depth': _int64_feature(shape[2]),
                            'image_raw': _bytes_feature(im),
                            'label_raw': _bytes_feature(gt.tostring())
                        }))
                writer.write(example.SerializeToString())

    def parse_record(self, record_serialized):
        """
        Parses a sample proto containing a training or validation example of an image. The output of the
        pascal_voc_dataset.py image preprocessing script is a dataset containing serialized sample protocol
        buffers. Each sample proto contains the following fields (values are included as examples):
            height: 281
            width: 500
            channels: 3
            format: 'JPEG'
            filename: '2007_000032'
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

        # Decode the raw data
        im = tf.image.decode_png(parsed['image_raw'])
        gt = tf.io.decode_raw(parsed['label_raw'], tf.uint8)
        gt = tf.reshape(gt, [tf.cast(parsed['height'][0], tf.int64), tf.cast(parsed['width'][0], tf.int64)])

        return im, gt, (parsed['height'][0], parsed['width'][0], parsed['depth'][0])

    def transform_record(self, im, gt):
        """Randomly transforms the record according to the data augmentation params."""
        return random_transform(im, gt, self.image_shape, **self.augmentation_params)

    def pad_record(self, im, gt, shape=None):
        """Pads the record to the size expected by the model. Used by PASCAL VOC."""
        im_padded = pad(im, self.image_shape, center=True)
        gt_padded = pad(gt, self.image_shape, center=True, cval=self.n_classes)
        if shape is not None:
            return im_padded, gt_padded, shape
        else:
            return im_padded, gt_padded

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
    """
    Export the PASCAL VOC segmentation dataset in 2 ways:
        1. Converts ground truth segmentation classes to sparse labels.
        2. Export the dataset to TFRecords, one for the training set and another one for the validation set.
    """
    dataset = PascalVOC2012Dataset(augmentation_params=None)
    train_basenames = dataset.get_basenames('train', FLAGS.data_dir)
    print('Found', len(train_basenames), 'training samples')

    val_basenames = dataset.get_basenames('val', FLAGS.data_dir)
    print('Found', len(val_basenames), 'validation samples')

    # Encode and save sparse ground truth segmentation image labels
    print('Exporting training set sparse labels...')
    dataset.export_sparse_encoding('train', FLAGS.data_dir)
    print('Exporting validation set sparse labels...')
    dataset.export_sparse_encoding('val', FLAGS.data_dir)

    # Export train and validation datasets to TFRecords
    dataset.export_tfrecord('train', FLAGS.data_dir, 'segmentation_train.tfrecords')
    dataset.export_tfrecord('val', FLAGS.data_dir, 'segmentation_val.tfrecords')
    print('Finished exporting')


if __name__ == '__main__':
    FLAGS, unparsed = parser.parse_known_args()
    tf.compat.v1.app.run(argv=[sys.argv[0]] + unparsed)

