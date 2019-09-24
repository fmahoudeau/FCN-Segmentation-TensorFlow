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

from image_utils import (imread, imwrite, bytesread,
                         colors2labels, labels2colors,
                         pad, center_crop, apply_mask,
                         random_transform)


class Dataset(object):
    """Dataset base class."""

    def __init__(self, augmentation_params):
        self.augmentation_params = augmentation_params
        self.image_shape = (224, 224)
        self.n_classes = 2  # Excluding the ignore/void class
        self.class_labels = []
        self.n_images = None
        self.cmap = None

    def get_basenames(self, is_training, dataset_path):
        """Obtains a list of images base names that have been labelled for semantic segmentation."""
        pass

    def export_sparse_encoding(self, dataset_path):
        """Converts ground truth images to sparse labels and saves them to disk in PNG format."""
        pass

    def export_tfrecord(self, is_training, dataset_path, tfrecord_filename):
        """Exports a semantic image segmentation dataset to TFRecords."""
        pass

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
        """Parses a sample proto."""
        pass

    def transform_record(self, im, gt):
        """Randomly transforms the record according to the data augmentation params."""
        return random_transform(im, gt, self.image_shape, **self.augmentation_params)

    def pad_record(self, im, gt, shape=None):
        """Pads the record to the size expected by the model. Not used by all datasets."""
        im_padded = pad(im, self.image_shape, center=True)
        gt_padded = pad(gt, self.image_shape, center=True, cval=self.n_classes)
        if shape is not None:  # Return the original image shape when passed to this function.
            return im_padded, gt_padded, shape
        else:
            return im_padded, gt_padded

    def predict_dataset(self, save_path, dataset_filepath, model, batch_size):
        """Predicts semantic labels for all images of the speficied dataset and saves results to disk."""
        pass