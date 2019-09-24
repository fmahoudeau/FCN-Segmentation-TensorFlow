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

from scipy.io import loadmat
import os, os.path
import numpy as np
#from scipy.misc import imsave
from tqdm import *
import argparse
from distutils.dir_util import copy_tree
import random
import sys
import tensorflow as tf

from pascal_voc_dataset import PascalVOC2012Dataset

random.seed(1337)

"""
It creates a dataset composed of PASCAL VOC 2012 augmented with the work from:
Hariharan et al. Semantic contours from inverse detectors. ICCV 2011.
"""

parser = argparse.ArgumentParser()
parser.add_argument('--contours_dir',
                    default='/tmp/pascal_plus_data/benchmark_RELEASE/dataset/')
parser.add_argument('--voc_dir', default='/tmp/pascal_voc_data/VOCdevkit/VOC2012/')
parser.add_argument('--vocplus_dir', default='/tmp/pascal_plus_data/prepared',
                    help='This folder will be created')
parser.add_argument('--val_split', default=0., type=float, help='Percentage of samples to use for validation')
parser.add_argument('--force_gen', dest='force_gen', action='store_true')
parser.add_argument('--nocopy', dest='copy', action='store_false')
parser.set_defaults(force_gen=False)
parser.set_defaults(copy=True)


class PascalPlusDataset(PascalVOC2012Dataset):
    """Dataset class for PASCAL Plus, ie. PASCAL VOC 2012 augmented
       with the annotations from Hariharan et al."""

    def __init__(self, augmentation_params):
        super().__init__(augmentation_params)
        # Override n_images parameter values
        self.n_images = {
            'train': 10582,
            'val': 0,
            'test': 1449
        }

    def prepare(self, voc_dir, contours_dir, vocplus_dir, val_split=0., force_gen=False, copy=True):
        """
        Credits: this function is based on `pascalplus_gen.py` located on github.com/imatge-upc/rsis

        :param voc_dir: The source directory of the PASCAL VOC 2012 dataset.
        :param contours_dir: The source directory of the additional annotations.
        :param vocplus_dir: The destination directory for the augmented dataset.
        :param val_split: The proportion of images for validation. Defaults to 0,
            ie. merges train and val in one large training dataset. The test set
            remains unchanged.
        :param force_gen: Defaults to False.
        :param copy: Defaults to True.

        :return: None
        """
        def make_dir(dir):
            if not os.path.exists(dir):
                os.mkdir(dir)

        def read_file(filedir):
            with open(filedir, 'r') as f:
                lines = f.readlines()
            return lines

        def write_file(filedir, itemlist):
            with open(filedir, 'w') as f:
                for item in itemlist:
                    f.write(item)

        def pascal_palette():
            # RGB to int conversion
            palette = {(0, 0, 0): 0,
                       (128, 0, 0): 1,
                       (0, 128, 0): 2,
                       (128, 128, 0): 3,
                       (0, 0, 128): 4,
                       (128, 0, 128): 5,
                       (0, 128, 128): 6,
                       (128, 128, 128): 7,
                       (64, 0, 0): 8,
                       (192, 0, 0): 9,
                       (64, 128, 0): 10,
                       (192, 128, 0): 11,
                       (64, 0, 128): 12,
                       (192, 0, 128): 13,
                       (64, 128, 128): 14,
                       (192, 128, 128): 15,
                       (0, 64, 0): 16,
                       (128, 64, 0): 17,
                       (0, 192, 0): 18,
                       (128, 192, 0): 19,
                       (0, 64, 128): 20,
                       (224, 224, 192): 255}
            return palette

        palette = pascal_palette()
        id_to_rgb = {v: k for k, v in palette.items()}

        # create directories for new dataset
        make_dir(vocplus_dir)
        make_dir(os.path.join(vocplus_dir, 'SegmentationClass'))
        make_dir(os.path.join(vocplus_dir, 'SegmentationObject'))
        make_dir(os.path.join(vocplus_dir, 'ImageSets'))
        make_dir(os.path.join(vocplus_dir, 'JPEGImages'))
        make_dir(os.path.join(vocplus_dir, 'ImageSets', 'Segmentation'))

        # train and val splits from augmented dataset will both be used for training
        for split in ['train', 'val']:
            print("Processing %s set:" % split)
            names = read_file(os.path.join(contours_dir, split+'.txt'))
            print("Found %d images to process." %(len(names)))
            for n, name in tqdm(enumerate(names)):
                # extract segmentation info from mat files
                seg_png = os.path.join(vocplus_dir, 'SegmentationClass', name.rstrip()+'.png')
                obj_png = os.path.join(vocplus_dir, 'SegmentationObject', name.rstrip()+'.png')
                if not os.path.isfile(seg_png) or not os.path.isfile(obj_png) or force_gen:
                    m = loadmat(os.path.join(contours_dir, 'inst', name.rstrip()+'.mat'))['GTinst'][0][0]
                    seg_object = m[0]
                    classes = m[2]
                    sem_seg = np.zeros((seg_object.shape[0], seg_object.shape[1], 3), dtype=np.uint8)
                    ins_seg = np.zeros((seg_object.shape[0], seg_object.shape[1], 3), dtype=np.uint8)
                    for i in np.unique(seg_object):
                        if i == 0:
                            continue
                        class_ins = classes[i-1][0]
                        # encode class with corresponding RGB triplet
                        sem_seg[seg_object == i, 0] = id_to_rgb[class_ins][0]
                        sem_seg[seg_object == i, 1] = id_to_rgb[class_ins][1]
                        sem_seg[seg_object == i, 2] = id_to_rgb[class_ins][2]

                        # use i as class id (it will be unique) to color code instance masks
                        ins_seg[seg_object == i, 0] = id_to_rgb[i][0]
                        ins_seg[seg_object == i, 1] = id_to_rgb[i][1]
                        ins_seg[seg_object == i, 2] = id_to_rgb[i][2]
                        if i == 20:
                            break

                    imsave(seg_png, sem_seg)
                    imsave(obj_png, ins_seg)
                else:
                    print("File %d already exists ! Skipping..."%(n))

        print("Merging index lists and creating trainval split...")
        voc_train = read_file(os.path.join(voc_dir, 'ImageSets', 'Segmentation', 'train.txt'))
        print("VOC 2012 - train: Found %d samples" % (len(voc_train)))
        contours_train = read_file(os.path.join(contours_dir, 'train.txt'))
        print("Contours - train: Found %d samples" % (len(contours_train)))
        contours_val = read_file(os.path.join(contours_dir, 'val.txt'))
        print("Contours - val: Found %d samples" % (len(contours_val)))

        # the validation set of pascal voc will be used for testing
        test_samples = read_file(os.path.join(voc_dir, 'ImageSets', 'Segmentation', 'val.txt'))
        print("Val set from VOC will be used for testing.")

        samples = []
        samples.extend(voc_train)
        # Make sure we don't train with samples in the split we use for testing !!
        for sample in contours_train:
            if sample not in test_samples:
                samples.append(sample)
        for sample in contours_val:
            if sample not in test_samples:
                samples.append(sample)

        samples = list(set(samples))
        print("Total samples for training: ", len(samples))
        print("Note that images of VOC are part of the Contours dataset (duplicates are only used once)")
        random.shuffle(samples)

        sep = int(len(samples)*(1-val_split))
        train_samples = samples[:sep]
        val_samples = samples[sep:]

        print("The percentage of samples used for val was set to be %.2f" % val_split)
        print("Training samples:", len(train_samples))
        print("Validation samples:", len(val_samples))

        write_file(os.path.join(vocplus_dir, 'ImageSets', 'Segmentation', 'train.txt'), train_samples)
        write_file(os.path.join(vocplus_dir, 'ImageSets', 'Segmentation', 'val.txt'), val_samples)
        write_file(os.path.join(vocplus_dir, 'ImageSets', 'Segmentation', 'test.txt'), test_samples)

        if copy:
            print("Copying images from Contours dataset...")
            copy_tree(os.path.join(contours_dir, 'img'),
                      os.path.join(vocplus_dir, 'JPEGImages'))

            print ("Copying files from Pascal VOC to new dataset directory...")
            copy_tree(os.path.join(voc_dir, 'SegmentationClass'),
                      os.path.join(vocplus_dir, 'SegmentationClass'))
            copy_tree(os.path.join(voc_dir, 'SegmentationObject'),
                      os.path.join(vocplus_dir, 'SegmentationObject'))
            copy_tree(os.path.join(voc_dir, 'JPEGImages'),
                      os.path.join(vocplus_dir, 'JPEGImages'))

        print("All done.")


def main(_):
    """
    Export the PASCAL Plus segmentation dataset in 2 ways:
        1. Converts ground truth segmentation classes to sparse labels
        2. Export the dataset to TFRecords, one for the training set and
        another one for the validation set
    """
    dataset = PascalPlusDataset(augmentation_params=None)
    dataset.prepare(FLAGS.voc_dir, FLAGS.contours_dir, FLAGS.vocplus_dir,
                    FLAGS.val_split, FLAGS.force_gen, FLAGS.copy)

    print(FLAGS.vocplus_dir)
    train_basenames = dataset.get_basenames('train', FLAGS.vocplus_dir)
    print('Found', len(train_basenames), 'training samples')

    val_basenames = dataset.get_basenames('test', FLAGS.vocplus_dir)
    print('Found', len(val_basenames), 'test samples')

    # Encode and save sparse ground truth segmentation image labels
    print('Exporting training set sparse labels...')
    dataset.export_sparse_encoding('train', FLAGS.vocplus_dir)
    print('Exporting testing set sparse labels...')
    dataset.export_sparse_encoding('test', FLAGS.vocplus_dir)

    # Export train and validation datasets to TFRecords
    dataset.export_tfrecord('train', FLAGS.vocplus_dir, 'segmentation_train.tfrecords')
    # Note that in accordance with the other datasets in this project,
    # the filename ends with 'val', whereas the set name is 'test'
    dataset.export_tfrecord('test', FLAGS.vocplus_dir, 'segmentation_val.tfrecords')
    print('Finished exporting')


if __name__ == '__main__':
    FLAGS, unparsed = parser.parse_known_args()
    tf.compat.v1.app.run(argv=[sys.argv[0]] + unparsed)
