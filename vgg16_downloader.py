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

"""Downloads and extracts the VGG16 model."""

import argparse
import os
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf

DATA_URL = 'https://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz'
parser = argparse.ArgumentParser()

parser.add_argument(
    '--data-dir', type=str, default='/tmp/vgg16/',
    help='Directory to download the zip archive')


def download_dataset(data_url, data_path):
    """
    Download the VGG16 weights archive from Davi Frossard's website.

    :param data_url: the URL of the zip archive
    :param data_path: the destination path
    :return: None
    """
    tarball_name = data_url.split('/')[-1]

    def _progress(count, block_size, total_size):
        sys.stdout.write('\rDownloading %s %.1f%%' % (
            tarball_name, 100.0*count*block_size/total_size))
        sys.stdout.flush()

    filepath, _ = urllib.request.urlretrieve(data_url, os.path.join(data_path, tarball_name), _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', tarball_name, statinfo.st_size, 'bytes.')


def main(_):
    """Download the VGG16 weights archive from Davi Frossard's website."""

    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)

    filepath = os.path.join(FLAGS.data_dir, DATA_URL.split('/')[-1])

    if not os.path.exists(filepath):
        download_dataset(DATA_URL, FLAGS.data_dir)


if __name__ == '__main__':
    FLAGS, unparsed = parser.parse_known_args()
    tf.compat.v1.app.run(argv=[sys.argv[0]] + unparsed)
