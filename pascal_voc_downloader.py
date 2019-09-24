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

"""Downloads and extracts the PASCAL VOC 2012 dataset."""

import argparse
import os
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf

DATA_URL = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'
parser = argparse.ArgumentParser()

parser.add_argument(
    '--data_dir', type=str, default='/tmp/pascal_voc_data/',
    help='Directory to download data and extract the tarball')


def download_dataset(data_url, data_path):
    """
    Download the tarball from the PASCAL Visual Object Classes website.

    :param data_url: the URL of the tarball
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


def extract_dataset(filepath, dest_dir):
    print('Extracting tarball...')
    tarfile.open(filepath, 'r').extractall(dest_dir)
    print('Finished extracting')


def main(_):
    """Download and extract the tarball from the PASCAL Visual Object Classes website."""

    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)

    filepath = os.path.join(FLAGS.data_dir, DATA_URL.split('/')[-1])

    if not os.path.exists(filepath):
        download_dataset(DATA_URL, FLAGS.data_dir)

    extract_dataset(filepath, FLAGS.data_dir)


if __name__ == '__main__':
    FLAGS, unparsed = parser.parse_known_args()
    tf.compat.v1.app.run(argv=[sys.argv[0]] + unparsed)
