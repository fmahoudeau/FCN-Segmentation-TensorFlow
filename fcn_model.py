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

import os.path
import numpy as np
import tensorflow as tf


def _bilinear_initializer(n_channels, kernel_size, cross_channel=False):
    """
    Creates a weight matrix that performs bilinear interpolation.

    :param n_channels: The number of channels, one per semantic class.
    :param kernel_size: The filter size, which is 2x the up-sampling factor,
        eg. a kernel/filter size of 4 up-samples 2x.
    :param cross_channel: Add contribution from all other channels to each channel.
        Defaults to False, meaning that each channel is up-sampled separately without
        contribution from the other channels.

    :return: A tf.constant_initializer with the weight initialized to bilinear interpolation.
    """
    # Make a 2D bilinear kernel suitable for up-sampling of the given (h, w) size.
    upscale_factor = (kernel_size+1)//2
    if kernel_size % 2 == 1:
        center = upscale_factor - 1
    else:
        center = upscale_factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    bilinear = (1-abs(og[0]-center)/upscale_factor) * (1-abs(og[1]-center)/upscale_factor)

    # The kernel filter needs to have shape [kernel_height, kernel_width, in_channels, num_filters]
    weights = np.zeros([kernel_size, kernel_size, n_channels, n_channels])
    if cross_channel:
        for i in range(n_channels):
            for j in range(n_channels):
                weights[:, :, i, j] = bilinear
    else:
        for i in range(n_channels):
            weights[:, :, i, i] = bilinear

    return tf.compat.v1.constant_initializer(value=weights)


class Model(object):
    """Base class for building the FCN model."""

    def __init__(self, image_shape, n_classes, vgg16_weights_path):
        """
        Creates a FCN model for semantic segmentation of images.

        :param image_shape: The images' input shape of the model, including padding.
            Both width and height must be divisible by 32.
        :param n_classes: The number of semantic classes, excluding the void/ignore class.
        :param vgg16_weights_path: The filename path to the pre-trained VGG16 numpy weights.
        """
        if len(image_shape) != 2:
            raise ValueError('Parameter image_shape must be 2D. Got {} dimensions.'.format(len(image_shape)))
        if not os.path.isfile(vgg16_weights_path):
            raise ValueError('VGG16 weights file not found. Check path {}'.format(vgg16_weights_path))

        self.image_shape = image_shape
        self.n_classes = n_classes
        self.vgg16_weights_path = vgg16_weights_path
        self.saver = None  # Cannot instantiate a saver before any variables are created

    def save_protobuf(self, model_path, tags):
        """
        Saves the model's meta graph and variables to the specified folder with the specified tags.
        All operations are saved including the optimizer's state for resuming training.

        :param model_path: The directory of the model graph and variables.
        :param tags: A list of model qualifiers used to describe the model, such as ['FCN8', 'VGG16'].

        :return: None
        """
        builder = tf.compat.v1.saved_model.Builder(model_path)
        builder.add_meta_graph_and_variables(tf.compat.v1.get_default_session(), tags)
        builder.save()

    def load_protobuf(self, model_path, tags):
        """
        Loads the model from a SavedModel as specified by tags.
        The tags must match with the ones used at time of saving the model.

        :param model_path: The directory of the model graph and variables.
        :param tags: The list of model qualifiers used to describe the model at saving time,
            such as ['FCN8', 'VGG16'].

        :return: The MetaGraphDef protocol buffer loaded in the current session.
        """
        if not os.path.exists(model_path):
            raise ValueError('Folder not found: {}'.format(model_path))

        model = tf.compat.v1.saved_model.loader.load(tf.compat.v1.get_default_session(), tags, model_path)
        return model

    def save_variables(self, variables_filename, global_step):
        """
        Saves the model's variables to the specified filename.

        :param variables_filename: The location of the filename to save.
        :param global_step: The number of training iterations. Appended to the filename.

        :return: None
        """
        if self.saver is None:
            self.saver = tf.compat.v1.train.Saver(max_to_keep=3)
        # Note that the meta graph could also be saved if we wanted to
        self.saver.save(tf.compat.v1.get_default_session(), variables_filename, global_step=global_step, write_meta_graph=False)

    def load_variables(self, variables_filename):
        """
        Loads the model's variables from the specified filename.
        The corresponding meta graph must be created before calling this method.

        :param variables_filename: The location of the filename to load.

        :return: None
        """
        if not os.path.exists(variables_filename + '.data-00000-of-00001'):
            raise ValueError('File not found: {}'.format(variables_filename + '.data-00000-of-00001'))

        # Do not use the same saver object for both loading and saving.
        # Any new variables added to the graph after loading won't be saved properly.
        # Therefore the class saver is reserved for saving actions only.
        saver = tf.compat.v1.train.Saver(max_to_keep=3)
        saver.restore(tf.compat.v1.get_default_session(), variables_filename)

    def __call__(self, model_name, saved_model=None, saved_variables=None):
        """
        :param model_name: The name of the model to create, one of FCN32, FCN16 or FCN8.
        :param saved_model: Optional 2-keys dictionary to restore a model variables and operations:
            1. 'model_path' is the path to the model's protobuf file.
            2. 'tags' is a list of model qualifiers.
            used to continue staged training, or for serving purposes.
        :param saved_variables: Optional filename path to restore pre-trained FCN weights.

        :return: The output layer of the chosen model.
        """
        if model_name not in ('FCN32', 'FCN16', 'FCN8'):
            raise ValueError('{} is an invalid model'.format(model_name))
        if (saved_model is not None) & (saved_variables is not None):
            raise ValueError('Only one of \'saved_model\' or \'saved_variables\' parameters can be different from None')

        # Create a FCN model, restoring VGG16 pre-trained weights or FCN32/FCN16
        # pre-trained weights in case of staged training of respectively FCN16/FCN8.
        if saved_model is None:
            print('Building {} model...'.format(model_name))
            self.inputs = tf.compat.v1.placeholder(tf.float32, [None, *self.image_shape, 3], name='inputs')
            self.labels = tf.compat.v1.placeholder(tf.float32, [None, *self.image_shape], name='labels')
            self.dropout_rate = tf.compat.v1.placeholder(tf.float32, shape=[], name='rate')
            self._fcn_base()
            if model_name == 'FCN32':
                self.outputs = self._fcn_32()
            elif model_name == 'FCN16':
                self._fcn_32()
                if saved_variables is not None:
                    print('Restoring FCN32 pre-trained weights...')
                    self.load_variables(saved_variables)
                self.outputs = self._fcn_16()
            else:
                self._fcn_32()
                self._fcn_16()
                if saved_variables is not None:
                    print('Restoring FCN16 pre-trained weights...')
                    self.load_variables(saved_variables)
                self.outputs = self._fcn_8()
        # Load a pre-trained model graph and its weights to resume training or for inference.
        else:
            print('Loading {} model...'.format(model_name))
            if model_name not in saved_model['tags']:  # Ensure the model being loaded is the one expected
                raise ValueError(
                    'Invalid model tags. Expected {}, but got {}.'.format(model_name, str(saved_model['tags'])))

            self.load_protobuf(**saved_model)
            # Retrieve key tensors from the graph.
            graph = tf.compat.v1.get_default_graph()
            self.inputs = graph.get_tensor_by_name('inputs:0')
            self.labels = graph.get_tensor_by_name('labels:0')
            self.dropout_rate = graph.get_tensor_by_name('rate:0')
            self.fcn32_out = graph.get_tensor_by_name('fcn32_out:0')
            self.logits = tf.compat.v1.get_default_graph().get_tensor_by_name('logits:0')
            if 'FCN32' in saved_model['tags']:
                self.outputs = self.fcn32_out
            elif 'FCN16' in saved_model['tags']:
                self.fcn16_out = graph.get_tensor_by_name('fcn16_out:0')
                self.outputs = self.fcn16_out
            elif 'FCN8' in saved_model['tags']:
                self.fcn16_out = graph.get_tensor_by_name('fcn16_out:0')
                self.fcn8_out = graph.get_tensor_by_name('fcn8_out:0')
                self.outputs = self.fcn8_out
            else:
                raise ValueError(
                    'Invalid model tags. Expected FCN32, FCN16, or FCN8 but got {}.'.format(
                        str(saved_model['tags'])))
        return self.outputs

    def _fcn_base(self):
        """
        Builds the base FCN layers using VGG16. This base has the following differences with VGG16:
            1. The last layer (ie. the classifier) of VGG16 is removed.
            2. The remaining fully connected layers of VGG16 (`fc6` and `fc7`) are convolutionized.
        All layers are initialised with VGG16 pre-trained weights.
        The RGB mean of the ImageNet training set is subtracted to the input images batch.

        :return: The last layer of the base network, ie. the convolutionized version of `fc7`.
        """
        weights = np.load(self.vgg16_weights_path)

        # Subtract the mean RGB value, computed on the VGG16 training set, from each pixel
        with tf.compat.v1.name_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32,
                               shape=[1, 1, 1, 3], name='img_mean')
            images = self.inputs - mean

        # Block 1
        with tf.compat.v1.name_scope('conv1_1') as scope:
            kernel = tf.Variable(weights['conv1_1_W'], name='weights', dtype=tf.float32, expected_shape=[3, 3, 3, 64])
            conv = tf.nn.conv2d(input=images, filters=kernel, strides=[1, 1, 1, 1], padding='SAME')
            bias = tf.Variable(weights['conv1_1_b'], name='biases', dtype=tf.float32, expected_shape=[64])
            out = tf.nn.bias_add(conv, bias)
            self.conv1_1 = tf.nn.relu(out, name=scope)

        with tf.compat.v1.name_scope('conv1_2') as scope:
            kernel = tf.Variable(weights['conv1_2_W'], name='weights', dtype=tf.float32, expected_shape=[3, 3, 64, 64])
            conv = tf.nn.conv2d(input=self.conv1_1, filters=kernel, strides=[1, 1, 1, 1], padding='SAME')
            bias = tf.Variable(weights['conv1_2_b'], name='biases', dtype=tf.float32, expected_shape=[64])
            out = tf.nn.bias_add(conv, bias)
            self.conv1_2 = tf.nn.relu(out, name=scope)

        self.pool1 = tf.nn.max_pool2d(input=self.conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                    padding='SAME', name='pool1')

        # Block 2
        with tf.compat.v1.name_scope('conv2_1') as scope:
            kernel = tf.Variable(weights['conv2_1_W'], name='weights', dtype=tf.float32, expected_shape=[3, 3, 64, 128])
            conv = tf.nn.conv2d(input=self.pool1, filters=kernel, strides=[1, 1, 1, 1], padding='SAME')
            bias = tf.Variable(weights['conv2_1_b'], name='biases', dtype=tf.float32, expected_shape=[128])
            out = tf.nn.bias_add(conv, bias)
            self.conv2_1 = tf.nn.relu(out, name=scope)

        with tf.compat.v1.name_scope('conv2_2') as scope:
            kernel = tf.Variable(weights['conv2_2_W'], name='weights', dtype=tf.float32,
                                 expected_shape=[3, 3, 128, 128])
            conv = tf.nn.conv2d(input=self.conv2_1, filters=kernel, strides=[1, 1, 1, 1], padding='SAME')
            bias = tf.Variable(weights['conv2_2_b'], name='biases', dtype=tf.float32, expected_shape=[128])
            out = tf.nn.bias_add(conv, bias)
            self.conv2_2 = tf.nn.relu(out, name=scope)

        self.pool2 = tf.nn.max_pool2d(input=self.conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                    padding='SAME', name='pool2')

        # Block 3
        with tf.compat.v1.name_scope('conv3_1') as scope:
            kernel = tf.Variable(weights['conv3_1_W'], name='weights', dtype=tf.float32,
                                 expected_shape=[3, 3, 128, 256])
            conv = tf.nn.conv2d(input=self.pool2, filters=kernel, strides=[1, 1, 1, 1], padding='SAME')
            bias = tf.Variable(weights['conv3_1_b'], name='biases', dtype=tf.float32, expected_shape=[256])
            out = tf.nn.bias_add(conv, bias)
            self.conv3_1 = tf.nn.relu(out, name=scope)

        with tf.compat.v1.name_scope('conv3_2') as scope:
            kernel = tf.Variable(weights['conv3_2_W'], name='weights', dtype=tf.float32,
                                 expected_shape=[3, 3, 256, 256])
            conv = tf.nn.conv2d(input=self.conv3_1, filters=kernel, strides=[1, 1, 1, 1], padding='SAME')
            bias = tf.Variable(weights['conv3_2_b'], name='biases', dtype=tf.float32, expected_shape=[256])
            out = tf.nn.bias_add(conv, bias)
            self.conv3_2 = tf.nn.relu(out, name=scope)

        with tf.compat.v1.name_scope('conv3_3') as scope:
            kernel = tf.Variable(weights['conv3_3_W'], name='weights', dtype=tf.float32,
                                 expected_shape=[3, 3, 256, 256])
            conv = tf.nn.conv2d(input=self.conv3_2, filters=kernel, strides=[1, 1, 1, 1], padding='SAME')
            bias = tf.Variable(weights['conv3_3_b'], name='biases', dtype=tf.float32, expected_shape=[256])
            out = tf.nn.bias_add(conv, bias)
            self.conv3_3 = tf.nn.relu(out, name=scope)

        self.pool3 = tf.nn.max_pool2d(input=self.conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                    padding='SAME', name='pool3')
        self.layer3_out = tf.identity(self.pool3, name='layer3_out')

        # Block 4
        with tf.compat.v1.name_scope('conv4_1') as scope:
            kernel = tf.Variable(weights['conv4_1_W'], name='weights', dtype=tf.float32,
                                 expected_shape=[3, 3, 256, 512])
            conv = tf.nn.conv2d(input=self.pool3, filters=kernel, strides=[1, 1, 1, 1], padding='SAME')
            bias = tf.Variable(weights['conv4_1_b'], name='biases', dtype=tf.float32, expected_shape=[512])
            out = tf.nn.bias_add(conv, bias)
            self.conv4_1 = tf.nn.relu(out, name=scope)

        with tf.compat.v1.name_scope('conv4_2') as scope:
            kernel = tf.Variable(weights['conv4_2_W'], name='weights', dtype=tf.float32,
                                 expected_shape=[3, 3, 512, 512])
            conv = tf.nn.conv2d(input=self.conv4_1, filters=kernel, strides=[1, 1, 1, 1], padding='SAME')
            bias = tf.Variable(weights['conv4_2_b'], name='biases', dtype=tf.float32, expected_shape=[512])
            out = tf.nn.bias_add(conv, bias)
            self.conv4_2 = tf.nn.relu(out, name=scope)

        with tf.compat.v1.name_scope('conv4_3') as scope:
            kernel = tf.Variable(weights['conv4_3_W'], name='weights', dtype=tf.float32,
                                 expected_shape=[3, 3, 512, 512])
            conv = tf.nn.conv2d(input=self.conv4_2, filters=kernel, strides=[1, 1, 1, 1], padding='SAME')
            bias = tf.Variable(weights['conv4_3_b'], name='biases', dtype=tf.float32, expected_shape=[512])
            out = tf.nn.bias_add(conv, bias)
            self.conv4_3 = tf.nn.relu(out, name=scope)

        self.pool4 = tf.nn.max_pool2d(input=self.conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                    padding='SAME', name='pool4')
        self.layer4_out = tf.identity(self.pool4, name='layer4_out')

        # Block 5
        with tf.compat.v1.name_scope('conv5_1') as scope:
            kernel = tf.Variable(weights['conv5_1_W'], name='weights', dtype=tf.float32,
                                 expected_shape=[3, 3, 512, 512])
            conv = tf.nn.conv2d(input=self.pool4, filters=kernel, strides=[1, 1, 1, 1], padding='SAME')
            bias = tf.Variable(weights['conv5_1_b'], name='biases', dtype=tf.float32, expected_shape=[512])
            out = tf.nn.bias_add(conv, bias)
            self.conv5_1 = tf.nn.relu(out, name=scope)

        with tf.compat.v1.name_scope('conv5_2') as scope:
            kernel = tf.Variable(weights['conv5_2_W'], name='weights', dtype=tf.float32,
                                 expected_shape=[3, 3, 512, 512])
            conv = tf.nn.conv2d(input=self.conv5_1, filters=kernel, strides=[1, 1, 1, 1], padding='SAME')
            bias = tf.Variable(weights['conv5_2_b'], name='biases', dtype=tf.float32, expected_shape=[512])
            out = tf.nn.bias_add(conv, bias)
            self.conv5_2 = tf.nn.relu(out, name=scope)

        with tf.compat.v1.name_scope('conv5_3') as scope:
            kernel = tf.Variable(weights['conv5_3_W'], name='weights', dtype=tf.float32,
                                 expected_shape=[3, 3, 512, 512])
            conv = tf.nn.conv2d(input=self.conv5_2, filters=kernel, strides=[1, 1, 1, 1], padding='SAME')
            bias = tf.Variable(weights['conv5_3_b'], name='biases', dtype=tf.float32, expected_shape=[512])
            out = tf.nn.bias_add(conv, bias)
            self.conv5_3 = tf.nn.relu(out, name=scope)

        self.pool5 = tf.nn.max_pool2d(input=self.conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                    padding='SAME', name='pool5')

        # Block 6 is a convolutionized version of fc6 with kernel shape (25088, 4096)
        with tf.compat.v1.name_scope('conv6') as scope:
            kernel = tf.Variable(weights['fc6_W'].reshape(7, 7, 512, 4096), name='weights',
                                 dtype=tf.float32, expected_shape=[7, 7, 512, 4096])
            conv = tf.nn.conv2d(input=self.pool5, filters=kernel, strides=[1, 1, 1, 1], padding='SAME')
            bias = tf.Variable(weights['fc6_b'], name='biases', dtype=tf.float32, expected_shape=[4096])
            out = tf.nn.bias_add(conv, bias)
            self.conv6 = tf.nn.relu(out, name=scope)

        self.drop1 = tf.nn.dropout(self.conv6, name='dropout1', rate=self.dropout_rate)

        # Block 7 is a convolutionized version of fc7 with kernel shape (4096, 4096)
        with tf.compat.v1.name_scope('conv7') as scope:
            kernel = tf.Variable(weights['fc7_W'].reshape(1, 1, 4096, 4096), name='weights',
                                 dtype=tf.float32, expected_shape=[1, 1, 4096, 4096])
            conv = tf.nn.conv2d(input=self.drop1, filters=kernel, strides=[1, 1, 1, 1], padding='SAME')
            bias = tf.Variable(weights['fc7_b'], name='biases', dtype=tf.float32, expected_shape=[4096])
            out = tf.nn.bias_add(conv, bias)
            self.conv7 = tf.nn.relu(out, name=scope)

        self.drop2 = tf.nn.dropout(self.conv7, name='dropout2', rate=self.dropout_rate)
        self.layer7_out = tf.identity(self.drop2, name='layer7_out')

    def _fcn_32(self):
        """
        Builds the FCN32 network specific layers on top of the base FCN layers.
        This function must be called after calling the `_fcn_base` function.
        It performs:
            1. Class prediction at stride 32 of the last layer of the FCN base.
            2. Bilinear interpolation back to the input image shape.

        :return: The output layer of FCN32.
        """
        # Apply 1x1 convolution to predict classes of layer 7 at stride 32
        self.conv7_classes = tf.compat.v1.layers.conv2d(self.layer7_out, filters=self.n_classes+1, kernel_size=1,
                                              kernel_initializer=tf.compat.v1.zeros_initializer(), name="conv7_classes")

        # 32x bilinear interpolation
        self.fcn32_out = tf.image.resize(self.conv7_classes, self.image_shape, name="fcn32_out", method=tf.image.ResizeMethod.BILINEAR)

        return self.fcn32_out

    def _fcn_16(self):
        """
        Builds the FCN16 network specific layers on top of the FCN32 layers.
        This function must be called after calling the `_fcn_base` and `_fcn_32` functions.
        It performs:
            1. 2x up-sampling of the ouptput layer of FCN32. The weights are trainable and
               initialized to bilinear interpolation.
            2. Class prediction of layer 4 of VGG16 at stride 16.
            3. Element-wise sum of the above tensors, thereby implementing the first skip connection.
            4. Bilinear interpolation back to the input image shape.

        :return: The output layer of FCN16.
        """
        # Apply 1x1 convolution to predict classes of layer 4 at stride 16
        self.pool4_classes = tf.compat.v1.layers.conv2d(self.layer4_out, filters=self.n_classes+1, kernel_size=1,
                                              kernel_initializer=tf.compat.v1.zeros_initializer(), name="pool4_classes")

        # Up-sample (2x) conv7 class predictions to match the size of layer 4
        self.fcn32_upsampled = tf.compat.v1.layers.conv2d_transpose(self.conv7_classes, filters=self.n_classes+1,
                                                          kernel_size=4, strides=2, padding='SAME', use_bias=False,
                                                          kernel_initializer=_bilinear_initializer(self.n_classes+1, 4),
                                                          name="fcn32_upsampled")

        # Add a skip connection between class predictions of layer 4 and up-sampled class predictions of layer 7
        self.skip_1 = tf.add(self.pool4_classes, self.fcn32_upsampled, name="skip_cnx_1")

        # 16x bilinear interpolation
        self.fcn16_out = tf.image.resize(self.skip_1, self.image_shape, name="fcn16_out", method=tf.image.ResizeMethod.BILINEAR)

        return self.fcn16_out

    def _fcn_8(self):
        """
        Builds the FCN8 network specific layers on top of the FCN16 layers.
        This function must be called after calling the `_fcn_base`, `_fcn_32` and `_fcn_16` functions.
        It performs:
            1. 2x up-sampling of the output layer of FCN16. The weights are trainable and
               initialized to bilinear interpolation.
            2. Class prediction of layer 3 of VGG16 at stride 8.
            3. Element-wise sum of the above tensors, thereby implementing the second skip connection.
            4. Bilinear interpolation back to the input image shape.

        :return: The output layer of FCN8.
        """
        # Apply 1x1 convolution to predict classes of layer 3 at stride 8
        self.pool3_classes = tf.compat.v1.layers.conv2d(self.layer3_out, filters=self.n_classes+1, kernel_size=1,
                                              kernel_initializer=tf.compat.v1.zeros_initializer(), name="pool3_classes")

        # Up-sample (2x) skip_1 class predictions to match the size of layer 3
        self.fcn16_upsampled = tf.compat.v1.layers.conv2d_transpose(self.skip_1, filters=self.n_classes+1,
                                                          kernel_size=4, strides=2, padding='SAME', use_bias=False,
                                                          kernel_initializer=_bilinear_initializer(self.n_classes+1, 4),
                                                          name="fcn16_upsampled")

        # Add a skip connection between class predictions of layer 4 and up-sampled class predictions of layer 7
        self.skip_2 = tf.add(self.pool3_classes, self.fcn16_upsampled, name="skip_cnx_2")

        # 8x bilinear interpolation
        self.fcn8_out = tf.image.resize(self.skip_2, self.image_shape, name="fcn8_out", method=tf.image.ResizeMethod.BILINEAR)

        return self.fcn8_out
