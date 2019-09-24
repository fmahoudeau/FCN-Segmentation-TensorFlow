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

import sys
import argparse
import os.path
import itertools
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.python.framework import ops

from image_utils import *
from pascal_voc_dataset import PascalVOC2012Dataset
from kitty_road_dataset import KittyRoadDataset
from cam_vid_dataset import CamVidDataset
from pascal_plus_dataset import PascalPlusDataset
import fcn_model


# Parse command line arguments
parser = argparse.ArgumentParser(description='Train FCN on PASCAL VOC 2012')
parser.add_argument('command',
                    metavar='<command>',
                    help="'train', 'evaluate', or 'predict'")
parser.add_argument('--dataset', required=True, type=str,
                    metavar='<dataset_name>',
                    help="The name of the dataset: 'kitty_road', 'cam_vid', "
                         "'pascal_voc_2012', or 'pascal_plus'.")
parser.add_argument('--fcn_version', required=True, type=str,
                    metavar='<FCN32|FCN16|FCN8>',
                    help="The FCN version to train: 'FCN8', 'FCN16' or 'FCN32'.")
parser.add_argument('--model_name', required=True, type=str,
                    metavar='<model_name>',
                    help='The model name is used to save checkpoints and summary data.')
parser.add_argument('--saved_variables', type=str, default=None,
                    metavar='<path/to/variables>',
                    help="Filename of FCN pre-trained weights. Optional for training and "
                         "mandatory for evaluation.")

# Paths arguments
parser.add_argument('--data_dir', type=str, default='/tmp/pascal_voc_data/',
                    metavar='<path/to/dataset>',
                    help='Path to the dataset root directory.')
parser.add_argument('--vgg16_weights_path', type=str, default='/tmp/vgg16/',
                    metavar='<path/to/vgg16/weights>',
                    help='Directory where the VGG16 pre-trained weights are located.')
parser.add_argument('--save_dir', type=str, default='/tmp/saved_models/',
                    metavar='<path/to/save/checkpoints>',
                    help='Directory where to save model checkpoints and summary data.')

# Training arguments
parser.add_argument('--optimizer', metavar='<Adam|SGD>', type=str, default='Adam',
                    help='Optimizer name')
parser.add_argument('--n_epochs', metavar='<n_epochs>', type=int, default=10,
                    help='Number of training epochs')
parser.add_argument('--batch_size', metavar='<batch_size>', type=int, default=8,
                    help='Batch size')
parser.add_argument('--learning_rate', metavar='<learning_rate>', type=float, default=1e-5,
                    help='Optimizer learning rate')
parser.add_argument('--weight_decay', metavar='<weight_decay>', type=float, default=1e-6,
                    help='Optimizer weight decay')
parser.add_argument('--momentum', metavar='<momentum>', type=float, default=0.9,
                    help='Optimizer momentum. Used only with SGD optimizer.')


#################################################################################
#  Utilities
#################################################################################


def save_learning_curve(model_name, epoch, metrics, training_loss, training_metrics,
                        validation_loss=None, validation_metrics=None):
    """Saves the learning curve to disk"""

    colors = ['r', 'b', 'g', 'm']  # The first color in the list is for the loss. The others are for the metrics.

    # Save a plot of the loss and accuracy
    plt.clf()
    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax2 = ax1.twinx()
    ax1.plot(training_loss[1:], colors[0], label="Training loss")
    for i, metric in enumerate(metrics.keys()):
        ax2.plot(training_metrics[metric][1:], colors[i + 1], label='Training {}'.format(metric))
    if validation_loss is not None and validation_metrics is not None:
        ax1.plot(validation_loss[1:], '{}--'.format(colors[0]), label="Validation loss")
        for i, metric in enumerate(metrics.keys()):
            ax2.plot(validation_metrics[metric][1:], '{}--'.format(colors[i + 1]), label='Validation {}'.format(metric))
    plt.title('Learning curve - {}\n{} with lr={} and weight_decay={}'.format(model_name, FLAGS.optimizer,
                                                                              FLAGS.learning_rate, FLAGS.weight_decay))
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=colors[0])
    ax2.set_ylabel('Score')
    ax1.legend(loc="lower center")
    ax2.legend(loc="center")
    plt.savefig('{}_training_history_{}.png'.format(os.path.join(FLAGS.save_dir, model_name), epoch + 1));

    # Save loss and accuracy in a DataFrame
    df = pd.DataFrame()
    df['train_loss'] = training_loss
    for metric in list(metrics.keys()):
        df['train_' + metric] = training_metrics[metric]
    if validation_loss is not None and validation_metrics is not None:
        df['val_loss'] = validation_loss
        for metric in list(metrics.keys()):
            df['val_' + metric] = validation_metrics[metric]
    df.to_csv('{}_{}.csv'.format(os.path.join(FLAGS.save_dir, model_name), epoch + 1))


def clear_session():
    """
    Clears the current graph variables and operations, and closes
    the current session before opening a new one.

    :return: A newly created interactive session
    """
    ops.reset_default_graph()
    if tf.compat.v1.get_default_session() is not None:
        tf.compat.v1.get_default_session().close()
    sess = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(intra_op_parallelism_threads=4))
    sess.as_default()
    return sess


#################################################################################
#  Compile graph
#################################################################################


def learning_rate_with_exp_decay(batch_size, n_images, decay_epochs, decay_rate=0.95, staircase=False, base_lr=1e-5):
    """
    Creates a learning rate function with exponential decay.

    :param batch_size: The batch size.
    :param n_images: The number of images in the training set.
    :param decay_epochs: The number of epochs after which the learning rate is decayed by `decay_rate`.
    :param decay_rate: The amount of decay applied after every `decay_epochs`. Defaults to 0.95.
        Set to 1 for a fixed learning rate.
    :param staircase: See TensorFlow documentation. Defaults to False.
    :param base_lr: The starting learning rate. Defaults to 1e-5.

    :return: The learning rate function.
    """

    global_step = tf.Variable(0, name='global_step', trainable=False)
    n_batches = int(n_images / batch_size)

    if FLAGS.debug:
        if decay_rate > 0:
            print('Applying exponential decay every {} epoch(s)'.format(decay_epochs))
        else:
            print('Applying fixed learning rate of {}'.format(base_lr))

    def learning_rate_fn():
        if decay_rate > 0:
            lr = tf.compat.v1.train.exponential_decay(base_lr, global_step, n_batches * decay_epochs,
                                            decay_rate, staircase=staircase, name='exp_decay')
        else:
            lr = tf.constant(base_lr, name='constant_lr')
        return lr

    return learning_rate_fn


def compile_model(model, metrics, learning_rate_fn, ignore_label=True, **kwargs):
    """
    Adds the optimizer, loss and metrics to the graph, and initialises all variables.

    :param model: The FCN model. Its weights must have been restored.
    :param metrics: The list of training and validation metrics to evaluate
        after each epoch. Only the following metrics are supported: `acc`
        for pixel accuracy, `mean_acc` for mean class accuracy, and `mean_iou`
        for mean intersection of union.
    :param learning_rate_fn: A function providing the learning rate for each
        training step.
    :param ignore_label: If the dataset contains a void/ignore label, the
        label in question is assumed to be equal to the number of classes.
        This label is used to create an ignore mask, that disables
        back-propagation of the loss for the ignored pixels. Defaults to True
        since all datasets in this project have an ignore label.
    :param kwargs: additional training parameters such as `momentum` or `decay`.
    :return: None
    """
    # Reshape 4D tensor (batch, row, column, channel) to 2D, each row represents a pixel, each column a class
    logits = tf.reshape(model.outputs, (-1, model.n_classes+1), name="logits")
    # Vector of ground truth labels
    labels = tf.stop_gradient(tf.reshape(model.labels, (-1,)))

    if FLAGS.debug:
        print('logits = {}'.format(str(logits.shape)))
        print('labels = {}'.format(str(model.labels.shape)))

    # Mask pixels to be ignored for datasets with a void class.
    # The void class is assumed to be last label in the color map
    if ignore_label:
        # Pixel mask vector, 1=keep, 0=ignore. Vector dim = batch_size * num_pixels
        ignore_pixels = tf.stop_gradient(tf.subtract(tf.ones((tf.shape(input=logits)[0],)),
                                                     tf.cast(tf.reshape(tf.equal(model.labels,
                                                                                 tf.constant(model.n_classes,
                                                                                             dtype=tf.float32)),
                                                                        [-1]),
                                                             dtype=tf.float32)))

        loss = tf.compat.v1.losses.compute_weighted_loss(
            weights=ignore_pixels,
            losses=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(labels, tf.int32),
                                                                  logits=logits, name="cross_entropy"))
    # Calculate mean distance from actual labels using cross entropy
    else:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(labels, tf.int32),
                                                                       logits=logits)
        loss = tf.reduce_mean(input_tensor=cross_entropy, name="cross_entropy")

    learning_rate = learning_rate_fn()

    # Create a tensor named learning_rate for logging purposes
    tf.identity(learning_rate, name='learning_rate')

    # Create optimizer
    if FLAGS.optimizer.lower() == 'adam':
        if 'weight_decay' in kwargs.keys():
            if kwargs['weight_decay'] > 0:
                opt = tf.contrib.opt.AdamWOptimizer(kwargs['weight_decay'],
                                                    learning_rate=learning_rate)
                if FLAGS.debug:
                    print('Adam optimizer with weight decay = {}'.format(kwargs['weight_decay']))
            else:
                opt = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
                if FLAGS.debug:
                    print('Adam optimizer w/o weight decay')
        else:
            opt = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
            if FLAGS.debug:
                print('Adam optimizer w/o weight decay')
    elif FLAGS.optimizer.lower() == 'sgd':
        momentum = kwargs['momentum'] if 'momentum' in kwargs.keys() else 0.0
        decay = 0.0  # For future use
        nesterov = kwargs['nesterov'] if 'nesterov' in kwargs.keys() else False

        if 'weight_decay' in kwargs.keys():
            if kwargs['weight_decay'] > 0:
                opt = tf.contrib.opt.MomentumWOptimizer(kwargs['weight_decay'], learning_rate=learning_rate,
                                                        momentum=momentum, use_nesterov=nesterov)
                if FLAGS.debug:
                    print('SGD optimizer: weight_decay={}, momentum={}, nestorov={}'.format(
                        kwargs['weight_decay'], momentum, nesterov))
            else:
                opt = tf.compat.v1.train.MomentumOptimizer(learning_rate=learning_rate,
                                                 momentum=momentum,
                                                 use_nesterov=nesterov)
                if FLAGS.debug:
                    print('SGD optimizer: momentum={}, nestorov={}'.format(momentum, nesterov))
        else:
            opt = tf.compat.v1.train.MomentumOptimizer(learning_rate=learning_rate,
                                             momentum=momentum,
                                             use_nesterov=nesterov)
            if FLAGS.debug:
                print('SGD optimizer: momentum={}, nestorov={}'.format(momentum, nesterov))
    else:
        raise ValueError('{} is an unrecognized optimizer'.format(FLAGS.optimizer))

    """
    # Doubling learning rates for biases was not used in the final solution 
    if FLAGS.double_bias_lr:
        grads_and_vars = opt.compute_gradients(loss)
        grads_and_vars_mult = []
        for grad, var in grads_and_vars:
            if 'bias' in var.op.name.lower():
                grad *= 2
            grads_and_vars_mult.append((grad, var))
        train_op = opt.apply_gradients(grads_and_vars_mult, global_step=global_step, name="train_op")
    else:"""
    train_op = opt.minimize(loss, global_step=tf.compat.v1.train.get_global_step(), name="train_op")

    # Create metrics
    tf_metrics = {}
    predictions = tf.argmax(input=logits, axis=1, name="predictions")  # The vector of predicted labels
    if 'acc' in metrics:  # Pixel accuracy
        tf_metrics['acc'] = tf.compat.v1.metrics.accuracy(labels, predictions,
                                                weights=ignore_pixels if ignore_label else None, name='acc')
    if 'mean_acc' in metrics:  # Mean class pixel accuracy
        tf_metrics['mean_acc'] = tf.compat.v1.metrics.mean_per_class_accuracy(labels, predictions, model.n_classes+1,
                                                                    weights=ignore_pixels if ignore_label else None,
                                                                    name='mean_acc')
    if 'mean_iou' in metrics:  # Mean Intersection-over-Union (mIoU)
        tf_metrics['mean_iou'] = tf.compat.v1.metrics.mean_iou(labels, predictions, model.n_classes+1,
                                                     weights=ignore_pixels if ignore_label else None,
                                                     name='mean_iou')

    sess = tf.compat.v1.get_default_session()
    sess.run(tf.compat.v1.global_variables_initializer())
    sess.run(tf.compat.v1.local_variables_initializer())
    model.logits = logits
    model.opt = opt
    model.train_op = train_op
    model.loss = loss
    model.metrics = tf_metrics
    print("Model build successful")


#################################################################################
#  Training
#################################################################################


def fit_model(model, epochs, batch_size, dataset_train, dataset_val, model_name, initial_epoch=0):
    """
    Trains a model for the specified number of epochs. A model checkpoint and
    training curve are saved every 25 epochs, and at the end of the training.
    The model will be evaluated after each epoch if a `dataset_val` is provided.

    :param model: The FCN model to train.
    :param epochs: The total number of training epochs to achieve.
    :param batch_size: The number of images per batch.
    :param dataset_train: The training dataset. An instance of TFRecordDataset.
    :param dataset_val: The validation dataset. An instance of TFRecordDataset.
        If set to None, the model is not evaluated during training.
    :param model_name: The name of your model, for example `fcn8_trial`.
        It is used to save weights, and the training curve CSV and plots to disk.
    :param initial_epoch: The number of training epochs already achieved
        prior to calling this method. The `epochs` parameter must be greater
        than `initial_epoch`, or else this method will return without
        performing additional training.
    :return: None
    """
    sess = tf.compat.v1.get_default_session()

    # TODO: Preserve these lists across multiple calls to this function.
    training_loss, validation_loss = [], []
    training_metrics, validation_metrics = {}, {}
    for metric in list(model.metrics.keys()):
        training_metrics[metric] = []
        validation_metrics[metric] = []

    # The minimum value of the loss to begin saving weights checkpoints.
    # I recommend 0.2 or lower for PASCAL VOC 2012 and PASCAL Plus,
    # and 0.05 or lower for KittyRoad & CamVid.
    # TODO: This deserves to be handled as an additional FLAGS parameter.
    best_loss = 0.05

    # Isolate the variables stored behind the scenes by the metrics operation
    metrics_vars = []
    for metric in list(model.metrics.keys()):
        metrics_vars.append(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.LOCAL_VARIABLES, scope=metric))
        training_metrics[metric] = []
        validation_metrics[metric] = []
    metrics_vars = list(itertools.chain.from_iterable(metrics_vars))  # Flatten list of lists

    # Define initializer to initialize/reset metrics variable(s)
    metrics_vars_initializer = tf.compat.v1.variables_initializer(var_list=metrics_vars)

    # A reinitializable iterator is defined by its structure. We could use the
    # `output_types` and `output_shapes` properties of either `training_dataset`
    # or `validation_dataset` here, because they are compatible.
    iterator = tf.compat.v1.data.Iterator.from_structure(tf.compat.v1.data.get_output_types(dataset_train),
                                                         tf.compat.v1.data.get_output_shapes(dataset_train))
    next_batch = iterator.get_next()

    train_init_op = iterator.make_initializer(dataset_train)
    val_init_op = iterator.make_initializer(dataset_val)

    print(datetime.now().isoformat())

    for epoch in range(initial_epoch, epochs):
        print("Epoch {}/{}, LR={}".format(epoch+1, epochs, sess.run(model.opt._lr)))
        training_loss.append(0)
        n_batches = 0

        # Initialize an iterator over the training dataset
        sess.run(train_init_op)

        # Train the model
        while True:
            try:
                im_batch, gt_batch = sess.run(next_batch)
                if len(im_batch) < batch_size:
                    continue
                res = sess.run({**{"loss": [model.loss, model.train_op]}, **model.metrics},
                               feed_dict={model.inputs: im_batch,
                                          model.labels: gt_batch,
                                          model.dropout_rate: FLAGS.dropout_rate})
                training_loss[-1] += res["loss"][0]
                n_batches += 1
            except tf.errors.OutOfRangeError:
                break

        # Update training loss and metrics
        training_loss[-1] /= n_batches
        current_loss = training_loss[-1]
        message = 'loss = {:.3f}'.format(training_loss[-1])
        for metric in list(model.metrics.keys()):
            # Remove the void/ignore class accuracy in the mean calculation because its value is 0
            if metric == 'mean_acc':
                val = np.mean(res[metric][1][:model.n_classes])
            # Remove the void/ignore class IoU in the mean calculation because its value is NaN
            elif metric == 'mean_iou':
                mat = res[metric][1][:model.n_classes, :model.n_classes]
                val = np.mean((np.diag(mat) / (mat.sum(axis=0) + mat.sum(axis=1) - np.diag(mat))))
            # No need to adjust other metrics
            else:
                val = res[metric][0]
            training_metrics[metric].append(val)
            message += ', {} = {:.3f}'.format(metric, val)

        # Reset the metrics variables
        sess.run(metrics_vars_initializer)

        # Evaluate the model if a validation dataset is provided
        if dataset_val is not None:
            validation_loss.append(0)
            n_batches = 0

            # Initialize an iterator over the validation dataset
            sess.run(val_init_op)

            while True:
                try:
                    im_batch, gt_batch = sess.run(next_batch)
                    if len(im_batch) < batch_size:  # TODO: This shouldn't be here. To be removed.
                        continue
                    res = sess.run({**{"loss": model.loss}, **model.metrics},
                                   feed_dict={model.inputs: im_batch,
                                              model.labels: gt_batch,
                                              model.dropout_rate: 0.0})
                    validation_loss[-1] += res["loss"]
                    n_batches += 1
                except tf.errors.OutOfRangeError:
                    break

            # Save validation metrics results
            validation_loss[-1] /= n_batches
            message += ', val_loss = {:.3f}'.format(validation_loss[-1])
            for metric in list(model.metrics.keys()):
                # Remove the void/ignore class accuracy in the mean calculation because its value is 0
                if metric == 'mean_acc':
                    val = np.mean(res[metric][1][:model.n_classes])
                # Remove the void/ignore class IoU in the mean calculation because its value is NaN
                elif metric == 'mean_iou':
                    mat = res[metric][1][:model.n_classes, :model.n_classes]
                    val = np.mean((np.diag(mat) / (mat.sum(axis=0) + mat.sum(axis=1) - np.diag(mat))))
                # No need to adjust other metrics
                else:
                    val = res[metric][0]
                validation_metrics[metric].append(val)
                message += ', val_{} = {:.3f}'.format(metric, val)

            # Reset the metrics variables
            sess.run(metrics_vars_initializer)

        print(message + '\n' + datetime.now().isoformat())

        model_saved = False
        # Checkpoint every 25 epochs and after the final epoch
        if ((epoch+1) % 25 == 0) | ((epoch+1) == epochs):
            print("Saving learning curve")
            if dataset_val is None:
                save_learning_curve(model_name, epoch, model.metrics,
                                    training_loss, training_metrics)
            else:
                save_learning_curve(model_name, epoch, model.metrics,
                                    training_loss, training_metrics,
                                    validation_loss, validation_metrics)
            print("Saving model checkpoint")
            model.save_variables(os.path.join(FLAGS.save_dir, model_name), global_step=epoch+1)
            model_saved = True

        # Save the best model
        if current_loss <= best_loss:
            if not model_saved:
                print("Saving model checkpoint")
                model.save_variables(os.path.join(FLAGS.save_dir, model_name), global_step=epoch+1)
            best_loss = current_loss
        print()


def oneoff_training(fcn_version, dataset_name, dataset_path, metrics, model_name, saved_variables=None):
    """
    Create a FCN model and perform one-off training. Using the `saved_variables`
    parameter, one can train a FCN16 model.

    :param fcn_version: The type of FCN, one of `FCN32`, `FCN16`, or `FCN8`.
    :param dataset_name: The name of the dataset to use for training, one of
        `kitty_road`, `cam_vid`, `pascal_voc_2012`, `pascal_plus`.
    :param dataset_path: The path to the dataset root directory.
    :param metrics: The list of training and validation metrics to evaluate
        after each epoch. Only the following metrics are supported: `acc`
        for pixel accuracy, `mean_acc` for mean class accuracy, and `mean_iou`
        for mean intersection of union.
    :param model_name: The name of your model, for example `fcn8_trial`.
        It is used to save weights, and the training curve CSV and plots to disk.
    :param saved_variables: Optional filename with pre-trained `FCN32` or
        `FCN16` weights to load. Do not use this parameter to indicate the path
         to VGG16 pre-trained weights.
    :return: The model, and a tuple with the training and validation datasets.
    """
    if fcn_version not in ('FCN32', 'FCN16', 'FCN8'):
        raise ValueError('{} is an invalid model'.format(fcn_version))

    print('One-off {} end-to-end training using {}'.format(fcn_version, dataset_name))

    sess = clear_session()
    if dataset_name == 'kitty_road':
        dataset = KittyRoadDataset(FLAGS.augmentation_params)
    elif dataset_name == 'cam_vid':
        dataset = CamVidDataset(FLAGS.augmentation_params)
    elif dataset_name == 'pascal_voc_2012':
        dataset = PascalVOC2012Dataset(FLAGS.augmentation_params)
    elif dataset_name == 'pascal_plus':
        dataset = PascalPlusDataset(FLAGS.augmentation_params)
    else:
        raise ValueError('{} is an invalid dataset'.format(dataset_name))

    dataset_train = dataset.load_dataset(is_training=True, data_dir=dataset_path, batch_size=FLAGS.batch_size)
    dataset_val = dataset.load_dataset(is_training=False, data_dir=dataset_path, batch_size=FLAGS.batch_size)

    # Build the model
    saved_variables = None if saved_variables is None else os.path.join(FLAGS.save_dir, saved_variables)
    model = fcn_model.Model(dataset.image_shape, dataset.n_classes, FLAGS.vgg16_weights_path)
    model(fcn_version, saved_variables=saved_variables)

    # No decay is applied, this is a constant learning rate
    learning_rate_fn = learning_rate_with_exp_decay(FLAGS.batch_size, dataset.n_images['train'], 10,
                                                    decay_rate=0, base_lr=FLAGS.learning_rate)
    compile_model(model, metrics, learning_rate_fn, weight_decay=FLAGS.weight_decay)

    # Train the model
    fit_model(model, FLAGS.n_epochs, FLAGS.batch_size, dataset_train, dataset_val, model_name)
    print("Total steps = {}".format(tf.compat.v1.train.global_step(sess, tf.compat.v1.train.get_global_step())))
    return model, (dataset_train, dataset_val)


def evaluate_model(fcn_version, dataset_name, dataset_path, metrics, saved_variables):
    """
    Create a FCN model and perform one-off training. Using the `saved_variables`
    parameter, one can train a FCN16 model.

    :param fcn_version: The type of FCN, one of `FCN32`, `FCN16`, or `FCN8`.
    :param dataset_name: The name of the dataset to use for training, one of
        `kitty_road`, `cam_vid`, `pascal_voc_2012`, `pascal_plus`.
    :param dataset_path: The path to the dataset root directory.
    :param metrics: The list metrics to calculate during model evaluation.
        Only the following metrics are supported: `acc` for pixel accuracy,
        `mean_acc` for mean class accuracy, and `mean_iou` for mean intersection of union.
    :param saved_variables: Path to the pre-trained FCN weights.
    :return: None
    """
    if fcn_version not in ('FCN32', 'FCN16', 'FCN8'):
        raise ValueError('{} is an invalid model'.format(fcn_version))

    print('{} evaluation on {}'.format(fcn_version, dataset_name))

    sess = clear_session()
    if dataset_name == 'kitty_road':
        dataset = KittyRoadDataset(FLAGS.augmentation_params)
    elif dataset_name == 'cam_vid':
        dataset = CamVidDataset(FLAGS.augmentation_params)
    elif dataset_name == 'pascal_voc_2012':
        dataset = PascalVOC2012Dataset(FLAGS.augmentation_params)
    elif dataset_name == 'pascal_plus':
        dataset = PascalPlusDataset(FLAGS.augmentation_params)
    else:
        raise ValueError('{} is an invalid dataset'.format(dataset_name))

    dataset_val = dataset.load_dataset(is_training=False, data_dir=dataset_path, batch_size=FLAGS.batch_size)

    # Build the model
    model = fcn_model.Model(dataset.image_shape, dataset.n_classes, FLAGS.vgg16_weights_path)
    model(fcn_version)

    # TODO: remove unnecessary graph operations
    # No exponential decay is applied, this is a constant learning rate
    learning_rate_fn = learning_rate_with_exp_decay(FLAGS.batch_size, dataset.n_images['train'], 10,
                                                    decay_rate=0, base_lr=FLAGS.learning_rate)
    compile_model(model, metrics, learning_rate_fn, weight_decay=FLAGS.weight_decay)
    model.load_variables(os.path.join(FLAGS.save_dir, saved_variables))

    # Evaluate the model
    validation_loss = 0
    n_batches = 0
    iterator = tf.compat.v1.data.Iterator.from_structure(dataset_val.output_types,
                                               dataset_val.output_shapes)
    next_batch = iterator.get_next()

    # Initialize an iterator over the validation dataset
    val_init_op = iterator.make_initializer(dataset_val)
    sess.run(val_init_op)

    # Train the model
    print('Now evaluating validation set...')
    while True:
        try:
            im_batch, gt_batch = sess.run(next_batch)
            if len(im_batch) < FLAGS.batch_size:
                continue
            res = sess.run({**{"loss": model.loss}, **model.metrics},
                           feed_dict={model.inputs: im_batch,
                                      model.labels: gt_batch,
                                      model.keep_prob: 1.0})
            validation_loss += res["loss"]
            n_batches += 1
        except tf.errors.OutOfRangeError:
            break

    # Save validation metrics results
    validation_loss /= n_batches
    message = 'val_loss = {:.3f}'.format(validation_loss)
    for metric in list(model.metrics.keys()):
        # Remove the void/ignore class accuracy in the mean calculation because its value is 0
        if metric == 'mean_acc':
            val = np.mean(res[metric][1][:model.n_classes])
        # Remove the void/ignore class IoU in the mean calculation because its value is NaN
        elif metric == 'mean_iou':
            mat = res[metric][1][:model.n_classes, :model.n_classes]
            val = np.mean((np.diag(mat) / (mat.sum(axis=0) + mat.sum(axis=1) - np.diag(mat))))
        # No need to adjust other metrics
        else:
            val = res[metric][0]
        message += ', val_{} = {:.3f}'.format(metric, val)
    print(message)


def predict_model(fcn_version, dataset_name, dataset_path, saved_variables):
    """
    Create a FCN model and perform one-off training. Using the `saved_variables`
    parameter, one can train a FCN16 model.

    :param fcn_version: The type of FCN, one of `FCN32`, `FCN16`, or `FCN8`.
    :param dataset_name: The name of the dataset to use for training, one of
        `kitty_road`, `cam_vid`, `pascal_voc_2012`, `pascal_plus`.
    :param dataset_path: The path to the dataset root directory.
    :param saved_variables: Path to the pre-trained FCN weights.
    :return: None
    """
    if fcn_version not in ('FCN32', 'FCN16', 'FCN8'):
        raise ValueError('{} is an invalid model'.format(fcn_version))

    print('{} evaluation on {}'.format(fcn_version, dataset_name))

    sess = clear_session()
    if dataset_name == 'kitty_road':
        dataset = KittyRoadDataset(FLAGS.augmentation_params)
        dataset_filepath = os.path.join(dataset_path, 'training/TFRecords/segmentation_test.tfrecords')
    elif dataset_name == 'cam_vid':
        dataset = CamVidDataset(FLAGS.augmentation_params)
        dataset_filepath = os.path.join(dataset_path, 'TFRecords/segmentation_val.tfrecords')
    elif dataset_name == 'pascal_voc_2012':
        dataset = PascalVOC2012Dataset(FLAGS.augmentation_params)
        dataset_filepath = os.path.join(dataset_path, 'TFRecords/segmentation_val.tfrecords')
    elif dataset_name == 'pascal_plus':
        dataset = PascalPlusDataset(FLAGS.augmentation_params)
        dataset_filepath = os.path.join(dataset_path, 'TFRecords/segmentation_val.tfrecords')
    else:
        raise ValueError('{} is an invalid dataset'.format(dataset_name))

    # Build the model
    model = fcn_model.Model(dataset.image_shape, dataset.n_classes, FLAGS.vgg16_weights_path)
    model(fcn_version)

    # TODO: remove unnecessary graph operations
    # No exponential decay is applied, this is a constant learning rate
    learning_rate_fn = learning_rate_with_exp_decay(FLAGS.batch_size, dataset.n_images['train'], 10,
                                                    decay_rate=0, base_lr=FLAGS.learning_rate)
    compile_model(model, FLAGS.metrics, learning_rate_fn, weight_decay=FLAGS.weight_decay)
    model.load_variables(os.path.join(FLAGS.save_dir, saved_variables))

    # And now predict masks
    dataset.predict_dataset(FLAGS.save_dir, dataset_filepath, model, FLAGS.batch_size)


#################################################################################
#  Main
#################################################################################


def main(_):
    print('Command:', FLAGS.command)
    print('Model:  ', FLAGS.fcn_version)
    print('Dataset:', FLAGS.dataset)

    if not os.path.exists(FLAGS.save_dir):
        os.makedirs(FLAGS.save_dir)

    # Adjust data augmentation params for other datasets than PASCAL.
    if FLAGS.dataset == 'kitty_road':
        FLAGS.augmentation_params['rotation_range'] = (-5, 5)
        FLAGS.augmentation_params['ignore_label'] = 2
    elif FLAGS.dataset == 'cam_vid':
        FLAGS.augmentation_params['rotation_range'] = (-5, 5)
        FLAGS.augmentation_params['ignore_label'] = 11

    # Train or evaluate
    if FLAGS.command == 'train':
        FLAGS.debug = True  # Prints some additional information, not really debug related
        oneoff_training(FLAGS.fcn_version, FLAGS.dataset, FLAGS.data_dir,
                        FLAGS.metrics, FLAGS.model_name, FLAGS.saved_variables)
    elif FLAGS.command == 'evaluate':
        FLAGS.debug = False  # Skip debug messages which are tailored for training
        evaluate_model(FLAGS.fcn_version, FLAGS.dataset, FLAGS.data_dir,
                       FLAGS.metrics, FLAGS.saved_variables)
    elif FLAGS.command == 'predict':
        FLAGS.debug = False  # Skip debug messages which are tailored for training
        predict_model(FLAGS.fcn_version, FLAGS.dataset, FLAGS.data_dir,
                      FLAGS.saved_variables)
    else:
        print('Unrecognized command: {}. Check help.'.format(FLAGS.command))


if __name__ == '__main__':
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    FLAGS.augmentation_params = {'saturation_range': (-20, 20), 'value_range': (-20, 20),
                                 'brightness_range': None, 'contrast_range': None, 'blur_params': None,
                                 'flip_lr': True, 'rotation_range': (-10, 10), 'shift_range': (32, 32),
                                 'zoom_range': (0.5, 2.0), 'ignore_label': 21}
    FLAGS.dropout_rate = 0.5
    FLAGS.metrics = ['acc', 'mean_acc', 'mean_iou']
    tf.compat.v1.app.run(argv=[sys.argv[0]] + unparsed)
