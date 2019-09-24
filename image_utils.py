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

import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
import PIL.Image


def listfiles(pathname):
    return [f for f in listdir(pathname) if isfile(join(pathname, f))]


def bytesread(filename):
    with open(filename, 'rb') as file:
        return file.read()


def imread(filename, target_shape=None, interpolation=cv2.INTER_AREA):
    """
    Loads an image from disk

    :param filename: the path to the image file to load
    :param target_shape: optional resizing to the specified shape
    :param interpolation: interpolation method. Defaults to cv2.INTER_AREA which is recommended for downsizing.
    :return: the loaded image in RGB format
    """
    im = cv2.imread(filename)
    if im is None:
        print('Error loading image. Check path:', filename)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # OpenCV loads images in BGR format
    if target_shape is not None:
        assert len(target_shape) == 2, 'Parameter target_shape must be 2-dimensional'
        im = cv2.resize(im, target_shape[::-1], interpolation=interpolation)
    return im


def imwrite(filename, im):
    """
    Saves an image to disk

    :param filename: the path to the image file to save
    :return: None
    """
    bgr = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, bgr)


def bgr2rgb(im):
    """
    Converts a BGR image to RGB.

    :param im: the BGR image to transform
    :return: the image in RGB mode
    """
    return np.asarray(im[2], im[1], im[0])


def rgb2bgr(im):
    """
    Converts a RBG image to BGR.

    :param im: the RGB image to transform
    :return: the image in BGR mode
    """
    return np.asarray(im[2], im[1], im[0])


def imhist(im):
    """
    Returns a color histogram of the given image.

    :param im: a numpy array with shape (rows, columns, 3)
    :return a list of colors and a list of pixel counters for each color
    """
    return np.unique(im.reshape(-1, im.shape[2]), axis=0, return_counts=True)


def subtract_mean(im):
    """
    Subtracts the RBG mean to each pixel. The image must be in RGB format.
    Returns a copy, the input image remains unchanged.

    :param im: the image to transform
    :return: the image with the mean removed
    """
    x = im.astype(np.float32)
    x[:, :] -= np.asarray([103.939, 116.779, 123.68]).astype(np.float32)
    return x


def pad(im, target_shape, center=False, cval=0):
    """
    Pads an image to the specified shape. The image must be smaller than the target shape.
    Returns a copy, the input image remains unchanged.

    :param im: the image to pad
    :param target_shape: the shape of the image after padding
    :param center: center the image or append rows and columns to the image
    :param cval: constant value for the padded pixels
    :return:
    """
    h_pad, w_pad = np.asarray(target_shape) - im.shape[:2]
    assert h_pad >= 0, 'Height padding must be non-negative'
    assert w_pad >= 0, 'Width padding must be non-negative'

    if center:
        padding = ((h_pad//2, h_pad-h_pad//2), (w_pad//2, w_pad-w_pad//2))
        if len(im.shape) == 3:
            padding += ((0, 0),)
        im_padded = np.pad(im, padding, mode='constant', constant_values=cval)
    else:
        padding = ((0, h_pad), (0, w_pad))
        if len(im.shape) == 3:
            padding += ((0, 0),)
        im_padded = np.pad(im, padding, mode='constant', constant_values=cval)
    return im_padded


def flip_axis(im, axis):
    """
    Flips a numpy array along the given axis.
    Returns a copy, the input image remains unchanged.

    :param im: numpy array
    :param axis: the axis along which to flip the data
    :return: the flipped array
    """
    x = np.asarray(im).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


def random_flip_axis(im, gt, axis):
    """
    Randomly flips the input image and its segmentation labels along the given axis.

    :param im: the image to transform
    :param gt: the ground truth pixels' labels (ie. semantic class) in sparse format
    :param axis: the axis along which to flip the data
    :return: the original input or a flipped copy
    """
    if np.random.random() < 0.5:
        return flip_axis(im, axis), flip_axis(gt, axis)
    return im, gt


def random_blur(im, ksize_max, sigma_max):
    """
    Randomly blurs an image using Gaussian filtering.

    :param im: the image to blur
    :param ksize_max: a tuple with the maximum kernel size along the X and Y axes
    :param sigma_max: a tuple with the maximum kernel sigma along the X and Y axes
    :return: the blurred image
    """
    # The kernel size must be odd
    ksize = [np.random.randint(ksize_max[0]), np.random.randint(ksize_max[1])]
    if (ksize[0] % 2) != 1:
        ksize[0] += 1
    if (ksize[1] % 2) != 1:
        ksize[1] += 1

    sigmaX = sigma_max[0]*np.random.random()
    sigmaY = sigma_max[1]*np.random.random()
    im_blur = cv2.GaussianBlur(im, tuple(ksize), sigmaX=sigmaX, sigmaY=sigmaY)
    return im_blur


def zoom(im, scale, interpolation):
    """
    Zooms an input image to the specified zoom factor.

    :param im: the image to zoom
    :param scale: the zoom factor, 1.0 means no zoom
    :param interpolation: the interpolation method:
     - cv2.INTER_LINEAR for an image,
     - cv2.INTER_NEAREST for its ground truth pixels' labels
    :return: the resized image
    """
    return cv2.resize(im, dsize=(0, 0), fx=scale, fy=scale, interpolation=interpolation)


def random_zoom(im, gt, zoom_range):
    """
    Randomly zooms in/out of an image and its ground truth segmentation labels.

    :param im: the image
    :param gt: the segmentation labels
    :param zoom_range: a tuple made of the min & max zoom factors, for example (0.8, 1.2)
    :return: the resized image and ground truth labels
    """
    scale = np.random.uniform(*zoom_range)
    x = zoom(im, scale, cv2.INTER_LINEAR)
    y = zoom(gt, scale, cv2.INTER_NEAREST)
    return x, y


def adjust_saturation_and_value(im, saturation=0, value=0):
    """
    Adjusts the saturation and value of the input image by the specified integer amounts.
    Pixels are clipped to maintain their HSV values in [0, 255].
    Returns a copy, the input image remains unchanged.

    :param im: the image to transform
    :param saturation: the absolute 'saturation' amount to apply
    :param value: the absolute 'value' amount to apply
    :return: the transformed image
    """
    if (saturation == 0) & (value == 0):
        return im

    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    if saturation != 0:
        if saturation > 0:
            s = np.where(s <= 255-saturation, s+saturation, 255).astype('uint8')
        else:
            s = np.where(s <= -saturation, 0, s+saturation).astype('uint8')

    if value != 0:
        if value > 0:
            v = np.where(v <= 255-value, v+value, 255).astype('uint8')
        else:
            v = np.where(v <= -value, 0, v+value).astype('uint8')

    hsv = cv2.merge((h, s, v))
    hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return hsv


def adjust_brightness_and_contrast(im, brightness=0, contrast=0):
    """
    Adjusts the brightness and contrast of the input image by the specified integer amounts.
    Pixels are clipped to maintain their RGB values in [0, 255].
    Returns a copy, the input image remains unchanged.

    :param im: the image to transform
    :param brightness: the absolute 'brightness' amount to apply
    :param contrast: the absolute 'contrast' amount to apply
    :return: the transformed image
    """
    if (brightness == 0) & (contrast == 0):
        return im

    buf = im.copy()

    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255+brightness
        alpha_b = (highlight-shadow)/255
        gamma_b = shadow

        buf = cv2.addWeighted(buf, alpha_b, buf, 0, gamma_b)

    if contrast != 0:
        f = 131*(contrast+127) / (127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf


def center_crop(im, target_shape):
    h, w = target_shape
    y, x = im.shape[:2]
    start_y = max(0, y//2-(h//2))
    start_x = max(0, x//2-(w//2))
    return im[start_y:start_y+h, start_x:start_x+w]


def random_crop(im, gt_im, target_shape):
    h, w = target_shape
    h_crop, w_crop = im.shape[:2] - np.asarray(target_shape)
    start_y = np.random.randint(0, h_crop)
    start_x = np.random.randint(0, w_crop)
    return im[start_y:start_y+h, start_x:start_x+w], gt_im[start_y:start_y+h, start_x:start_x+w]


def pad_or_crop(im, target_shape, cval=0):
    h, w = target_shape
    y, x = im.shape[:2]
    h_pad, w_pad = h-y, w-x

    # Vertical center padding
    if h_pad > 0:
        padding = ((h_pad//2, h_pad-h_pad//2), (0, 0))
        if len(im.shape) == 3:
            padding += ((0, 0),)
        im_padded = np.pad(im, padding, mode='constant', constant_values=cval)
    # Vertical center cropping
    else:
        start_y = max(0, (y//2)-(h//2))
        im_padded = im[start_y:start_y+h, :]
    # Horizontal center padding
    if w_pad > 0:
        padding = ((0, 0), (w_pad//2, w_pad-w_pad//2))
        if len(im.shape) == 3:
            padding += ((0, 0),)
        im_padded = np.pad(im_padded, padding, mode='constant', constant_values=cval)
    # Horizontal center cropping
    else:
        start_x = max(0, x//2-(w//2))
        im_padded = im_padded[:, start_x:start_x+w]

    return im_padded


def rotate(im, angle, scale, interpolation, cval=0):
    """
    Rotates an image by the specified angle, with optional rescaling/zooming.

    :param im: the input image
    :param angle: the rotation angle in degrees
    :param scale: the zoom factor, 1.0 means no zoom
    :param interpolation: the interpolation method:
     - cv2.INTER_LINEAR for an image,
     - cv2.INTER_NEAREST for its ground truth pixels' labels
    :param cval: the constant value to use for filling empty pixels
    :return: the rotated and optionally rescaled/zoomed image
    """
    h, w = im.shape[:2]
    c_x, c_y = w//2, h//2

    mat = cv2.getRotationMatrix2D((c_x, c_y), angle, scale)
    cos = np.abs(mat[0, 0])
    sin = np.abs(mat[0, 1])

    # New image bounding box
    new_w = int(round(h*sin+w*cos))
    new_h = int(round(h*cos+w*sin))

    # Add translation
    mat[0, 2] += (new_w/2)-c_x
    mat[1, 2] += (new_h/2)-c_y

    # Return the rotated image
    return cv2.warpAffine(im, mat, (new_w, new_h), flags=interpolation, borderValue=(cval, cval, cval))


def random_rotate(im, gt, rotation_range, zoom_range, ignore_label):
    """
    Randomly rotates an image and its ground truth labels within the specified angles range,
    with optional zoom/scale randomly chosen from the specified factors range.

    :param im: the image
    :param gt: the segmentation labels
    :param rotation_range: a tuple made of the min & max rotation angles in degrees, for example (-10, 10)
    :param zoom_range: a tuple made of the min & max zoom factors, for example (0.8, 1.2)
    :param ignore_label:
    :return: the rotated and optionally rescaled/zoomed image and ground truth labels
    """
    angle = np.random.uniform(*rotation_range)
    zoom = 1.0 if zoom_range is None else np.random.uniform(*zoom_range)
    x = rotate(im, angle, scale=zoom, interpolation=cv2.INTER_LINEAR)
    y = rotate(gt, angle, scale=zoom, interpolation=cv2.INTER_NEAREST, cval=ignore_label)
    return x, y


def labels2colors(labels, cmap):
    """
    Converts a matrix of labels (ie. pixels' semantic class) to an image with their equivalent color.

    :param labels: a matrix of class labels
    :param cmap: the color map, as a list of RGB tuples:
     - The first item is the background class color,
     - The last item is the color of the void/ignore class
    :return: an image with colorized pixels' class
    """
    labels_colored = np.zeros((*labels.shape, 3), dtype='uint8')
    for label in np.unique(labels):
        label_mask = labels == label
        label_mask = np.dot(np.expand_dims(label_mask, axis=2), np.array(cmap[label]).reshape((1, -1)))
        labels_colored += label_mask.astype('uint8')
    return labels_colored


def colors2labels(im, cmap, one_hot=False):
    """
    Converts a RGB ground truth segmentation image into a labels matrix with optional one-hot encoding.
    """
    if one_hot:
        labels = np.zeros((*im.shape[:-1], len(cmap)-1), dtype='uint8')
        for i, color in enumerate(cmap):
            labels[:, :, i] = np.all(im == color, axis=2).astype('uint8')
    else:
        labels = np.zeros(im.shape[:-1], dtype='uint8')
        for i, color in enumerate(cmap):
            labels += i * np.all(im == color, axis=2).astype(dtype='uint8')
    return labels


def apply_mask(im, im_pred):
    """
    Overlays the predicted class labels onto an image using the alpha channel.
    This function assumes that the background label is the black color.
    """
    b_channel, g_channel, r_channel = cv2.split(im_pred)
    alpha_channel = 127 * np.ones(b_channel.shape, dtype=b_channel.dtype)
    # Make background pixels fully transparent
    alpha_channel -= 127 * np.all(im_pred == np.array([0, 0, 0]), axis=2).astype(b_channel.dtype)
    im_pred = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
    mask = PIL.Image.fromarray(im_pred, mode='RGBA')
    masked_img = PIL.Image.fromarray(im)
    masked_img.paste(mask, box=None, mask=mask)
    return np.array(masked_img)


def random_transform(im, gt, target_shape, saturation_range=None, value_range=None,
                     brightness_range=None, contrast_range=None, blur_params=None, flip_lr=False,
                     rotation_range=None, shift_range=None, zoom_range=None, ignore_label=0):
    """
    Applies a list of transformations to an image and its segmentation labels.
    Suggestion for 'blur_params': {'ksize_max': (9, 9), 'sigma_max': (1.5, 1.5)}

    :param im: the image to transform
    :param gt: the pixels' labels (ie. semantic class) in sparse format
    :param target_shape: the transformed image and its segmentation labels matrix are either padded or cropped
    to the target shape. Depending on the image size after zooming and rotation The image will be padded
    :param saturation_range: adjust saturation by randomly choosing a value in the specified range
    :param value_range:
    :param brightness_range:
    :param contrast_range:
    :param blur_params
    :param flip_lr: horizontally flip the image
    :param rotation_range: tuple with min/max rotation factor in degrees
    :param shift_range: tuple with the maximum vertical & horizontal absolute translation amount in pixels.
           The translation is achieved by center padding the image by 'shift_range' and then randomly cropping
           the result to 'target_shape'.
    :param zoom_range: tuple with min/max zoom percentage factor
    :param ignore_label: the label of the ignore/void class, defaults to 0, ie. the background class label
    :return: the transformed image and pixel labels
    """
    x, y = im, gt

    # Adjust saturation and value
    saturation = 0 if saturation_range is None else np.random.randint(*saturation_range)
    value = 0 if value_range is None else np.random.randint(*value_range)
    x = adjust_saturation_and_value(x, saturation, value)

    # Adjust brightness and contrast
    brightness = 0 if brightness_range is None else np.random.randint(*brightness_range)
    contrast = 0 if contrast_range is None else np.random.randint(*contrast_range)
    x = adjust_brightness_and_contrast(x, brightness, contrast)

    # Blur image
    if blur_params is not None:
        x = random_blur(x, blur_params['ksize_max'], blur_params['sigma_max'])

    # Horizontal flip
    if flip_lr:
        x, y = random_flip_axis(x, y, 1)

    # Rotation with optional zoom using an affine transform
    if rotation_range is not None:
        x, y = random_rotate(x, y, rotation_range, zoom_range, ignore_label)

    # Zoom only using resize
    elif zoom_range is not None:
        x, y = random_zoom(x, y, zoom_range)

    # Translate image using a combination of padding and random cropping
    padded_shape = np.asarray(target_shape)
    if shift_range is not None:
        padded_shape += np.asarray(shift_range)

    x = pad_or_crop(x, padded_shape)
    y = pad_or_crop(y, padded_shape, cval=ignore_label)

    if shift_range is not None:
        x, y = random_crop(x, y, target_shape)

    return x, y
