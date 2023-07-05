import tensorflow as tf
from model import IMG_HEIGHT, IMG_WIDTH
from matplotlib import pyplot as plt
import numpy as np

# def load(image_file):
#   image = tf.io.read_file(image_file)
#   image = tf.io.decode_jpeg(image)
#   w = tf.shape(image)[1]
#   w = w // 2
#   input_image = image[:, w:, :]
#   real_image = image[:, :w, :]
#   input_image = tf.cast(input_image, tf.float32)
#   real_image = tf.cast(real_image, tf.float32)

#   return input_image, real_image
# cnt = 0

def resize(input_image, real_image, height, width):
  input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  return input_image, real_image

def random_crop(input_image, real_image):
  stacked_image = tf.stack([input_image, real_image], axis=0)
  cropped_image = tf.image.random_crop(
      stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

  return cropped_image[0], cropped_image[1]


def normalize(input_image, real_image):
  input_image = (input_image / 127.5) - 1
  real_image = (real_image / 127.5) - 1

  return input_image, real_image

def rescale(image):
    # Find the minimum and maximum values in the image

    # plt.imshow(image)
    # plt.show()
    min_val = np.min(image)
    max_val = np.max(image)

    print(f'min_val: {min_val}')
    print(f'max val: {max_val}')
    
    # Scale the image to the range of 0 to 255
    scaled_image = (image - min_val) * (255.0 / (max_val - min_val))
    
    # Convert the data type to uint8 (8-bit unsigned integer)
    # scaled_image = scaled_image.astype(np.uint8)

    # plt.imshow(scaled_image)
    # plt.show()
    
    return scaled_image

@tf.function()
def random_jitter(input_image, real_image):
  
  input_image, real_image = resize(input_image, real_image, 286, 286)

  
  input_image, real_image = random_crop(input_image, real_image)

  if tf.random.uniform(()) > 0.5:
    
    input_image = tf.image.flip_left_right(input_image)
    real_image = tf.image.flip_left_right(real_image)

  return input_image, real_image

def load_image_train(input_image, real_image):
  # input_image, real_image = load(image_file)
  # input_image, real_image = random_jitter(input_image, real_image)
  input_image, real_image = normalize(input_image, real_image)
  # global cnt
  # if cnt % 200 == 0:
  #   print(input_image.shape)
  #   print(real_image.shape)
    # fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(12, 6))
    # ax1.imshow(real_image.permute(1, 2, 0))
    # ax2.imshow(input_image.permute(1, 2, 0))
    # plt.show()
  # cnt += 1

  return input_image, real_image

def load_image_test(input_image, real_image):
  # input_image, real_image = load(image_file)
  # input_image, real_image = resize(input_image, real_image,
                                  #  IMG_HEIGHT, IMG_WIDTH)
  input_image, real_image = normalize(input_image, real_image)
  # global cnt
  # if cnt % 200 == 0:
  #   print(input_image.shape)
  #   print(real_image.shape)
    # fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(12, 6))
    # ax1.imshow(real_image.permute(1, 2, 0))
    # ax2.imshow(input_image.permute(1, 2, 0))
    # plt.show()
  # cnt += 1

  return input_image, real_image