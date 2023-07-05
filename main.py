from model import IMG_HEIGHT, IMG_WIDTH
import tensorflow as tf
from data_loader import GaziBrainsDataLoader

from data_processing import load_image_test, load_image_train, rescale, resize
from training import fit

from matplotlib import pyplot as plt
import numpy as np

PATH = "/Users/tolgaozgun/gazi_brains_2020/data/GAZI_BRAINS_2020"
BUFFER_SIZE = 400
BATCH_SIZE = 1

gazi_brains_data = GaziBrainsDataLoader(PATH, BATCH_SIZE, validation_split=0.2)
train_x, train_y = gazi_brains_data.get_train_data()

# Plot random images from the dataset



# for i in range(5):
#     image = train_x[i]
#     mask = train_y[i]
#     print(f'Image shape: {image.shape}')
#     print(f'Mask shape: {mask.shape}')
#     print(f'Image dtype: {image.dtype}')
#     print(f'Mask dtype: {mask.dtype}')

#     # Split the image into channels
#     t1w_img = image[..., 0]
#     t2w_img = image[..., 1]
#     flair_img = image[..., 2]


#     plt.title("t1w_img")
#     plt.imshow(t1w_img)
#     plt.show()
#     three_d_image = tf.expand_dims(t1w_img, axis=-1)
#     resized_image = tf.image.resize(three_d_image, [IMG_WIDTH, IMG_HEIGHT])
#     plt.title("resized image")
#     plt.imshow(resized_image)
#     plt.show()

#     plt.imshow(t2w_img)
#     # resized_image = tf.image.resize(t2w_img, [IMG_WIDTH, IMG_HEIGHT])
#     # plt.imshow(resized_image)
#     plt.show()

#     plt.imshow(flair_img)
#     # resized_image = tf.image.resize(flair_img, [IMG_WIDTH, IMG_HEIGHT])
#     # plt.imshow(resized_image)
#     plt.show()

    



assert(len(train_x) == len(train_y))
train_len = len(train_x)

new_train_x, new_train_y = [], []
for i in range(train_len):
    cur_y = np.expand_dims(train_y[i], axis=-1)
    # np.append(cur_y, np.zeros_like(cur_y), axis=2)
    # np.append(cur_y, np.zeros_like(cur_y), axis=2)
    # cur_y = np.stack((train_y[i],) * 3, axis=-1)

    # height, width = train_y[i].shape

    # cur_y = np.empty((height, width, 3), dtype=np.float64)
    # cur_y[:, :, 0] = train_y[i]
    # cur_y[:, :, 1] = train_y[i]
    # cur_y[:, :, 2] = train_y[i]

    
    # if i % 1000 == 0:
    #     fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(12, 6))
    #     ax1.imshow(train_y[i])
    #     ax2.imshow(cur_y)
    #     plt.show()
    out_x = rescale(train_x[i])
    cur_y = rescale(cur_y)
    out_x, out_y = resize(out_x, cur_y, IMG_HEIGHT, IMG_WIDTH)
    new_train_x.append(out_x)
    new_train_y.append(out_y)

    # plt.imshow(out_y)
    # plt.show()
    # print(f'out_x {out_x.shape}')
    # print(f'out_y {out_y.shape}')


train_dataset = tf.data.Dataset.from_tensor_slices((new_train_x, new_train_y))

train_dataset = train_dataset.map(load_image_train,
                                  num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

print("Train dataset loaded successfully.")

print("------------------")
print(f'Train: {train_len} datapoints')
print("------------------")

val_x, val_y = gazi_brains_data.get_validation_data()
assert(len(val_x) == len(val_y))
test_len = len(val_x)

new_val_x, new_val_y = [], []
for i in range(test_len):
    cur_y = np.expand_dims(val_y[i], axis=-1)
    # cur_y = np.stack((val_y[i],) * 3, axis=-1)

    height, width = val_y[i].shape

    # cur_y = np.empty((height, width, 3), dtype=np.float64)
    # cur_y[:, :, 0] = val_y[i]
    # cur_y[:, :, 1] = val_y[i]
    # cur_y[:, :, 2] = val_y[i]

    # if i % 100 == 0:
    #     fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(12, 6))
    #     ax1.imshow(val_y[i])
    #     ax2.imshow(cur_y)
    #     plt.show()
    # cur_y[:, :, 1] = val_y[i] 
    # cur_y[:, :, 2] = val_y[i] 

    out_x = rescale(val_x[i])
    cur_y = rescale(cur_y)
    # print(f'out_x type: {type(out_x)}')
    # print(f'cur_y type: {type(cur_y)}')
    out_x, out_y = resize(out_x, cur_y, IMG_HEIGHT, IMG_WIDTH)
    new_val_x.append(out_x)
    new_val_y.append(out_y)

test_dataset = tf.data.Dataset.from_tensor_slices((new_val_x, new_val_y))

test_dataset = test_dataset.map(load_image_test,
                                  num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.shuffle(BUFFER_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

print("Test dataset loaded successfully.")

print("------------------")
print(f'Test: {test_len} datapoints')
print("------------------")

fit(train_dataset, test_dataset, steps=40000)


# try:
#   test_dataset = tf.data.Dataset.list_files(str(PATH / 'test/*.jpg'))
# except tf.errors.InvalidArgumentError:
#   test_dataset = tf.data.Dataset.list_files(str(PATH / 'val/*.jpg'))
# test_dataset = test_dataset.map(load_image_test)
# test_dataset = test_dataset.batch(BATCH_SIZE)
