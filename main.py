import tensorflow as tf
from data_loader import GaziBrainsDataLoader

from data_processing import load_image_test, load_image_train
from training import fit

from matplotlib import pyplot as plt

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

#     plt.imshow(t1w_img)
#     plt.show()

#     plt.imshow(t2w_img)
#     plt.show()

#     plt.imshow(flair_img)
#     plt.show()

assert(len(train_x) == len(train_y))
train_len = len(train_x)

val_x, val_y = gazi_brains_data.get_validation_data()
assert(len(val_x) == len(val_y))
test_len = len(val_x)

print("------------------")
print(f'Train: {train_len} datapoints\nTest: {test_len} datapoints')
print("------------------")


train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))

# train_dataset = tf.data.Dataset.list_files(str(PATH / 'train/*.jpg'))
train_dataset = train_dataset.map(load_image_train,
                                  num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

# try:
#   test_dataset = tf.data.Dataset.list_files(str(PATH / 'test/*.jpg'))
# except tf.errors.InvalidArgumentError:
#   test_dataset = tf.data.Dataset.list_files(str(PATH / 'val/*.jpg'))
# test_dataset = test_dataset.map(load_image_test)
# test_dataset = test_dataset.batch(BATCH_SIZE)

# fit(train_dataset, test_dataset, steps=40000)