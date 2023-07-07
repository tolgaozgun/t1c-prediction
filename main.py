from model import IMG_HEIGHT, IMG_WIDTH
import tensorflow as tf
from data_loader import GaziBrainsDataLoader

from data_processing import load_image_test, load_image_train, rescale, resize
from training import fit, load_checkpoint

from matplotlib import pyplot as plt
import numpy as np

PATH = "/tf/shared-datas/TurkBeyinProjesi/GaziBrains_BIDS/GAZI_BRAINS_2020/sourcedata/"
BUFFER_SIZE = 400
BATCH_SIZE = 1


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    print(e)


gazi_brains_data = GaziBrainsDataLoader(PATH, BATCH_SIZE, validation_split=0.2)
train_x, train_y = gazi_brains_data.get_train_data()


assert(len(train_x) == len(train_y))
train_len = len(train_x)

new_train_x, new_train_y = [], []
for i in range(train_len):
    cur_y = np.expand_dims(train_y[i], axis=-1)
    
    out_x = rescale(train_x[i])
    cur_y = rescale(cur_y)
    out_x, out_y = resize(out_x, cur_y, IMG_HEIGHT, IMG_WIDTH)
    new_train_x.append(out_x)
    new_train_y.append(out_y)


train_dataset = tf.data.Dataset.from_tensor_slices((new_train_x, new_train_y))
train_dataset = train_dataset.map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
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

    out_x = rescale(val_x[i])
    cur_y = rescale(cur_y)
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


print("Trying to load latest checkpoint...")
load_checkpoint()

print("Starting training...")
fit(train_dataset, test_dataset, steps=40000)
