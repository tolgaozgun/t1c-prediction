import tensorflow as tf
from data_loader import GaziBrainsDataLoader

from data_processing import load_image_test, load_image_train
from training import fit

PATH = "/Users/tolgaozgun/gazi_brains_2020/data/GAZI_BRAINS_2020"
BUFFER_SIZE = 400
BATCH_SIZE = 1

gazi_brains_data = GaziBrainsDataLoader(PATH, BATCH_SIZE, validation_split=0.2)


test_dataset = gazi_brains_data.get_validation_data()

print(type(test_dataset))
print(test_dataset)

# train_dataset = tf.data.Dataset.list_files(str(PATH / 'train/*.jpg'))
# train_dataset = train_dataset.map(load_image_train,
#                                   num_parallel_calls=tf.data.AUTOTUNE)
# train_dataset = train_dataset.shuffle(BUFFER_SIZE)
# train_dataset = train_dataset.batch(BATCH_SIZE)

# try:
#   test_dataset = tf.data.Dataset.list_files(str(PATH / 'test/*.jpg'))
# except tf.errors.InvalidArgumentError:
#   test_dataset = tf.data.Dataset.list_files(str(PATH / 'val/*.jpg'))
# test_dataset = test_dataset.map(load_image_test)
# test_dataset = test_dataset.batch(BATCH_SIZE)

# fit(train_dataset, test_dataset, steps=40000)