from matplotlib import pyplot as plt
import tensorflow as tf
import time
import os
import datetime
from model import Discriminator, Generator, discriminator_loss, generator_loss
from IPython import display


log_dir = "logs/"

summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# Mac only
# generator_optimizer = tf.keras.optimizers.legacy.Adam(2e-4, beta_1=0.5)
# discriminator_optimizer = tf.keras.optimizers.legacy.Adam(2e-4, beta_1=0.5)

discriminator = Discriminator()
generator = Generator()

checkpoint_dir = 'training_checkpoints'
current_dir = os.getcwd()
checkpoint_abs_dir = os.path.join(current_dir, checkpoint_dir)

print(f"Checkpoint directory: {checkpoint_abs_dir}")

if not (os.path.exists(checkpoint_abs_dir) and os.path.isdir(checkpoint_abs_dir)):
  print("Checkpoint directory does not exist, creating one..")
  os.mkdir(checkpoint_abs_dir)


checkpoint_prefix = os.path.join(checkpoint_abs_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

def generate_images(model, test_input, tar):
  prediction = model(test_input, training=True)
  plt.figure(figsize=(15, 15))

  display_list = [test_input[0], tar[0], prediction[0]]
  title = ['Input Image', 'Ground Truth', 'Predicted Image']

  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    
    # Show the ground truth as grayscale as it only has a single channel.
    if i == 1:
      plt.imshow(display_list[i] * 0.5 + 0.5, cmap="gray")
    else:
      plt.imshow(display_list[i] * 0.5 + 0.5)
      
    plt.axis('off')
  plt.show()



@tf.function
def train_step(input_image, target, step):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator(input_image, training=True)

    disc_real_output = discriminator([input_image, target], training=True)
    disc_generated_output = discriminator([input_image, gen_output], training=True)

    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

  generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))

  with summary_writer.as_default():
    tf.summary.scalar('gen_total_loss', gen_total_loss, step=step//1000)
    tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step//1000)
    tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step//1000)
    tf.summary.scalar('disc_loss', disc_loss, step=step//1000)


def fit(train_ds, test_ds, steps):
  example_input, example_target = next(iter(test_ds.take(1)))
  start = time.time()

  for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
    # print(f"Step {step}")
    if (step) % 1000 == 0:
      display.clear_output(wait=True)

      if step != 0:
        print(f'Time taken for 1000 steps: {time.time()-start:.2f} sec\n')

      start = time.time()

      generate_images(generator, example_input, example_target)
      print(f"Step: {step//1000}k")

    train_step(input_image, target, step)

    if (step+1) % 10 == 0:
      print('.', end='', flush=True)

    if (step + 1) % 5000 == 0:
      checkpoint.save(file_prefix=checkpoint_prefix)


def load_checkpoint():
  checkpoint.restore(tf.train.latest_checkpoint(checkpoint_abs_dir))
  print("Checkpoint restored.")