import tensorflow as tf
from unet import UNet

model = UNet(time_steps=1000, input_size=64, num_classes=5)
model.load_weights("directory")
interpolate_weight = 3


@tf.function(input_signature=[tf.TensorSpec(shape=(None, 64, 64, 3), dtype=tf.float32),
                              tf.TensorSpec(shape=(), dtype=tf.int32),
                              tf.TensorSpec(shape=(None, 1), dtype=tf.int32)])
def reverse_diffusion(images, time, label):
    batch = tf.shape(images)[0]
    time = tf.repeat(time, repeats=batch, axis=0)

    alpha = tf.gather(model.alphas, time)[:, None, None, None]
    beta = tf.gather(model.betas, time)[:, None, None, None]
    alpha_hat = tf.gather(model.alpha_hats, time)[:, None, None, None]

    time = tf.expand_dims(time, axis=-1)
    zero_labels = tf.fill(dims=tf.shape(label), value=0)

    predicted_noise_uncond = model([images,
                                    time,
                                    zero_labels])
    predicted_noise_cond = model([images,
                                  time,
                                  label])

    predicted_noise = predicted_noise_uncond + interpolate_weight * (
            predicted_noise_cond - predicted_noise_uncond)

    if time[0] > 1:
        noise = tf.random.normal(shape=tf.shape(images))
    else:
        noise = tf.zeros_like(images)

    images = (1 / tf.sqrt(alpha)) * (images - ((1 - alpha) / (tf.sqrt(1 - alpha_hat))) * predicted_noise) + tf.sqrt(
        beta) * noise
    return images


def generate(no_of, class_no):
    images = tf.random.normal((no_of, model.input_size, model.input_size, 3))
    for t in reversed(range(1, model.time_steps)):
        images = reverse_diffusion(
            images,
            tf.constant(t, dtype=tf.int32),
            tf.expand_dims(class_no,axis=-1)
        )
    images = tf.clip_by_value(images, 0, 1)
    return images

# generated_images = generated()
