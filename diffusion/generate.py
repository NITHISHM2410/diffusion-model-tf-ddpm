import tensorflow as tf
from tqdm import tqdm


class GenerateImages:
    def __init__(self, model, device):
        """
        :param model: A trained diffusion model instance
        :param device: A tf.distribute.Strategy instance
        """
        self.model = model
        self.device = device

    @tf.function
    def distribute_fn(self, images, t, cls_list):
        output = self.device.run(self.model.diffuse_step, args=(images, t, cls_list))
        return output

    def sample(self, no_of, cls_list):
        # Sample Gaussian noise
        images = tf.random.normal((no_of, self.model.img_res, self.model.img_res, 3))

        images = self.device.experimental_distribute_dataset(
            tf.data.Dataset.from_tensor_slices(images).batch(no_of, drop_remainder=True)
        )
        images = next(iter(images))

        # Reverse diffusion for t time steps
        for t in tqdm(reversed(tf.range(0, self.model.time_steps)), "Sampling Images...",
                      total=self.model.time_steps, leave=True, position=0):
            images = self.distribute_fn(images, t, cls_list)

        images = self.device.experimental_local_results(images)
        images = tf.concat(images, axis=0)

        # Set pixel values in display range
        images = tf.clip_by_value(images, 0, 1)
        return images


def read_image(image_locations, size):
    """
    Read images from locations and applies basic transformations

    :param image_locations: list of image locations
    :param size: required image size
    """
    image_locations = tf.data.Dataset.from_tensor_slices(image_locations)

    def transform(img):
        img = tf.io.decode_jpeg(tf.io.read_file(img), channels=3)
        img = tf.image.resize(img, (size, size))
        img = img / 255
        return img

    images = image_locations.map(transform, num_parallel_calls=tf.data.AUTOTUNE)
    return next(iter(images))


