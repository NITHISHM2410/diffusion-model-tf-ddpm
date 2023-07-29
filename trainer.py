from unet import UNet
import tensorflow as tf
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self, time_steps, epochs, lr, ema_start, time_dim=256, image_size=64):
        self.image_size = image_size
        self.time_steps = time_steps
        self.time_dim = time_dim
        self.epochs = epochs
        self.lr = lr
        self.ema_start = ema_start

        self.model = UNet(time_steps=self.time_steps,
                          time_dim=self.time_dim,
                          input_size=self.image_size)
        self.model.build([(None, self.image_size, self.image_size, 3), (None, 1)])

        self.ema_model = UNet(time_steps=self.time_steps,
                              time_dim=self.time_dim,
                              input_size=self.image_size)
        self.ema_model.build([(None, self.image_size, self.image_size, 3), (None, 1)])

        self.mse = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

        self.loss_tracker = tf.keras.metrics.Mean()
        self.val_loss_tracker = tf.keras.metrics.Mean()
        self.ema = tf.train.ExponentialMovingAverage(decay=0.99)

    def sample_time_step(self, size):
        return tf.experimental.numpy.random.randint(1, self.time_steps, size=(size,))

    @tf.function
    def train_step(self, data):
        image = data['image']
        t = self.sample_time_step(size=tf.shape(image)[0])
        noised_image, noise = self.model.forward_noiser([image, t])
        t = tf.expand_dims(t, axis=-1)

        with tf.GradientTape() as tape:
            output = self.model([noised_image, t])
            loss = self.mse(output, noise)

        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return loss

    # forward passes the data using ema_weights model
    @tf.function
    def test_step(self, data):
        image = data['image']
        t = self.sample_time_step(size=tf.shape(image)[0])
        noised_image, noise = self.ema_model.forward_noiser([image, t])
        t = tf.expand_dims(t, axis=-1)

        output = self.ema_model([noised_image, t])
        loss = self.mse(output, noise)

        return loss

    # forward passes the data using main model
    @tf.function
    def test_step_main(self, data):
        image = data['image']
        t = self.sample_time_step(size=tf.shape(image)[0])
        noised_image, noise = self.model.forward_noiser([image, t])
        t = tf.expand_dims(t, axis=-1)

        output = self.model([noised_image, t])
        loss = self.mse(output, noise)

        return loss

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 64, 64, 3), dtype=tf.float32),
                                  tf.TensorSpec(shape=(), dtype=tf.int32),
                                  tf.TensorSpec(shape=(), dtype=tf.bool)])
    def reverse_diffusion(self, images, time, use_main=False):
        batch = tf.shape(images)[0]
        time = tf.repeat(time, repeats=batch, axis=0)

        alpha = tf.gather(self.model.alphas, time)[:, None, None, None]
        beta = tf.gather(self.model.betas, time)[:, None, None, None]
        alpha_hat = tf.gather(self.model.alpha_hats, time)[:, None, None, None]

        time = tf.expand_dims(time, axis=-1)

        if use_main:
            predicted_noise = self.model([images, time])
        else:
            predicted_noise = self.ema_model([images, time])

        if time[0] > 1:
            noise = tf.random.normal(shape=tf.shape(images))
        else:
            noise = tf.zeros_like(images)

        images = (1 / tf.sqrt(alpha)) * (images - ((1 - alpha) / (tf.sqrt(1 - alpha_hat))) * predicted_noise) + tf.sqrt(
            beta) * noise
        return images

    # samples images either using main model or ema_model based on 'use_main' param
    def sample(self, epoch, no_of=2, use_main=False):
        images = tf.random.normal((no_of, self.image_size, self.image_size, 3))
        for t in reversed(range(1, self.time_steps)):
            images = self.reverse_diffusion(
                images,
                tf.constant(t, dtype=tf.int32),
                tf.constant(use_main, dtype=tf.bool)
            )
        images = tf.clip_by_value(images, 0, 1)

        fig, ax = plt.subplots(1, no_of, figsize=(5, 5), gridspec_kw={'hspace': 0.03, 'wspace': 0.03})
        for i in range(no_of):
            ax[i].imshow(images[i])
            ax[i].axis('off')

        if use_main:
            dir = "results/normal_epoch_{0}".format(epoch)
        else:
            dir = "results/ema_epoch_{0}".format(epoch)

        fig.savefig(dir, pad_inches=0.03, bbox_inches='tight')
        plt.close(fig)

    def train(self, train_data, val_data):
        for epoch in range(self.epochs):
            self.loss_tracker.reset_states()
            self.val_loss_tracker.reset_states()

            progbar = tf.keras.utils.Progbar(target=None, stateful_metrics=['loss', 'val_loss'])

            for train_batch, example in enumerate(train_data):
                train_loss = self.train_step(example)
                self.loss_tracker.update_state(train_loss)
                progbar.update(train_batch, values=[('loss', self.loss_tracker.result())],
                               finalize=False)

                if epoch >= self.ema_start:
                    self.ema.apply(self.model.trainable_weights)
                    for main_weights, ema_weights in zip(self.model.trainable_weights,
                                                         self.ema_model.trainable_weights):
                        ema_weights.assign(self.ema.average(main_weights))
                else:
                    self.ema_model.set_weights(self.model.get_weights())

            for val_batch, example in enumerate(val_data):
                val_loss = self.test_step(example)
                self.val_loss_tracker.update_state(val_loss)

            progbar.update(train_batch, values=[
                ('loss', self.loss_tracker.result()),
                ('val_loss', self.val_loss_tracker.result())
            ], finalize=True)

            if epoch % 5 == 0:
                self.sample(epoch, 2, False)
                self.sample(epoch, 2, True)

            self.model.save_weights("all_weights/main_model_weights/weights")
            self.ema_model.save_weights("all_weights/ema_model_weights/weights")

    # evaluate either main model or ema_model based on 'use_main' param
    def evaluate(self, val_data, use_main=False):
        self.val_loss_tracker.reset_states()

        if use_main:
            for val_batch, example in enumerate(val_data):
                val_loss = self.test_step_main(example)
                self.val_loss_tracker.update_state(val_loss)
        else:
            for val_batch, example in enumerate(val_data):
                val_loss = self.test_step(example)
                self.val_loss_tracker.update_state(val_loss)

        return self.val_loss_tracker.result()

# Create a 'Trainer' instance and call 'instance.train(train,val)' to train
