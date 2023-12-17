import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from diffusion_v2 import UNet
import gc


class Trainer:
    def __init__(self, c_in=3, c_out=3, ch_list=(128, 128, 256, 256, 512, 512), attn_res=(16,), heads=-1, cph=32,
                 num_classes=1, epochs=100, lr=2e-5, time_steps=1000, image_size=256, ema_iterations_start=5, no_of=64,
                 freq=5, sample_ema_only=True, beta_start=1e-4, beta_end=0.02,
                 tpu_instance=None, train_logdir="logs/train_logs/", val_logdir="logs/val_logs/"):

        self.image_size = image_size
        self.time_steps = time_steps
        self.epochs = epochs
        self.lr = lr
        self.ema_iterations_start = ema_iterations_start
        self.no_of = no_of
        self.freq = freq
        self.sample_ema_only = sample_ema_only
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.c_in = c_in
        self.c_out = c_out
        self.ch_list = ch_list
        self.attn_res = attn_res
        self.num_classes = num_classes
        self.heads = heads
        self.cph = cph

        if not isinstance(tpu_instance, tf.distribute.TPUStrategy):
            raise Exception("Provide a tf.distribute.TPUStrategy Instance")
        self.strategy = tpu_instance

        with self.strategy.scope():
            self.model = UNet(c_in=self.c_in, c_out=self.c_out, ch_list=self.ch_list, attn_res=self.attn_res,
                              heads=self.heads, cph=self.cph, num_classes=self.num_classes, cfg_weight=3, mid_attn=True,
                              resamp_with_conv=True, num_res_blocks=2, img_res=self.image_size, dropout=0,
                              time_steps=self.time_steps, beta_start=self.beta_start, beta_end=self.beta_end)

            self.ema_model = UNet(c_in=self.c_in, c_out=self.c_out, ch_list=self.ch_list, attn_res=self.attn_res,
                                  heads=self.heads, cph=self.cph, num_classes=self.num_classes, cfg_weight=3,
                                  mid_attn=True, resamp_with_conv=True, num_res_blocks=2, img_res=self.image_size, dropout=0,
                                  time_steps=self.time_steps, beta_start=self.beta_start, beta_end=self.beta_end)

            mse = tf.keras.losses.MeanSquaredError(name="MSELoss",
                                                   reduction=tf.keras.losses.Reduction.NONE
                                                   )

            def compute_loss(y_true, y_pred):
                return tf.nn.compute_average_loss(
                    tf.math.reduce_mean(
                        mse(y_true, y_pred),
                        axis=[1, 2]
                    )
                )

            self.compute_loss = compute_loss

            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
            self.ema_decay = tf.constant(0.99, dtype=tf.float32)

        self.loss_tracker = tf.keras.metrics.Mean()
        self.val_loss_tracker = tf.keras.metrics.Mean()

        self.train_logging = tf.summary.create_file_writer(train_logdir)
        self.val_logging = tf.summary.create_file_writer(val_logdir)

        self.best_loss = 24.0

        self.train_counter = 0
        self.val_counter = 0

    def sample_time_step(self, size):
        return tf.experimental.numpy.random.randint(0, self.time_steps, size=(size,))

    @tf.function
    def train_step(self, iterator):
        def unit_step(data):
            image = data['image']
            cls = data['context']

            if self.num_classes > 1:
                if tf.random.uniform(minval=0, maxval=1, shape=()) < 0.1:
                    cls = tf.fill(
                        dims=tf.shape(cls),
                        value=0
                    )
            else:
                cls = None

            # Sample time step
            t = self.sample_time_step(size=tf.shape(image)[0])

            # Forward Noise
            noised_image, noise = self.model.forward_diff([image, t])
            t = tf.expand_dims(t, axis=-1)

            # Forward pass
            with tf.GradientTape() as tape:
                output = self.model([noised_image, t, cls], training=True)
                loss = self.compute_loss(noise, output)

            # EMA
            if self.optimizer.iterations >= self.ema_iterations_start:
                for main_weights, ema_weights in zip(self.model.trainable_weights,
                                                     self.ema_model.trainable_weights):
                    ema_weights.assign(
                        ema_weights * self.ema_decay + main_weights * (1 - self.ema_decay)
                    )
            else:
                for main_weights, ema_weights in zip(self.model.trainable_weights,
                                                     self.ema_model.trainable_weights):
                    ema_weights.assign(main_weights)

            # BackProp & Update
            grads = tape.gradient(loss, self.model.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
            return loss

        losses = self.strategy.run(unit_step, args=(next(iterator),))
        return self.strategy.reduce(tf.distribute.ReduceOp.SUM, losses, axis=None)

    # eval step using ema_model
    @tf.function
    def test_step(self, iterator):
        def unit_step(data):
            image = data['image']
            cls = data['context']

            if not self.num_classes > 1:
                cls = None

            t = self.sample_time_step(size=tf.shape(image)[0])

            noised_image, noise = self.ema_model.forward_diff([image, t])
            t = tf.expand_dims(t, axis=-1)

            output = self.ema_model([noised_image, t, cls], training=False)
            loss = self.compute_loss(noise, output)

            return loss

        losses = self.strategy.run(unit_step, args=(next(iterator),))
        return self.strategy.reduce(tf.distribute.ReduceOp.SUM, losses, axis=None)

    # eval step using main_model
    @tf.function
    def test_step_main(self, iterator):
        def unit_step(data):
            image = data['image']
            cls = data['context']

            if not self.num_classes > 1:
                cls = None

            t = self.sample_time_step(size=tf.shape(image)[0])

            noised_image, noise = self.model.forward_diff([image, t])
            t = tf.expand_dims(t, axis=-1)

            output = self.model([noised_image, t, cls], training=False)
            loss = self.compute_loss(noise, output)

            return loss

        losses = self.strategy.run(unit_step, args=(next(iterator),))
        return self.strategy.reduce(tf.distribute.ReduceOp.SUM, losses, axis=None)

    @tf.function
    def reverse_diffusion(self, images, time, cls, use_main):
        if use_main:
            images = self.strategy.run(self.model.diffuse_step, args=(images, time, cls))
        else:
            images = self.strategy.run(self.ema_model.diffuse_step, args=(images, time, cls))
        return images

    def sample(self, epoch, no_of, use_main=False):
        # Sample Gaussian noise
        images = tf.random.normal((no_of, self.image_size, self.image_size, 3))
        images = self.strategy.experimental_distribute_dataset(
            tf.data.Dataset.from_tensor_slices(images).batch(no_of, drop_remainder=True)
        )
        images = next(iter(images))

        # Conditional guidance
        if self.num_classes > 1:
            cls = tf.random.uniform((no_of // self.strategy.num_replicas_in_sync,),
                                    minval=1, maxval=self.num_classes + 1, dtype=tf.int32)
        else:
            cls = None

        # Reverse diffusion for t time steps
        use_main = tf.constant(use_main, dtype=tf.bool)
        for t in reversed(tf.range(0, self.time_steps)):
            images = self.reverse_diffusion(images, t, cls, use_main)

        images = self.strategy.experimental_local_results(images)
        images = tf.concat(images, axis=0)

        # Set pixel values in display range
        images = tf.clip_by_value(images, 0, 1)

        # Creating and saving sampled images plot
        images = iter(images)

        h = int(no_of ** 0.5)
        fig, ax = plt.subplots(h, h, figsize=(7, 7), gridspec_kw={
            'hspace': 0.0,
            'wspace': 0.0
        })
        for i in range(h):
            for j in range(h):
                ax[i][j].imshow(next(images))
                ax[i][j].axis('off')

        if use_main:
            dir = "results/normal_epoch_{0}.jpeg".format(epoch)
        else:
            dir = "results/ema_epoch_{0}.jpeg".format(epoch)

        fig.savefig(dir, pad_inches=0.03, bbox_inches='tight')
        plt.close(fig)

    def train(self, train_ds, val_ds, train_steps, val_steps):
        for epoch in range(self.epochs):
            # Make training and validation data iterable
            train_data = iter(train_ds)
            val_data = iter(val_ds)

            # Reset metrics states
            self.loss_tracker.reset_states()
            self.val_loss_tracker.reset_states()

            print("Epoch :", epoch)
            progbar = tf.keras.utils.Progbar(target=None, stateful_metrics=['loss', 'val_loss'])

            for train_batch_no in range(train_steps):
                # train step
                train_loss = self.train_step(train_data)
                self.train_counter += 1

                # update metrics
                self.loss_tracker.update_state(train_loss)
                progbar.update(train_batch_no, values=[('loss', self.loss_tracker.result())],
                               finalize=False)

                # log scores
                with self.train_logging.as_default(self.train_counter):
                    tf.summary.scalar("loss", self.loss_tracker.result())

            # Validation process
            val_loss = self.evaluate(val_data, val_steps, log_data=True, use_main=False)

            # Update scores
            progbar.update(train_batch_no, values=[
                ('loss', self.loss_tracker.result()),
                ('val_loss', val_loss)
            ], finalize=True)

            # Sampling images
            if epoch % self.freq == 0:
                self.sample(epoch, self.no_of, False)
                if self.sample_ema_only is False:
                    self.sample(epoch, self.no_of, True)

            # Saving best weights
            if self.val_loss_tracker.result() < self.best_loss:
                self.best_loss = self.val_loss_tracker.result()
                self.ema_model.save_weights("all_weights/best_ema_weights/weights")
                print("Best Weights saved...")

            # Saving all weights
            self.model.save_weights("all_weights/main_model_weights/weights")
            self.ema_model.save_weights("all_weights/ema_model_weights/weights")
            tf.saved_model.save(self.optimizer, "all_weights/optimizer_states/states")

            # Garbage Collect
            gc.collect()

    # eval function using main_model or ema_model
    def evaluate(self, val_data, val_steps, log_data=False, use_main=False):
        self.val_loss_tracker.reset_states()

        for _ in range(val_steps):
            if use_main:
                val_loss = self.test_step_main(val_data)
            else:
                val_loss = self.test_step(val_data)

            self.val_loss_tracker.update_state(val_loss)

            if log_data:
                self.val_counter += 1
                with self.val_logging.as_default(self.val_counter):
                    tf.summary.scalar("val_loss", self.val_loss_tracker.result())

        return self.val_loss_tracker.result()
