import tensorflow as tf
import matplotlib.pyplot as plt
from diffusion_tf import UNet
from tqdm import tqdm
import gc
import os


class Trainer:
    def __init__(self, c_in=3, c_out=3, ch_list=(128, 256, 256, 256), attn_res=(16,), heads=1, cph=32,
                 norm_g=32, num_classes=1, lr=2e-5, time_steps=1000, image_size=64, ema_iterations_start=5000,
                 loss_type='l2', no_of=64, freq=3, sample_ema_only=True, beta_start=1e-4, beta_end=0.02,
                 is_logging=False, device=None, train_logdir="logs/train_logs/", val_logdir="logs/val_logs/",
                 save_dir=None, checkpoint_name=None):

        """

        :param c_in: no of input channels.
        :param c_out: no of output channels.
        :param ch_list: list of channels to use across Up and Down sampling.
        :param attn_res: list of resolutions to use Attention Mechanism.
        :param heads: no of attention heads (if set to -1, then heads are chosen as <current channels/cph>).
        :param cph: no of channels per head (used if heads is set to -1).
        :param norm_g: number of groups for group norm.
        :param num_classes: no of classes for Conditional training ( choose 1 for Unconditional training).
        :param lr: constant learning rate to train with.
        :param time_steps: no of diffusion time steps.
        :param image_size: input image size.
        :param ema_iterations_start: no of iterations to start EMA.
        :param loss_type: 'l1' or 'l2' loss.
        :param no_of: no of images to generate at 'freq' frequency.
        :param freq: frequency of generating samples.
        :param sample_ema_only: Sample only from EMA model.
        :param beta_start: Starting Beta for ForwardDiffusion.
        :param beta_end: Ending Beta for ForwardDiffusion.
        :param device: A tf.distribute.Strategy instance.
        :param is_logging: Boolean value, whether to log results.
        :param train_logdir: Directory to log train results.
        :param val_logdir: Directory to log validation results.
        :param save_dir: Directory of the saved checkpoint. If no saved checkpoint available(no previous training) in
        the mentioned path then, one will be created during model training in the mentioned path.
        :param checkpoint_name: Name of the saved checkpoint. If no saved checkpoint available(no previous training) in
        above-mentioned path and name, one will be created during model training.

        """

        self.image_size = image_size
        self.ema_iterations_start = ema_iterations_start
        self.no_of = no_of
        self.freq = freq
        self.sample_ema_only = sample_ema_only

        self.c_in = c_in
        self.c_out = c_out
        self.ch_list = ch_list
        self.attn_res = attn_res
        self.time_steps = time_steps
        self.norm_g = norm_g
        self.num_classes = num_classes
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.loss_type = loss_type
        self.heads = heads
        self.lr = lr
        self.cph = cph

        if not isinstance(device, tf.distribute.Strategy):
            raise Exception("Provide a tf.distribute.Strategy Instance")
        self.device = device

        with self.device.scope():
            # Initialize models
            self.model = self.initialize_model()
            self.ema_model = self.initialize_model()
            self.best_model = self.initialize_model()

            # Loss function MSE(l2) or MAE(l1)
            if loss_type == 'l2':
                self.base_loss = tf.keras.losses.MeanSquaredError(
                    name="MSELoss",
                    reduction=tf.keras.losses.Reduction.NONE
                )
            elif loss_type == 'l1':
                self.base_loss = tf.keras.losses.MeanAbsoluteError(
                    name="MAELoss",
                    reduction=tf.keras.losses.Reduction.NONE
                )
            else:
                raise Exception("provide l1 or l2 loss_fn")

            self.compute_loss = self.compute_loss

            # Optimizer
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
            self.optimizer.build(self.model.trainable_weights)
            self.ema_decay = tf.constant(0.99, dtype=tf.float32)

        # Train & Val step counter for tf.summary logging's.
        self.train_counter = tf.Variable(0, trainable=False, dtype=tf.int32)
        self.val_counter = tf.Variable(0, trainable=False, dtype=tf.int32)

        # Checkpoints
        self.checkpoint = None
        self.save_manager = None
        self.save_dir = save_dir
        self.checkpoint_name = checkpoint_name
        self.set_checkpoint(self.save_dir, self.checkpoint_name)

        # Loss trackers
        self.loss_tracker = tf.keras.metrics.Mean()
        self.val_loss_tracker = tf.keras.metrics.Mean()

        # Whether to perform logging
        self.is_logging = is_logging
        if self.is_logging:
            self.train_logging = tf.summary.create_file_writer(train_logdir)
            self.val_logging = tf.summary.create_file_writer(val_logdir)

        # Initial best loss
        self.best_loss = 24.0

    def set_checkpoint(self, save_dir, ckpt_name):
        """
        Points the checkpoint manager to a checkpoint in the specified directory.

        :param save_dir: Directory of the checkpoint to be restored or save.
        :param ckpt_name: Name of the checkpoint to be restored or save.

        If no checkpoint is available in 'save_dir' named 'ckpt_name' then one will be created during training.
        If a checkpoint is available in the above-mentioned location, then it will be tracked. No new checkpoint will be created.

        """

        self.save_dir = save_dir
        self.checkpoint_name = ckpt_name

        self.checkpoint = tf.train.Checkpoint(
            model=self.model,
            ema_model=self.ema_model,
            best_model=self.best_model,
            optimizer=self.optimizer,
            train_counter=self.train_counter,
            val_counter=self.val_counter
        )
        self.save_manager = tf.train.CheckpointManager(
            checkpoint=self.checkpoint,
            directory=save_dir,
            checkpoint_name=ckpt_name,
            max_to_keep=50
        )

    def initialize_model(self):
        """
        Initializes UNet model.

        """
        return UNet(
            c_in=self.c_in, c_out=self.c_out, ch_list=self.ch_list, attn_res=self.attn_res,
            norm_g=self.norm_g, heads=self.heads, cph=self.cph, num_classes=self.num_classes,
            cfg_weight=3, mid_attn=True, resamp_with_conv=True, num_res_blocks=2,
            img_res=self.image_size, dropout=0,
            time_steps=self.time_steps, beta_start=self.beta_start, beta_end=self.beta_end
        )

    def resume_training(self, checkpoint_dir, checkpoint_name):
        """
        Resumes training from checkpoint.

        :param checkpoint_dir: Directory of existing checkpoint.
        :param checkpoint_name: Name of the existing checkpoint in 'checkpoint_dir'.

        """
        self.set_checkpoint(checkpoint_dir, checkpoint_name)
        ckpt_name = self.checkpoint_name + '-0'
        self.checkpoint.restore(os.path.join(self.save_dir, ckpt_name))
        if self.is_logging:
            print("Training logging done: ", self.train_counter.numpy())
            print("Validations logging done: ", self.val_counter.numpy())
        print("Optimizer iterations passed: ", self.optimizer.iterations.numpy())

    def compute_loss(self, y_true, y_pred):
        """
        Computes loss b/w true & predictions based on initialized loss function.

        :param y_true: target data.
        :param y_pred: predicted data.

        """
        return tf.nn.compute_average_loss(
            tf.math.reduce_mean(
                self.base_loss(y_true, y_pred),
                axis=[1, 2]
            )
        )

    def sample_time_step(self, size):
        """
        Samples a number from range(time_steps).

        :param size: shape of sampled array.
        """
        return tf.experimental.numpy.random.randint(0, self.time_steps, size=(size,))

    @tf.function
    def train_step(self, iterator):
        """
        A single training step.

        :param iterator: train tf.data.Dataset iterator.
        """

        def unit_step(data):
            # Gather data
            image = data['image']
            cls = data['context']

            # Conditional guidance
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

            # BackProp & Update
            grads = tape.gradient(loss, self.model.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

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

            return loss

        # Distribute train batches across devices
        losses = self.device.run(unit_step, args=(next(iterator),))

        # Whether to increment train counter for tensorflow summary logging's
        if self.is_logging:
            self.train_counter.assign_add(1)

        # Combine losses
        return self.device.reduce(tf.distribute.ReduceOp.SUM, losses, axis=None)

    # eval step using ema_model or main_model
    @tf.function
    def test_step(self, iterator, use_main, log_val_results):
        """
        A Single validation step using EMA model.

        :param iterator: val tf.data.Dataset iterator.
        :param use_main: boolean, whether to use main or ema model for validation.
        :param log_val_results: boolean value, whether to log validation results.
        """

        def unit_step(data):
            # Gather data
            image = data['image']
            cls = data['context']

            # Conditional guidance
            if not self.num_classes > 1:
                cls = None

            # Sample time step
            t = self.sample_time_step(size=tf.shape(image)[0])

            # Forward noise
            noised_image, noise = self.ema_model.forward_diff([image, t])
            t = tf.expand_dims(t, axis=-1)

            # Forward pass Main model or EMA model
            if use_main:
                output = self.model([noised_image, t, cls], training=False)
            else:
                output = self.ema_model([noised_image, t, cls], training=False)

            # Compute loss
            loss = self.compute_loss(noise, output)
            return loss

        # Distribute train batches across devices
        losses = self.device.run(unit_step, args=(next(iterator),))

        # Whether to increment val counter for tensorflow summary logging's
        if log_val_results:
            self.val_counter.assign_add(1)

        # Combine losses
        return self.device.reduce(tf.distribute.ReduceOp.SUM, losses, axis=None)

    @tf.function
    def reverse_diffusion(self, images, time, cls, use_main):
        """
        Calls the reverse diffusion step.

        :param images: input images after diffusion step t-1.
        :param time: current diffusion time step.
        :param cls: label value for class conditional generation.
        :param use_main: boolean value, whether to use EMA model for generating.
        """
        if use_main:
            images = self.device.run(self.model.diffuse_step, args=(images, time, cls))
        else:
            images = self.device.run(self.ema_model.diffuse_step, args=(images, time, cls))
        return images

    def sample(self, epoch, no_of, use_main=False):
        """
        Generates plots of generated images.

        :param epoch: epoch number to name the output file (Any string or int).
        :param no_of: number of images to generate.
        :param use_main: boolean value, whether to use main model for generating.
        """
        # Sample Gaussian noise
        images = tf.random.normal((no_of, self.image_size, self.image_size, self.c_in))
        images = self.device.experimental_distribute_dataset(
            tf.data.Dataset.from_tensor_slices(images).batch(no_of, drop_remainder=True)
        )
        images = next(iter(images))

        # Conditional guidance
        if self.num_classes > 1:
            cls = tf.random.uniform((no_of // self.device.num_replicas_in_sync,),
                                    minval=1, maxval=self.num_classes + 1, dtype=tf.int32)
        else:
            cls = None

        # Reverse diffusion for t time steps
        use_main = tf.constant(use_main, dtype=tf.bool)
        for t in tqdm(reversed(tf.range(0, self.time_steps)), desc='Sampling images...',
                      total=self.time_steps, leave=True, position=0):
            images = self.reverse_diffusion(images, t, cls, use_main)

        images = self.device.experimental_local_results(images)
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

    def train(self, epochs, train_ds, val_ds, train_steps, val_steps):
        """
        Training loop.

        :param epochs: number of epochs.
        :param train_ds: train tf.data.Dataset.
        :param val_ds: val tf.data.Dataset.
        :param train_steps: number of iterations per epoch for training data.
        :param val_steps: number of iterations per epoch for validation data.
        """
        for epoch in range(1, epochs + 1):
            # Make training and validation data iterable
            train_data = iter(train_ds)
            val_data = iter(val_ds)

            # Reset metrics states
            self.loss_tracker.reset_states()
            self.val_loss_tracker.reset_states()

            print("Epoch :", epoch)
            progbar = tf.keras.utils.Progbar(target=train_steps, stateful_metrics=['loss', 'val_loss'])

            for train_batch_no in range(train_steps):
                # train step
                train_loss = self.train_step(train_data)

                # update metrics
                self.loss_tracker.update_state(train_loss)
                progbar.update(train_batch_no + 1, values=[('loss', self.loss_tracker.result())],
                               finalize=False)

                # log scores
                if self.is_logging:
                    with self.train_logging.as_default(self.train_counter.numpy()):
                        tf.summary.scalar("loss", self.loss_tracker.result())

            # Validation process
            val_loss = self.evaluate(val_data, val_steps, log_val_results=True, use_main=False)

            # Update scores
            progbar.update(train_batch_no + 1, values=[
                ('loss', self.loss_tracker.result()),
                ('val_loss', val_loss)
            ], finalize=True)

            # Sampling images
            if epoch % self.freq == 0:
                self.sample(epoch, self.no_of, False)
                if self.sample_ema_only is False:
                    self.sample(epoch, self.no_of, True)

            # Capturing best weights
            if self.val_loss_tracker.result() < self.best_loss:
                self.best_loss = self.val_loss_tracker.result()
                with self.device.scope():
                    for w1, w2 in zip(self.ema_model.trainable_weights, self.best_model.trainable_weights):
                        w2.assign(w1)
                print("Best model captured....")

            # Saving all weights
            self.save_manager.save(0)

            # Garbage Collect
            gc.collect()

    # eval function using main_model or ema_model
    def evaluate(self, val_data, val_steps, log_val_results=False, use_main=False):
        """
        Evaluation loop.

        :param val_data: Validation data iterator.
        :param val_steps: number of iterations for validation data.
        :param log_val_results:  boolean value, whether to log validation results.
        :param use_main: boolean value, whether to use main model for generating.
        """
        self.val_loss_tracker.reset_states()
        use_main = tf.constant(use_main)
        log_val_results = tf.constant(log_val_results)

        for _ in range(val_steps):
            val_loss = self.test_step(val_data, use_main, log_val_results)
            self.val_loss_tracker.update_state(val_loss)

            if log_val_results and self.is_logging:
                with self.val_logging.as_default(self.val_counter.numpy()):
                    tf.summary.scalar("val_loss", self.val_loss_tracker.result())

        return self.val_loss_tracker.result()
