import tensorflow as tf


class forward_diffusion_helper:
    def __init__(self, time_steps, beta_start=1e-4, beta_end=0.02):
        self.time_steps = tf.cast(time_steps, tf.float32)
        self.s = 0.008
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.betas = self.compute_betas()
        self.alphas = 1. - self.betas
        self.alpha_hat = tf.math.cumprod(self.alphas)

    def compute_betas(self):
        return tf.linspace(self.beta_start, self.beta_end, int(self.time_steps))


class ForwardDiffusion:
    def __init__(self, time_steps):
        self.time_steps = time_steps
        self.params = forward_diffusion_helper(self.time_steps)
        self.alpha_hat = self.params.alpha_hat
        self.alphas = self.params.alphas
        self.betas = self.params.betas

    def __call__(self, inputs):
        x, t = inputs
        noise = tf.random.normal(shape=tf.shape(x))

        sqrt_alpha_hat = tf.math.sqrt(
            tf.gather(self.alpha_hat, t)
        )[:, None, None, None]

        sqrt_one_minus_alpha_hat = tf.math.sqrt(
            1. - tf.gather(self.alpha_hat, t)
        )[:, None, None, None]

        noised_image = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise
        return noised_image, noise
