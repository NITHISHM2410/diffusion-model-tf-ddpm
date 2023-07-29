import tensorflow as tf


class ResBlock(tf.keras.layers.Layer):
    def __init__(self, hw, c_in, c_out):
        super(ResBlock, self).__init__()
        self.hw = hw
        self.c_in = c_in
        self.c_out = c_out

        self.double_conv = tf.keras.Sequential([
            tf.keras.layers.SeparableConv2D(self.c_out, kernel_size=3, padding='same'),
            tf.keras.layers.Activation("gelu"),
            tf.keras.layers.GroupNormalization(),
            tf.keras.layers.SeparableConv2D(self.c_out, kernel_size=3, padding='same'),
            tf.keras.layers.Activation("gelu"),
            tf.keras.layers.GroupNormalization(),
        ])
        self.double_conv.build((None, self.hw, self.hw, self.c_in))

        if self.c_in != self.c_out:
            self.proj = tf.keras.Sequential([
                tf.keras.layers.SeparableConv2D(self.c_out, kernel_size=3, padding='same'),
                tf.keras.layers.Activation("gelu"),
                tf.keras.layers.GroupNormalization()
            ])
            self.proj.build((None, self.hw, self.hw, self.c_in))

    def call(self, x):
        if self.c_in == self.c_out:
            return x + self.double_conv(x)
        else:
            return self.proj(x) + self.double_conv(x)
