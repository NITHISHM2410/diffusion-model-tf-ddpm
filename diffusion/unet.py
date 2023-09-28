from forward_diffusion import *


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, embed):
        super(PositionalEmbedding, self).__init__()
        self.embed = embed

        self.emb = tf.keras.Sequential([
            tf.keras.layers.Dense(self.embed),
            tf.keras.layers.Activation("swish"),
            tf.keras.layers.Dense(self.embed)
        ])
        self.emb.build((None, self.embed))

    def call(self, t):
        embed = self.embed / 2
        rates = 1.0 / (10000 ** (tf.range(0, embed, dtype=tf.float32) / embed))
        rates = tf.expand_dims(rates, axis=0)
        t = tf.cast(t, tf.float32)
        sines = tf.sin(t * rates)
        cosines = tf.cos(t * rates)
        embeddings = tf.concat([sines, cosines], axis=-1)
        return self.emb(embeddings)


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


class AttentionBlock(tf.keras.layers.Layer):
    def __init__(self, c, hw, heads=4):
        super(AttentionBlock, self).__init__()
        self.c = c
        self.hw = hw
        self.heads = heads
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=self.heads,
                                                      key_dim=self.c,
                                                      value_dim=self.c)

        self.start_norm = tf.keras.layers.LayerNormalization()
        self.attn_norm = tf.keras.layers.LayerNormalization()
        self.end_norm = tf.keras.layers.LayerNormalization()

        self.fully = tf.keras.Sequential([
            tf.keras.layers.Dense(self.c),
            tf.keras.layers.Activation('gelu'),
            tf.keras.layers.Dense(self.c),
        ])
        self.fully.build((None, self.hw * self.hw, self.c))

    def call(self, x):
        x = tf.reshape(x, (-1, self.hw * self.hw, self.c))
        x_attn = self.start_norm(x)
        x_attn = self.mha(key=x_attn, value=x_attn, query=x_attn)
        x = x + x_attn
        x = self.attn_norm(x)
        x = self.fully(x) + x
        x = self.end_norm(x)
        return tf.reshape(x, (-1, self.hw, self.hw, self.c))


class DownBlock(tf.keras.layers.Layer):
    def __init__(self, c_in, c_out, hw, embed=256):
        super(DownBlock, self).__init__()
        self.hw = hw
        self.c_in = c_in
        self.c_out = c_out
        self.embed = embed
        self.down = tf.keras.Sequential([
            tf.keras.layers.MaxPool2D((2, 2)),
            ResBlock(self.hw // 2, self.c_in, self.c_out),
            ResBlock(self.hw // 2, self.c_out, self.c_out),
        ])
        self.down.build((None, self.hw, self.hw, self.c_in))

        self.emb_down = tf.keras.Sequential([
            tf.keras.layers.Dense(self.c_out),
            tf.keras.layers.Reshape((1, 1, self.c_out))
        ])
        self.emb_down.build((None, self.embed))

    def call(self, inputs):
        x, t = inputs
        x = self.down(x)
        x_emb = self.emb_down(t)
        x = x + x_emb
        return x


class UpBlock(tf.keras.layers.Layer):
    def __init__(self, c_in, c_out, hw, embed=256):
        super(UpBlock, self).__init__()
        self.hw = hw
        self.c_in = c_in
        self.c_out = c_out
        self.embed = embed
        self.up = tf.keras.layers.UpSampling2D((2, 2))

        self.conv_block = tf.keras.Sequential([
            ResBlock(self.hw * 2, self.c_in, self.c_out),
            ResBlock(self.hw * 2, self.c_out, self.c_out),
        ])
        self.conv_block.build((None, self.hw * 2, self.hw * 2, self.c_in))

        self.emb_up = tf.keras.Sequential([
            tf.keras.layers.Dense(self.c_out),
            tf.keras.layers.Reshape((1, 1, self.c_out))
        ])
        self.emb_up.build((None, self.embed))

    def call(self, inputs):
        x, xr, t = inputs
        x = self.up(x)
        x = tf.keras.layers.concatenate([x, xr], axis=-1)
        x = self.conv_block(x)
        x_emb = self.emb_up(t)
        x = x + x_emb
        return x


class UNet(tf.keras.Model):
    def __init__(self, time_steps=1000, num_classes=5, input_size=64):
        super(UNet, self).__init__()
        self.num_classes = num_classes
        self.time_steps = time_steps
        self.input_size = input_size
        self.start = ResBlock(self.input_size, 3, 64)

        self.down1 = DownBlock(64, 128, 64)
        self.down_sa_1 = AttentionBlock(128, 32)

        self.down2 = DownBlock(128, 256, 32)
        self.down_sa_2 = AttentionBlock(256, 16)

        self.down3 = DownBlock(256, 256, 16)
        self.down_sa_3 = AttentionBlock(256, 8)

        self.bt1 = ResBlock(8, 256, 512)
        self.bt2 = ResBlock(8, 512, 512)
        self.bt3 = ResBlock(8, 512, 256)

        self.up1 = UpBlock(512, 128, 8)
        self.up_sa_4 = AttentionBlock(128, 16)

        self.up2 = UpBlock(256, 64, 16)
        self.up_sa_5 = AttentionBlock(64, 32)

        self.up3 = UpBlock(128, 64, 32)
        self.up_sa_6 = AttentionBlock(64, 64)

        self.end = tf.keras.layers.SeparableConv2D(3, kernel_size=(1, 1))

        self.class_embedding = tf.keras.layers.Embedding(self.num_classes + 1, 256)
        self.add = tf.keras.layers.Add()
        self.reshape = tf.keras.layers.Reshape(target_shape=())

        self.time_encoding = PositionalEmbedding(256)
        self.forward_noiser = ForwardDiffusion(self.time_steps)

        self.alphas = self.forward_noiser.alphas
        self.betas = self.forward_noiser.betas
        self.alpha_hats = self.forward_noiser.alpha_hat

    def call(self, inputs):
        x, t, label = inputs

        t = self.time_encoding(t)

        label = self.reshape(label)
        label = self.class_embedding(label)

        t = self.add([label, t])

        x1 = self.start(x)

        x2 = self.down1([x1, t])
        x2 = self.down_sa_1(x2)

        x3 = self.down2([x2, t])
        x3 = self.down_sa_2(x3)

        x4 = self.down3([x3, t])
        x4 = self.down_sa_3(x4)

        x4 = self.bt1(x4)
        x4 = self.bt2(x4)
        x4 = self.bt3(x4)

        x = self.up1([x4, x3, t])
        x = self.up_sa_4(x)

        x = self.up2([x, x2, t])
        x = self.up_sa_5(x)

        x = self.up3([x, x1, t])
        x = self.up_sa_6(x)

        out = self.end(x)
        return out



