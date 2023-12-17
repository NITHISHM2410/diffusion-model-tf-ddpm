import tensorflow as tf


class ForwardDiffusion:
    def __init__(self, time_steps, beta_start, beta_end):
        """
        Forward diffusion phase - Q(xt | xt-1)

        :param time_steps:  diffusion time steps count
        :param beta_start: variance schedule start
        :param beta_end: variance schedule end
        """
        self.time_steps = time_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.betas = tf.linspace(self.beta_start, self.beta_end, int(self.time_steps))
        self.alphas = 1. - self.betas
        self.alpha_hats = tf.math.cumprod(self.alphas)

    def __call__(self, inputs):
        x, t = inputs
        noise = tf.random.normal(shape=tf.shape(x))

        sqrt_alpha_hat = tf.math.sqrt(
            tf.gather(self.alpha_hats, t)
        )[:, None, None, None]

        sqrt_one_minus_alpha_hat = tf.math.sqrt(
            1. - tf.gather(self.alpha_hats, t)
        )[:, None, None, None]

        noised_image = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise
        return noised_image, noise

    def get_forward_diffusion_params(self):
        return self.alphas, self.betas, self.alpha_hats


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, embed):
        """
        Positional embedding layer for embedding time

        :param embed: embedding dim
        """
        super(PositionalEmbedding, self).__init__()
        self.embed = embed

        self.emb_layer = tf.keras.Sequential([
            tf.keras.layers.Dense(self.embed * 4),
            tf.keras.layers.Activation("swish"),
            tf.keras.layers.Dense(self.embed * 4)
        ])
        self.emb_layer.build((None, self.embed))

    def call(self, t, **kwargs):
        half_embed = self.embed // 2
        emb = tf.math.log(10000.0) / (half_embed - 1)
        emb = tf.exp(tf.range(half_embed, dtype=tf.float32) * -emb)
        emb = tf.cast(t, tf.float32) * emb[None, :]
        emb = tf.concat([
            tf.sin(emb),
            tf.cos(emb)
        ], axis=-1)
        return self.emb_layer(emb)


class UpSample(tf.keras.layers.Layer):
    def __init__(self, c_out, hw, with_conv):
        """
        Up sampling layer

        :param c_out: expected output channels for this layer's outputs
        :param hw: height, width of input to this layer
        :param with_conv: whether to use conv layer along with up sampling
        """
        super(UpSample, self).__init__()
        self.hw = hw
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = tf.keras.layers.SeparableConv2D(
                filters=c_out,
                kernel_size=3,
                padding='same'
            )

    def call(self, x, **kwargs):
        x = tf.image.resize(x, size=(self.hw * 2, self.hw * 2),
                            method='nearest')
        if self.with_conv:
            x = self.conv(x)
        return x


class DownSample(tf.keras.layers.Layer):
    def __init__(self, c_out, hw, with_conv):
        """
        Down sampling layer

        :param c_out: expected output channels for this layer's outputs
        :param hw: height, width of input to this layer
        :param with_conv: whether to use conv layer for down sampling, 'False' equals pooling.
        """
        super(DownSample, self).__init__()
        self.hw = hw
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = tf.keras.layers.SeparableConv2D(
                filters=c_out, kernel_size=3,
                strides=2, padding='same'
            )
        else:
            self.pool = tf.keras.layers.AvgPool2D(
                pool_size=(2, 2),
                padding='same'
            )

    def call(self, x, **kwargs):
        if self.with_conv:
            x = self.conv(x)
        else:
            x = self.pool(x)
        return x


class ResBlock(tf.keras.layers.Layer):
    def __init__(self, c_in, c_out, dropout, t_emb):
        """
        Resnet block which implements basic convolutions and embeds time embedding to the input.

        :param c_in: input channels of this layer's inputs
        :param c_out: expected output channels for this layer's outputs
        :param dropout: dropout value
        :param t_emb: embedding dimension of time embedding
        """
        super(ResBlock, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.t_emb = t_emb

        self.norm1 = tf.keras.layers.GroupNormalization()
        self.non_linear1 = tf.keras.layers.Activation("swish")
        self.conv1 = tf.keras.layers.SeparableConv2D(
            filters=self.c_out,
            kernel_size=3,
            padding='same'
        )

        if self.t_emb is not None:
            self.time_emb = tf.keras.Sequential([
                tf.keras.layers.Activation("swish"),
                tf.keras.layers.Dense(self.c_out),
                tf.keras.layers.Reshape((1, 1, self.c_out))
            ])
            self.time_emb.build((None, t_emb))

        self.norm2 = tf.keras.layers.GroupNormalization()
        self.non_linear2 = tf.keras.layers.Activation("swish")
        self.dropout_layer = tf.keras.layers.Dropout(dropout)
        self.conv2 = tf.keras.layers.SeparableConv2D(
            filters=self.c_out,
            kernel_size=3,
            padding='same'
        )

        if self.c_in != self.c_out:
            self.conv_p = tf.keras.layers.Conv2D(
                filters=self.c_out,
                kernel_size=1,
            )

    def call(self, x, **kwargs):
        h, t = x
        hr = h

        h = self.non_linear1(self.norm1(h))
        h = self.conv1(h)

        if self.t_emb is not None:
            h += self.time_emb(t)

        h = self.non_linear2(self.norm2(h))
        if kwargs['training']:
            h = self.dropout_layer(h)
        h = self.conv2(h)

        if self.c_in != self.c_out:
            hr = self.conv_p(hr)

        return h + hr


class AttentionBlock(tf.keras.layers.Layer):
    def __init__(self, c, hw):
        """
        A single head attention block

        :param c: input channels of this layer's inputs
        :param hw: height, width of input to this layer
        """
        super(AttentionBlock, self).__init__()
        self.c = c
        self.hw = hw
        self.attn = tf.keras.layers.Attention(use_scale=True)
        self.norm = tf.keras.layers.GroupNormalization()

        self.qkv_proj = tf.keras.layers.Conv1D(kernel_size=1, filters=c * 3)
        self.final_proj = tf.keras.layers.Conv1D(kernel_size=1, filters=c)

    def call(self, x, **kwargs):
        h = x
        h = self.norm(h)
        h = tf.reshape(h, (-1, self.hw ** 2, self.c))
        h = tf.split(self.qkv_proj(h), num_or_size_splits=3, axis=-1)
        h = self.attn(h)
        h = self.final_proj(h)
        h = tf.reshape(h, (-1, self.hw, self.hw, self.c))
        return x + h


class AttentionUnitLayer(tf.keras.layers.Layer):
    def __init__(self, c, hw):
        """
        Sub block of multi head attention

        :param c: input channels of this layer's inputs
        :param hw: height, width of input to this layer
        """
        super(AttentionUnitLayer, self).__init__()
        self.c = c
        self.hw = hw
        self.attn = tf.keras.layers.Attention(use_scale=True)
        self.norm = tf.keras.layers.GroupNormalization()
        self.qkv_proj = tf.keras.layers.Conv1D(kernel_size=1, filters=c * 3)

    def call(self, x, **kwargs):
        x = self.norm(x)
        x = tf.split(self.qkv_proj(x), num_or_size_splits=3, axis=-1)
        x = self.attn(x)
        return x


class MHAAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, c, heads, hw):
        """
        A Multi head attention layer

        :param c: input channels of this layer's inputs
        :param heads: number of attention heads
        :param hw: height, width of input to this layer
        """
        super(MHAAttentionBlock, self).__init__()
        self.c = c
        self.hw = hw
        self.heads = heads
        self.attn_heads_units = [AttentionUnitLayer(self.c // self.heads, self.hw) for _ in range(self.heads)]
        self.final_proj = tf.keras.layers.Conv1D(kernel_size=1, filters=self.c)

    def call(self, x, **kwargs):
        h = x
        h = tf.reshape(h, (-1, self.hw ** 2, self.c))
        parts = tf.split(h, num_or_size_splits=self.heads, axis=-1)

        for h_i in range(self.heads):
            parts[h_i] = self.attn_heads_units[h_i](parts[h_i])

        h = tf.concat(parts, axis=-1)
        h = self.final_proj(h)
        h = tf.reshape(h, (-1, self.hw, self.hw, self.c))
        return h + x


class Encoder(tf.keras.Model):
    def __init__(self, c_in=3, c_out=512, ch_list=(128, 128, 256, 256, 512, 512), attn_res=(16,),
                 heads=-1, cph=32, mid_attn=True, resamp_with_conv=True, num_res_blocks=2, img_res=256, dropout=0):
        """
        An Image encoder.

        :param c_in: input channels of this model's inputs
        :param c_out: output channels of this model's outputs
        :param ch_list: list of channels to be used across down & up sampling
        :param attn_res: list of resolution for which attention mechanism is to be implemented
        :param heads: number of attention heads
        :param cph: channels per heads, used when 'heads' is set to -1
        :param mid_attn: boolean value whether to use attention in bottleneck layer
        :param resamp_with_conv: boolean value whether to use conv layer during up and down sampling
        :param num_res_blocks: number of resnet blocks per channel in 'ch_list'
        :param img_res: input image resolution
        :param dropout: dropout value to be used in resnet blocks
        """
        super(Encoder, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.img_res = img_res
        num_res = len(ch_list)
        cur_res = self.img_res

        # down
        self.down_layers = [
            tf.keras.layers.SeparableConv2D(
                kernel_size=3,
                filters=ch_list[0],
                padding='same'
            )]
        self.down_requires_time = [False]

        for level in range(num_res):
            block_in = ch_list[max(level - 1, 0)]
            block_out = ch_list[level]

            for block in range(num_res_blocks):
                ResAttnBlock = tf.keras.Sequential()
                ResAttnBlock.add(ResBlock(c_in=block_in, c_out=block_out,
                                          t_emb=None, dropout=dropout))
                block_in = block_out
                if cur_res in attn_res:
                    ResAttnBlock.add(MHAAttentionBlock(
                        c=block_in,
                        heads=block_in//cph if heads == -1 else heads,
                        hw=cur_res
                    ))

                self.down_layers.append(ResAttnBlock)
                self.down_requires_time.append(True)
            if level != num_res - 1:
                self.down_layers.append(DownSample(ch_list[level], cur_res, resamp_with_conv))
                self.down_requires_time.append(False)
                cur_res //= 2

        # mid
        self.mid_layers = []
        self.mid_requires_time = []

        self.mid_layers.append(ResBlock(c_in=ch_list[-1], c_out=ch_list[-1],
                                        t_emb=None, dropout=dropout))
        self.mid_requires_time.append(True)

        if mid_attn:
            self.mid_layers.append(MHAAttentionBlock(ch_list[-1], ch_list[-1]//cph if heads == -1 else heads, cur_res))
            self.mid_requires_time.append(False)

        self.mid_layers.append(ResBlock(c_in=ch_list[-1], c_out=ch_list[-1],
                                        t_emb=None, dropout=dropout))
        self.mid_requires_time.append(True)

        # end
        self.end_norm = tf.keras.layers.GroupNormalization()
        self.end_non_linear = tf.keras.layers.Activation("swish")
        self.end_conv = tf.keras.layers.Conv2D(
            kernel_size=3, padding='same', filters=self.c_out
        )

        self.build([(None, self.img_res, self.img_res, self.c_in), None])

    def call(self, inputs, training=True, **kwargs):
        x, t = inputs

        t = None

        for depth, layer in enumerate(self.down_layers):
            if self.down_requires_time[depth]:
                inputs = [x, t]
            else:
                inputs = x
            x = layer(inputs, training=training)

        for depth, layer in enumerate(self.mid_layers):
            if self.mid_requires_time[depth]:
                inputs = [x, t]
            else:
                inputs = x
            x = layer(inputs, training=training)

        x = self.end_conv(self.end_non_linear(self.end_norm(x)))
        return x


class Decoder(tf.keras.Model):
    def __init__(self, c_in=512, c_out=3, ch_list=(128, 128, 256, 256, 512, 512), attn_res=(16,),
                 heads=-1, cph=32, mid_attn=True, resamp_with_conv=True, num_res_blocks=2, img_res=256, dropout=0):
        """
        An Image Decoder.

        :param c_in: input channels of this model's inputs
        :param c_out: output channels of this model's outputs
        :param ch_list: list of channels to be used across down & up sampling
        :param attn_res: list of resolution for which attention mechanism is to be implemented
        :param heads: number of attention heads
        :param cph: channels per heads, used when 'heads' is set to -1
        :param mid_attn: boolean value whether to use attention in bottleneck layer
        :param resamp_with_conv: boolean value whether to use conv layer during up and down sampling
        :param num_res_blocks: number of resnet blocks per channel in 'ch_list'
        :param img_res: input image resolution
        :param dropout: dropout value to be used in resnet blocks
        """
        super(Decoder, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.img_res = img_res
        num_res = len(ch_list)
        cur_res = self.img_res

        # up conv in
        self.conv_in = tf.keras.layers.SeparableConv2D(
            kernel_size=3, padding='same', filters=ch_list[-1]
        )

        # mid
        self.mid_layers = []
        self.mid_requires_time = []

        self.mid_layers.append(ResBlock(c_in=ch_list[-1], c_out=ch_list[-1],
                                        t_emb=None, dropout=dropout))
        self.mid_requires_time.append(True)

        if mid_attn:
            self.mid_layers.append(MHAAttentionBlock(ch_list[-1], ch_list[-1]//cph if heads == -1 else heads, cur_res))
            self.mid_requires_time.append(False)

        self.mid_layers.append(ResBlock(c_in=ch_list[-1], c_out=ch_list[-1],
                                        t_emb=None, dropout=dropout))
        self.mid_requires_time.append(True)

        # up
        self.up_requires_time = []
        self.up_layers = []

        for level in reversed(range(num_res)):
            block_in = ch_list[min(level + 1, num_res - 1)]
            block_out = ch_list[level]

            for block in range(num_res_blocks + 1):
                ResAttnBlock = tf.keras.Sequential([])
                ResAttnBlock.add(ResBlock(c_in=block_in, c_out=block_out,
                                          dropout=dropout, t_emb=None))
                block_in = block_out
                if cur_res in attn_res:
                    ResAttnBlock.add(MHAAttentionBlock(
                        c=block_in,
                        heads=block_in//cph if heads == -1 else heads,
                        hw=cur_res
                    ))
                self.up_layers.append(ResAttnBlock)
                self.up_requires_time.append(True)
            if level != 0:
                self.up_layers.append(UpSample(ch_list[level], cur_res, resamp_with_conv))
                self.up_requires_time.append(False)
                cur_res *= 2

        # end
        self.end_norm = tf.keras.layers.GroupNormalization()
        self.end_non_linear = tf.keras.layers.Activation("swish")
        self.end_conv = tf.keras.layers.Conv2D(
            kernel_size=3, padding='same', filters=self.c_out
        )

        self.build([(None, self.img_res, self.img_res, self.c_in), None])

    def call(self, inputs, training=True, **kwargs):
        x, t = inputs

        t = None

        x = self.conv_in(x)

        for depth, layer in enumerate(self.mid_layers):
            if self.mid_requires_time[depth]:
                inputs = [x, t]
            else:
                inputs = x
            x = layer(inputs, training=training)

        for depth, layer in enumerate(self.up_layers):
            if self.up_requires_time[depth]:
                inputs = [x, t]
            else:
                inputs = x
            x = layer(inputs, training=training)

        x = self.end_conv(self.end_non_linear(self.end_norm(x)))
        return x


class UNet(tf.keras.Model):
    def __init__(self, c_in=3, c_out=3, ch_list=(128, 128, 256, 256, 512, 512), attn_res=(16,), heads=-1, cph=32,
                 mid_attn=True, resamp_with_conv=True, num_res_blocks=2, img_res=256, dropout=0, time_steps=1000,
                 beta_start=1e-4, beta_end=0.02, num_classes=1, cfg_weight=3):
        """
        An UNet model down samples and up samples and allows skip connections across both the up and down sampling.
        Also applies Forward diffusion, Positional embedding and class conditioning.

        :param c_in: input channels of this model's inputs
        :param c_out: output channels of this model's outputs
        :param ch_list: list of channels to be used across down & up sampling
        :param attn_res: list of resolution for which attention mechanism is to be implemented
        :param heads: number of attention heads
        :param cph: channels per heads, used when 'heads' is set to -1
        :param mid_attn: boolean value whether to use attention in bottleneck layer
        :param resamp_with_conv: boolean value whether to use conv layer during up and down sampling
        :param num_res_blocks: number of resnet blocks per channel in 'ch_list'
        :param img_res: input image resolution
        :param dropout: dropout value to be used in resnet blocks
        :param time_steps: number of diffusion time steps
        :param beta_start: noise variance schedule start value
        :param beta_end: noise variance schedule end value
        :param num_classes: number of classes for conditional generation
        :param cfg_weight: interpolation weight for conditional generation
        """
        super(UNet, self).__init__()
        self.c_in = c_in
        self.img_res = img_res
        num_res = len(ch_list)
        cur_res = self.img_res
        self.num_classes = num_classes
        self.cfg_weight = cfg_weight

        # down
        self.down_layers = [
            tf.keras.layers.SeparableConv2D(
                kernel_size=3,
                filters=ch_list[0],
                padding='same'
            )]
        self.down_requires_time = [False]
        self.skip_con_channels = [ch_list[0]]

        for level in range(num_res):
            block_in = ch_list[max(level - 1, 0)]
            block_out = ch_list[level]

            for block in range(num_res_blocks):
                ResAttnBlock = tf.keras.Sequential()
                ResAttnBlock.add(ResBlock(c_in=block_in, c_out=block_out,
                                          t_emb=ch_list[0] * 4, dropout=dropout))
                block_in = block_out
                if cur_res in attn_res:
                    ResAttnBlock.add(MHAAttentionBlock(
                        heads=block_in//cph if heads == -1 else heads,
                        c=block_in,
                        hw=cur_res
                    ))

                self.skip_con_channels.append(block_in)
                self.down_layers.append(ResAttnBlock)
                self.down_requires_time.append(True)
            if level != num_res - 1:
                self.down_layers.append(DownSample(ch_list[level], cur_res, resamp_with_conv))
                self.down_requires_time.append(False)
                cur_res //= 2
                self.skip_con_channels.append(ch_list[level])

        # mid
        self.mid_layers = []
        self.mid_requires_time = []

        self.mid_layers.append(ResBlock(c_in=ch_list[-1], c_out=ch_list[-1],
                                        t_emb=ch_list[0] * 4, dropout=dropout))
        self.mid_requires_time.append(True)

        if mid_attn:
            self.mid_layers.append(MHAAttentionBlock(ch_list[-1], ch_list[-1]//cph if heads == -1 else heads, cur_res))
            self.mid_requires_time.append(False)

        self.mid_layers.append(ResBlock(c_in=ch_list[-1], c_out=ch_list[-1],
                                        t_emb=ch_list[0] * 4, dropout=dropout))
        self.mid_requires_time.append(True)

        # up
        self.up_layers = []
        self.up_requires_time = []

        for level in reversed(range(num_res)):
            block_in = ch_list[min(level + 1, num_res - 1)]
            block_out = ch_list[level]

            for block in range(num_res_blocks + 1):
                ResAttnBlock = tf.keras.Sequential([])
                ResAttnBlock.add(ResBlock(c_in=block_in + self.skip_con_channels.pop(), c_out=block_out,
                                          dropout=dropout, t_emb=ch_list[0] * 4))
                block_in = block_out
                if cur_res in attn_res:
                    ResAttnBlock.add(MHAAttentionBlock(
                        c=block_in,
                        heads=block_in//cph if heads == -1 else heads,
                        hw=cur_res
                    ))

                self.up_layers.append(ResAttnBlock)
                self.up_requires_time.append(True)
            if level != 0:
                self.up_layers.append(UpSample(ch_list[level], cur_res, resamp_with_conv))
                self.up_requires_time.append(False)
                cur_res *= 2

        # final
        self.exit_layers = tf.keras.Sequential([
            tf.keras.layers.GroupNormalization(),
            tf.keras.layers.Activation("swish"),
            tf.keras.layers.SeparableConv2D(filters=c_out, kernel_size=3,
                                            padding='same')
        ])

        # other
        self.pos_encoding = PositionalEmbedding(embed=ch_list[0])

        self.forward_diff = ForwardDiffusion(time_steps, beta_start=beta_start, beta_end=beta_end)
        self.alphas, self.betas, self.alpha_hats = self.forward_diff.get_forward_diffusion_params()

        if self.num_classes > 1:
            self.cls_encoding = tf.keras.layers.Embedding(input_dim=self.num_classes + 1,
                                                          output_dim=ch_list[0] * 4)
            self.flatten = tf.keras.layers.Reshape(target_shape=())

        # build
        self.build([(None, self.img_res, self.img_res, self.c_in), (None, 1), (None, 1)])

    def call(self, inputs, training=None, **kwargs):
        x, t, c = inputs

        t = self.pos_encoding(t)
        if self.num_classes > 1:
            t += self.cls_encoding(self.flatten(c))

        skip_cons = []
        for depth, layer in enumerate(self.down_layers):
            if self.down_requires_time[depth]:
                inputs = [x, t]
            else:
                inputs = x
            x = layer(inputs, training=training)
            skip_cons.append(x)

        for depth, layer in enumerate(self.mid_layers):
            if self.mid_requires_time[depth]:
                inputs = [x, t]
            else:
                inputs = x
            x = layer(inputs, training=training)

        for depth, layer in enumerate(self.up_layers):
            if self.up_requires_time[depth]:
                inputs = [tf.concat([x, skip_cons.pop()], axis=-1), t]
            else:
                inputs = x
            x = layer(inputs, training=training)

        x = self.exit_layers(x)
        return x

    @tf.function
    def diffuse_step(self, images, time, cls):
        """
        Single reverse diffusion step - P(xt | xt-1)

        :param images: input images
        :param time: current diffusion time step
        :param cls: label value for class conditional generation
        :return:
        """
        batch = tf.shape(images)[0]
        time = tf.repeat(time, repeats=batch, axis=0)

        alpha = tf.gather(self.alphas, time)[:, None, None, None]
        beta = tf.gather(self.betas, time)[:, None, None, None]
        alpha_hat = tf.gather(self.alpha_hats, time)[:, None, None, None]

        time = tf.expand_dims(time, axis=-1)

        if cls is None:
            predicted_noise = self([images, time, cls], training=False)
        else:
            cls_cond = tf.reshape(cls, (batch, 1))
            cls_uncond = tf.fill(dims=tf.shape(cls_cond), value=0)

            predicted_noise_uncond = self([images, time, cls_uncond], training=False)
            predicted_noise_cond = self([images, time, cls_cond], training=False)

            predicted_noise = (predicted_noise_uncond +
                               self.cfg_weight * (predicted_noise_cond - predicted_noise_uncond))

        if time[0] > 0:
            noise = tf.random.normal(shape=tf.shape(images))
        else:
            noise = tf.zeros_like(images)

        images = (1 / tf.sqrt(alpha)) * (
                images - ((1 - alpha) / (tf.sqrt(1 - alpha_hat))) * predicted_noise) + tf.sqrt(
            beta) * noise
        return images


