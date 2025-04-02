import tensorflow as tf
from tensorflow.keras import layers



class AttentionUnet:


    def __init__(self, filter=64, activation='linear'):
        self.model = None
        self.filter = filter
        self.activation = activation
        self._build()


    @classmethod    
    def double_conv_block(cls, x, n_filters):
        # Conv2D then ReLU activation
        x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
        # Conv2D then ReLU activation
        x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
        return x


    @classmethod    
    def downsample_block(cls, x, n_filters):
        f = AttentionUnet.double_conv_block(x, n_filters)
        p = layers.MaxPool2D(2)(f)
        p = layers.Dropout(0.3)(p)
        return f, p


    @classmethod
    def attention_gate(cls, g, s, num_filters):
        Wg = layers.Conv2D(num_filters, 3, padding="same")(g)
        Wg = layers.BatchNormalization()(Wg)
    
        Ws = layers.Conv2D(num_filters, 3, padding="same")(s)
        Ws = layers.BatchNormalization()(Ws)
    
        out = layers.Activation("relu")(Wg + Ws)
        out = layers.Conv2D(num_filters, 3, padding="same")(out)
        out = layers.Activation("sigmoid")(out)
    
        return out * s


    @classmethod  
    def upsample_block(cls, x, conv_features, n_filters):
        # upsample
        x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
        s = AttentionUnet.attention_gate(x, conv_features, n_filters)
        # concatenate
        x = layers.concatenate([x, s])
        # dropout
        x = layers.Dropout(0.3)(x)
        # Conv2D twice with ReLU activation
        x = AttentionUnet.double_conv_block(x, n_filters)
        return x


    @classmethod
    def double_conv_block(cls, x, n_filters):
        # Conv2D then ReLU activation
        x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
        # Conv2D then ReLU activation
        x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
        return x


    @classmethod
    def downsample_block(cls, x, n_filters):
        f = AttentionUnet.double_conv_block(x, n_filters)
        p = layers.MaxPool2D(2)(f)
        p = layers.Dropout(0.3)(p)
        return f, p


    @classmethod
    def attention_gate(cls, g, s, num_filters):
        Wg = layers.Conv2D(num_filters, 3, padding="same")(g)
        Wg = layers.BatchNormalization()(Wg)
    
        Ws = layers.Conv2D(num_filters, 3, padding="same")(s)
        Ws = layers.BatchNormalization()(Ws)
    
        out = layers.Activation("relu")(Wg + Ws)
        out = layers.Conv2D(num_filters, 3, padding="same")(out)
        out = layers.Activation("sigmoid")(out)
    
        return out * s
    

    @classmethod
    def upsample_block(cls, x, conv_features, n_filters):
        # upsample
        x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
        s = AttentionUnet.attention_gate(x, conv_features, n_filters)
        # concatenate
        x = layers.concatenate([x, s])
        # dropout
        x = layers.Dropout(0.3)(x)
        # Conv2D twice with ReLU activation
        x = AttentionUnet.double_conv_block(x, n_filters)
        return x


    def _build(self):
        inputs = layers.Input(shape=(256,256,3))
        # encoder: contracting path - downsample
        # 1 - downsample
        f1, p1 = AttentionUnet.downsample_block(inputs, self.filter)
        # 2 - downsample
        f2, p2 = AttentionUnet.downsample_block(p1, self.filter * 2)

        # 3 - downsample
        f3, p3 = AttentionUnet.downsample_block(p2, self.filter * 4)
        # 4 - downsample
        f4, p4 = AttentionUnet.downsample_block(p3, self.filter * 8)
        # 5 - bottleneck
        bottleneck = AttentionUnet.double_conv_block(p4, self.filter * 16)
        # decoder: expanding path - upsample
        # 6 - upsample
        u6 = AttentionUnet.upsample_block(bottleneck, f4, self.filter * 8)
        # 7 - upsample
        u7 = AttentionUnet.upsample_block(u6, f3, self.filter * 4)
        # 8 - upsample
        u8 = AttentionUnet.upsample_block(u7, f2, self.filter * 2)
        # 9 - upsample
        u9 = AttentionUnet.upsample_block(u8, f1, self.filter)
        # outputs
        outputs = layers.Conv2D(1, (1,1), padding="same", activation = self.activation)(u9)
        # unet model with Keras Functional API
        self.model = tf.keras.Model(inputs, outputs, name="U-Net")