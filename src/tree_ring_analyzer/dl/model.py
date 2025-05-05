import tensorflow as tf
from tensorflow.keras import layers



class Unet:


    def __init__(self, input_size, filter_num, n_labels, stack_num_down=2, stack_num_up=2,
                 activation='relu', output_activation='softmax', attention=True, name='Unet'):
        self.model = None
        self.input_size = input_size
        self.filter_num = filter_num
        self.n_labels = n_labels
        self.stack_num_down = stack_num_down
        self.stack_num_up = stack_num_up
        self.activation = activation
        self.output_activation = output_activation
        self.name = name
        self.attention = attention

        self.p = []
        self.f = []
        self.u = []
        self._build()

  
    def stack_conv_block(self, x, n_filters, stack_num):
        for i in range(0, stack_num):
            x = layers.Conv2D(n_filters, 3, padding = "same", activation = self.activation, kernel_initializer = "he_normal")(x)

        return x


    def downsample_block(self, x, n_filters):
        f = self.stack_conv_block(x, n_filters, self.stack_num_down)
        p = layers.MaxPool2D(2)(f)
        p = layers.Dropout(0.3)(p)
        return f, p


    def attention_gate(self, g, s, num_filters):
        Wg = layers.Conv2D(num_filters, 3, padding="same")(g)
        Wg = layers.BatchNormalization()(Wg)
    
        Ws = layers.Conv2D(num_filters, 3, padding="same")(s)
        Ws = layers.BatchNormalization()(Ws)
    
        out = layers.Activation(self.activation)(Wg + Ws)
        out = layers.Conv2D(num_filters, 3, padding="same")(out)
        out = layers.Activation("sigmoid")(out)
    
        return out * s


    def upsample_block(self, x, conv_features, n_filters):
        # upsample
        x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
        if self.attention:
            s = self.attention_gate(x, conv_features, n_filters)
            # concatenate
            x = layers.concatenate([x, s])
        else:
            # concatenate
            x = layers.concatenate([x, conv_features])
        
        # dropout
        x = layers.Dropout(0.3)(x)
        # Conv2D twice with ReLU activation
        x = self.stack_conv_block(x, n_filters, self.stack_num_up)
        return x


    def _build(self):
        inputs = layers.Input(shape=self.input_size)

        # encoder: contracting path - downsample
        self.p.append(inputs)
        for i in range(0, len(self.filter_num) - 1):
            f, p = self.downsample_block(self.p[i], self.filter_num[i])
            self.p.append(p)
            self.f.append(f)

        bottleneck = self.stack_conv_block(self.p[-1], self.filter_num[-1], self.stack_num_down)

        # decoder: expanding path - upsample
        self.u.append(bottleneck)
        for i in range(0, len(self.filter_num) - 1):
            u = self.upsample_block(self.u[i], self.f[- (i + 1)], self.filter_num[- (i + 2)])
            self.u.append(u)
        
        outputs = layers.Conv2D(self.n_labels, (1,1), padding="same", activation = self.output_activation)(self.u[-1])

        # unet model with Keras Functional API
        self.model = tf.keras.Model(inputs, outputs, name=self.name)

