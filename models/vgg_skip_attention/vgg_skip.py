from kaffe.tensorflow import Network


class VGG16_skip(Network):
    def setup(self):
        (self.feed('data')
         .conv(3, 3, 64, 1, 1, name='conv1_1', trainable=False)
         .conv(3, 3, 64, 1, 1, name='conv1_2', trainable=False)
         .max_pool(2, 2, 2, 2, name='pool1')
         .conv(3, 3, 128, 1, 1, name='conv2_1', trainable=False)
         .conv(3, 3, 128, 1, 1, name='conv2_2', trainable=False)
         .max_pool(2, 2, 2, 2, name='pool2')
         .conv(3, 3, 256, 1, 1, name='conv3_1', trainable=False)
         .conv(3, 3, 256, 1, 1, name='conv3_2', trainable=False)
         .conv(3, 3, 256, 1, 1, name='conv3_3', trainable=False)
         .max_pool(2, 2, 2, 2, name='pool3')
         .conv(3, 3, 512, 1, 1, name='conv4_1', trainable=False)
         .conv(3, 3, 512, 1, 1, name='conv4_2', trainable=False)
         .conv(3, 3, 512, 1, 1, name='conv4_3', trainable=False)
         .max_pool(2, 2, 2, 2, name='pool4')
         .conv(3, 3, 512, 1, 1, name='conv5_1', trainable=False)
         .conv(3, 3, 512, 1, 1, name='conv5_2', trainable=False)
         .conv(3, 3, 512, 1, 1, name='conv5_3', trainable=False)
         .max_pool(2, 2, 2, 2, name='pool5')
         .skip_conv(['pool3', 'pool4'],1000, trainable=False)
         .conv(1, 1, 512, 1, 1, name='skip_dim_reduction', trainable=False)

         .fc(4096, name='fc6', trainable=False)
         .fc(4096, name='fc7', trainable = False))
