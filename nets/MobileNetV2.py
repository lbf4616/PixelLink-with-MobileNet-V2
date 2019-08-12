import tensorflow as tf
import conv_blocks as ops

slim = tf.contrib.slim

expand_input = ops.expand_input_by_factor

def basenet(inputs, fatness = 32, dilation = True):
    """
    backbone net of MobileNetV2
    """
    # End_points collect relevant activations for external use.
    end_points = {}
    # Original VGG-16 blocks.
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d], 
    padding='SAME', activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm):
         net = slim.conv2d(inputs, 32, [3, 3], stride=2)
         net = ops.expanded_conv(net, expansion_size=expand_input(1, divisible_by=1), num_outputs=16, stride=1, normalizer_fn=slim.batch_norm)
         end_points['conv1'] = net
         print(net)
         net = ops.expanded_conv(net, expansion_size=expand_input(6), num_outputs=24, stride=2, normalizer_fn=slim.batch_norm)
         net = ops.expanded_conv(net, expansion_size=expand_input(6), num_outputs=24, stride=1, normalizer_fn=slim.batch_norm)
         end_points['conv2'] = net
         print(net)
         net = ops.expanded_conv(net, expansion_size=expand_input(6), num_outputs=32, stride=2, normalizer_fn=slim.batch_norm)
         net = ops.expanded_conv(net, expansion_size=expand_input(6), num_outputs=32, stride=1, normalizer_fn=slim.batch_norm)
         net = ops.expanded_conv(net, expansion_size=expand_input(6), num_outputs=32, stride=1, normalizer_fn=slim.batch_norm)
         end_points['conv3'] = net
         print(net)
         net = ops.expanded_conv(net, expansion_size=expand_input(6), num_outputs=64, stride=2, normalizer_fn=slim.batch_norm)
         net = ops.expanded_conv(net, expansion_size=expand_input(6), num_outputs=64, stride=1, normalizer_fn=slim.batch_norm)
         net = ops.expanded_conv(net, expansion_size=expand_input(6), num_outputs=64, stride=1, normalizer_fn=slim.batch_norm)
         net = ops.expanded_conv(net, expansion_size=expand_input(6), num_outputs=64, stride=1, normalizer_fn=slim.batch_norm)
         net = ops.expanded_conv(net, expansion_size=expand_input(6), num_outputs=96, stride=1, normalizer_fn=slim.batch_norm)
         net = ops.expanded_conv(net, expansion_size=expand_input(6), num_outputs=96, stride=1, normalizer_fn=slim.batch_norm)
         net = ops.expanded_conv(net, expansion_size=expand_input(6), num_outputs=96, stride=1, normalizer_fn=slim.batch_norm)
         end_points['conv4'] = net
         print(net)
         net = ops.expanded_conv(net, expansion_size=expand_input(6), num_outputs=160, stride=2, normalizer_fn=slim.batch_norm)
         net = ops.expanded_conv(net, expansion_size=expand_input(6), num_outputs=160, stride=1, normalizer_fn=slim.batch_norm)
         net = ops.expanded_conv(net, expansion_size=expand_input(6), num_outputs=160, stride=1, normalizer_fn=slim.batch_norm)
         net = ops.expanded_conv(net, expansion_size=expand_input(6), num_outputs=320, stride=1, normalizer_fn=slim.batch_norm)
         net = slim.conv2d(net, 1280, [1, 1], stride=1)
         end_points['fc5'] = net
         print(net)








        # # Block1
        # net = slim.repeat(inputs, 2, slim.conv2d, fatness, [3, 3], scope='conv1')
        # end_points['conv1_2'] = net
        # net = slim.max_pool2d(net, [2, 2], scope='pool1')
        # end_points['pool1'] = net
        
        
        # # Block 2.
        # net = slim.repeat(net, 2, slim.conv2d, fatness * 2, [3, 3], scope='conv2')
        # end_points['conv2_2'] = net
        # net = slim.max_pool2d(net, [2, 2], scope='pool2')
        # end_points['pool2'] = net
        
        
        # # Block 3.
        # net = slim.repeat(net, 3, slim.conv2d, fatness * 4, [3, 3], scope='conv3')
        # end_points['conv3_3'] = net
        # net = slim.max_pool2d(net, [2, 2], scope='pool3')
        # end_points['pool3'] = net
        
        # # Block 4.
        # net = slim.repeat(net, 3, slim.conv2d, fatness * 8, [3, 3], scope='conv4')
        # end_points['conv4_3'] = net
        # net = slim.max_pool2d(net, [2, 2], scope='pool4')
        # end_points['pool4'] = net
        
        # # Block 5.
        # net = slim.repeat(net, 3, slim.conv2d, fatness * 8, [3, 3], scope='conv5')
        # end_points['conv5_3'] = net
        # net = slim.max_pool2d(net, [3, 3], 1, scope='pool5')
        # end_points['pool5'] = net

        # # fc6 as conv, dilation is added
        # if dilation:
        #     net = slim.conv2d(net, fatness * 16, [3, 3], rate=6, scope='fc6')
        # else:
        #     net = slim.conv2d(net, fatness * 16, [3, 3], scope='fc6')
        # end_points['fc6'] = net

        # # fc7 as conv
        # net = slim.conv2d(net, fatness * 16, [1, 1], scope='fc7')
        # end_points['fc7'] = net

    return net, end_points;    

