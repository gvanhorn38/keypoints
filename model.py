import tensorflow as tf
slim = tf.contrib.slim

def residual(input, input_channels, output_channels, scope=None, reuse=None):

  with tf.variable_scope(scope, "residual", [input], reuse=reuse):
    with slim.arg_scope([slim.conv2d], stride=1):
      with tf.variable_scope("convolution_path"):
        conv = slim.conv2d(input, output_channels / 2, [1, 1], padding='VALID')
        conv1 = slim.conv2d(conv, output_channels / 2, [3, 3], padding='SAME')
        conv2 = slim.conv2d(conv1, output_channels, [1, 1], padding='VALID', activation_fn=None)
      with tf.variable_scope("skip_path"):
        if input_channels == output_channels:
          skip = input
        else:
          skip = slim.conv2d(input, output_channels, [1, 1], padding='VALID', activation_fn=None)
    res = conv2 + skip
    return res

def hourglass(input, num_branches, input_channels, output_channels, scope=None, reuse=None):
  
  with tf.variable_scope(scope, "hourglass", [input], reuse=reuse):
    with tf.variable_scope("upper_branch"):
      up1 = residual(input, input_channels, 256)
      up2 = residual(up1, 256, 256)
      up3 = residual(up2, 256, output_channels)
    with tf.variable_scope("lower_branch"):
      pool = slim.max_pool2d(input, 2, stride=2, padding='VALID')
      low1 = residual(pool, input_channels, 256)
      low2 = residual(low1, 256, 256)
      low3 = residual(low2, 256, 256)

      if num_branches > 1:
        low4 = hourglass(low3, num_branches-1, 256, output_channels)
      else:
        low4 = residual(low3, 256, output_channels)
      
      low5 = residual(low4, output_channels, output_channels)
      low5_shape = low5.get_shape().as_list()
      low5_height = low5_shape[1]
      low5_width = low5_shape[2]
      up4 = tf.image.resize_nearest_neighbor(images=low5, size=[low5_height * 2, low5_width * 2], align_corners=False)

    return up3 + up4

def _hourglass(input, num_branches, input_channels, output_channels, num_res_modules=1, scope=None, reuse=None):
  
  with tf.variable_scope(scope, "hourglass", [input], reuse=reuse):
    
    # Add the residual modules for the upper branch
    with tf.variable_scope("upper_branch"):
      up1 = input
      for i in range(num_res_modules):
        up1 = residual(up1, input_channels, input_channels)

    # Add the modules for the lower branch
    # 1. Pool -> Residuals -> Hourglass -> Residuals -> Upsample
    # 2. Pool -> Residuals -> Residuals -> Residuals -> Upsample
    with tf.variable_scope("lower_branch"):
      low1 = slim.max_pool2d(input, 2, stride=2, padding='VALID')
      for i in range(num_res_modules):
        low1 = residual(low1, input_channels, input_channels)
      
      # Are we recursing? 
      if num_branches > 1:
        low2 = hourglass(low1, num_branches-1, input_channels, input_channels, num_res_modules, scope, reuse)
      else:
        low2 = low1
        for i in range(num_res_modules):
          low2 = residual(low2, input_channels, input_channels)
    
      low3 = low2
      for i in range(num_res_modules):
        low3 = residual(low3, input_channels, input_channels)
      
      low3_shape = low3.get_shape().as_list()
      low3_height = low3_shape[1]
      low3_width = low3_shape[2]
      up2 = tf.image.resize_nearest_neighbor(images=low3, size=[low3_height * 2, low3_width * 2], align_corners=False)

    return up1 + up2

def build(input, num_parts, reuse=None, scope='HourGlass'):

  with tf.variable_scope(scope, 'HourGlass', [input], reuse=reuse):
    # Initial processing of the image
    conv = slim.conv2d(input, 64, [7,7], stride=2, padding='SAME')
    r1 = residual(conv, 64, 128)
    pool = slim.max_pool2d(r1, 2, stride=2, padding='VALID')
    r2 = residual(pool, 128, 128)
    r3 = residual(r2, 128, 128)
    r4 = residual(r3, 128, 256)

    # First hourglass
    hg1 = hourglass(r4, num_branches=4, input_channels=256, output_channels=512)

    # Linear layers to produce the first set of predictions
    with slim.arg_scope([slim.conv2d], kernel_size=[1, 1], stride=1, padding='VALID'):
      l1 = slim.conv2d(hg1, 512)
      l2 = slim.conv2d(l1, 256)
      # first set of output heatmaps
      out1 = slim.conv2d(l2, num_parts, activation_fn=None, normalizer_fn=None)
      # get the features ready to go back through the network
      out1_ = slim.conv2d(out1, 256 + 128, activation_fn=None, normalizer_fn=None)
      cat1 = tf.concat(3, [l2, pool])
      cat1_ = slim.conv2d(cat1, 256 + 128, activation_fn=None, normalizer_fn=None)
      comb1 = cat1_ + out1_
    
    # Second hourglass
    hg2 = hourglass(comb1, num_branches=4, input_channels=128 + 256, output_channels=512)

    # Produce the second set of heatmap predictions
    with slim.arg_scope([slim.conv2d], kernel_size=[1, 1], stride=1, padding='VALID'):
      l3 = slim.conv2d(hg2, 512)
      l4 = slim.conv2d(l3, 512)
      out2 = slim.conv2d(l4, num_parts, activation_fn=None, normalizer_fn=None)
    
    return [out1, out2]

def _build(input, num_parts, num_features=256, num_stacks=8, num_res_modules=1, reuse=None, scope='HourGlass'):

  with tf.variable_scope(scope, 'StackedHourGlassNetwork', [input], reuse=reuse):
    
    # Initial processing of the image
    conv = slim.conv2d(input, 64, [7,7], stride=2, padding='SAME')
    r1 = residual(conv, 64, 128)
    pool = slim.max_pool2d(r1, 2, stride=2, padding='VALID')
    r2 = residual(pool, 128, 128)
    r3 = residual(r2, 128, num_features)
    
    intermediate_features = r3

    heatmaps = []
    for i in range(num_stacks):

      # Build the hourglass
      hg = hourglass(intermediate_features, num_branches=4, input_channels=num_features, output_channels=num_features)
      
      # Residual layers at the output resolution
      ll = hg
      for j in range(num_res_modules):
        ll = residual(ll, num_features, num_features)
      
      with slim.arg_scope([slim.conv2d], kernel_size=[1, 1], stride=1, padding='VALID'):
        
        # Linear layers to produce the first set of predictions
        ll = slim.conv2d(ll, num_features)
        
        # Predicted heatmaps
        heatmap = slim.conv2d(ll, num_parts, activation_fn=None, normalizer_fn=None)
        heatmaps.append(heatmap)

        # Add the predictions back
        if i < num_stacks - 1:
          ll_ = slim.conv2d(ll, num_features, activation_fn=None, normalizer_fn=None)
          heatmap_ = slim.conv2d(heatmap, num_features, activation_fn=None, normalizer_fn=None)
          intermediate_features = intermediate_features + ll_ + heatmap_
    
  return heatmaps