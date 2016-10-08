"""
The heatmaps have been precomputed and saved in the tfrecord file.
"""

import numpy as np
import tensorflow as tf

from inputs import distort_color, apply_with_random_selector, flip_parts_left_right, flip_heatmaps_left_right

def input_nodes(
  
  tfrecords, 

  num_parts,

  # number of times to read the tfrecords
  num_epochs=None,

  # Data queue feeding the model
  batch_size=32,
  num_threads=2,
  shuffle_batch = True,
  capacity = 1000,
  min_after_dequeue = 96,
  
  # Tensorboard Summaries
  add_summaries = True,

  # Global configuration
  cfg=None):

  with tf.name_scope('inputs'):

    # A producer to generate tfrecord file paths
    filename_queue = tf.train.string_input_producer(
      tfrecords,
      num_epochs=num_epochs,
      shuffle=shuffle_batch
    )

    # Construct a Reader to read examples from the tfrecords file
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    # Parse an Example to access the Features
    features = tf.parse_single_example(
      serialized_example,
      features = {
        'image/id' : tf.FixedLenFeature([], tf.string),
        'image/encoded'  : tf.FixedLenFeature([], tf.string),
        'image/height' : tf.FixedLenFeature([], tf.int64),
        'image/width' : tf.FixedLenFeature([], tf.int64),
        'image/object/bbox/xmin' : tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin' : tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax' : tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax' : tf.VarLenFeature(dtype=tf.float32),
        'image/object/parts/x' : tf.VarLenFeature(dtype=tf.float32), # x coord for all parts and all objects
        'image/object/parts/y' : tf.VarLenFeature(dtype=tf.float32), # y coord for all parts and all objects
        'image/object/parts/v' : tf.VarLenFeature(dtype=tf.int64),   # part visibility for all parts and all objects
        'image/object/parts/heatmaps' : tf.FixedLenFeature([cfg.HEATMAP_SIZE * cfg.HEATMAP_SIZE * num_parts], dtype=tf.float32)
      }
    )

    # Read in a jpeg image
    image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
    
     # Convert the pixel values to be in the range [0,1]
    if image.dtype != tf.float32:
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    image_height = tf.cast(features['image/height'], tf.float32)
    image_width = tf.cast(features['image/width'], tf.float32)
    
    image_id = features['image/id']

    xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
    ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
    xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
    ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)
    
    parts_x = tf.expand_dims(features['image/object/parts/x'].values, 0)
    parts_y = tf.expand_dims(features['image/object/parts/y'].values, 0)
    parts_v = tf.cast(tf.expand_dims(features['image/object/parts/v'].values, 0), tf.int32)

    heatmaps = features['image/object/parts/heatmaps']
    heatmaps = tf.reshape(heatmaps, [cfg.HEATMAP_SIZE, cfg.HEATMAP_SIZE, num_parts])    

    # Add a summary of the original data
    if add_summaries:
      bboxes_to_draw = tf.transpose(tf.concat(0, [ymin, xmin, ymax, xmax]), [1, 0])
      bboxes_to_draw = tf.reshape(bboxes_to_draw, [1, -1, 4])
      image_with_bboxes = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0), bboxes_to_draw)
      tf.image_summary('original_image', image_with_bboxes)

    # Randomly flip the image:
    if cfg.DO_RANDOM_FLIP_LEFT_RIGHT:
      r = tf.random_uniform([], minval=0, maxval=1, dtype=tf.float32)
      do_flip = tf.less(r, 0.5)
      image = tf.cond(do_flip, lambda: tf.image.flip_left_right(image), lambda: tf.identity(image))
      xmin, xmax = tf.cond(do_flip, lambda: tf.tuple([1. - xmax, 1. - xmin]), lambda: tf.tuple([xmin, xmax]))
      parts_x, parts_y, parts_v = tf.cond(do_flip, 
        lambda: tf.py_func(flip_parts_left_right, [parts_x, parts_y, parts_v, cfg.PARTS.LEFT_RIGHT_PAIRS, num_parts], [tf.float32, tf.float32, tf.int32]), 
        lambda: tf.tuple([parts_x, parts_y, parts_v])
      )
      heatmaps = tf.cond(do_flip, 
        lambda: tf.py_func(flip_heatmaps_left_right, [heatmaps, cfg.PARTS.LEFT_RIGHT_PAIRS], [tf.float32]), 
        lambda: tf.identity(heatmaps)
      )
      

    parts = tf.reshape(tf.transpose(tf.concat(0, [parts_x, parts_y])), [-1])
    part_visibilities = tf.reshape(parts_v, [-1])# tf.reshape(parts_v, tf.pack([num_bboxes, num_parts]))

    # Distort the colors
    r = tf.random_uniform([], minval=0, maxval=1, dtype=tf.float32)
    do_color_distortion = tf.less(r, cfg.DO_COLOR_DISTORTION)
    num_color_cases = 1 if cfg.COLOR_DISTORT_FAST else 4
    distorted_image = apply_with_random_selector(
      image,
      lambda x, ordering: distort_color(x, ordering, fast_mode=cfg.COLOR_DISTORT_FAST),
      num_cases=num_color_cases)
    image = tf.cond(do_color_distortion, lambda: tf.identity(distorted_image), lambda: tf.identity(image))
    image.set_shape([cfg.INPUT_SIZE, cfg.INPUT_SIZE, 3])

    # Add a summary
    if add_summaries:
      bboxes_to_draw = tf.transpose(tf.concat(0, [ymin, xmin, ymax, xmax]), [1, 0])
      bboxes_to_draw = tf.reshape(bboxes_to_draw, [1, -1, 4])
      image_with_bboxes = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0), bboxes_to_draw)
      tf.image_summary('flipped_distorted_image', image_with_bboxes)

    image = tf.sub(image, 0.5)
    image = tf.mul(image, 2.0)

    # Set the shape of everything for the queue
    image.set_shape([cfg.INPUT_SIZE, cfg.INPUT_SIZE, 3])
    image_ids = [image_id]
    
    heatmaps.set_shape([cfg.HEATMAP_SIZE, cfg.HEATMAP_SIZE, num_parts])

    bboxes = tf.concat(0, [xmin, ymin, xmax, ymax])
    bboxes = tf.reshape(tf.transpose(bboxes, [1, 0]), [4])
    bboxes.set_shape([4])
    
    parts.set_shape([num_parts * 2])
    part_visibilities.set_shape([num_parts]) 

    if shuffle_batch:
      batched_images, batched_heatmaps, batched_parts, batched_part_visibilities, batched_image_ids = tf.train.shuffle_batch(
        [image, heatmaps, parts, part_visibilities, image_ids],
        batch_size=batch_size,
        num_threads=num_threads,
        capacity= capacity, #batch_size * (num_threads + 2),
        # Ensures a minimum amount of shuffling of examples.
        min_after_dequeue= min_after_dequeue, # 3 * batch_size,
        seed = cfg.RANDOM_SEED,
        enqueue_many=False,
        name="shuffle_batch_queue"
      )

    else:
      batched_images, batched_heatmaps, batched_parts, batched_part_visibilities, batched_image_ids = tf.train.batch(
        [image, heatmaps, parts, part_visibilities, image_ids],
        batch_size=batch_size,
        num_threads=num_threads,
        capacity= capacity, #batch_size * (num_threads + 2),
        enqueue_many=False
      )

  # return a batch of images and their labels
  return batched_images, batched_heatmaps, batched_parts, batched_part_visibilities, batched_image_ids

