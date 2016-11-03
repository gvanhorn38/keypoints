import numpy as np
import tensorflow as tf

from inputs import distorted_shifted_bounding_box, distort_color, apply_with_random_selector, build_heatmaps, extract_crop, two_d_gaussian, flip_parts_left_right, build_heatmaps_etc

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
        'image/object/bbox/count' : tf.FixedLenFeature([], tf.int64),
        'image/object/parts/x' : tf.VarLenFeature(dtype=tf.float32), # x coord for all parts and all objects
        'image/object/parts/y' : tf.VarLenFeature(dtype=tf.float32), # y coord for all parts and all objects
        'image/object/parts/v' : tf.VarLenFeature(dtype=tf.int64),   # part visibility for all parts and all objects
        'image/object/area' : tf.VarLenFeature(dtype=tf.float32), # the area of the object, based on segmentation mask or bounding box mask
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
    
    num_bboxes = tf.cast(features['image/object/bbox/count'], tf.int32)
    no_bboxes = tf.equal(num_bboxes, 0)

    parts_x = tf.expand_dims(features['image/object/parts/x'].values, 0)
    parts_y = tf.expand_dims(features['image/object/parts/y'].values, 0)
    parts_v = tf.cast(tf.expand_dims(features['image/object/parts/v'].values, 0), tf.int32)
    
    #part_visibilities = tf.cast(features['image/object/parts/v'], tf.int32)
    #part_visibilities = tf.reshape(tf.sparse_tensor_to_dense(part_visibilities), tf.pack([num_bboxes, num_parts]))

    areas = features['image/object/area'].values
    areas = tf.reshape(areas, [num_bboxes])


    # Add a summary of the original data
    if add_summaries:
      bboxes_to_draw = tf.cond(no_bboxes, lambda:  tf.constant([[0, 0, 1, 1]], tf.float32), lambda: tf.transpose(tf.concat(0, [ymin, xmin, ymax, xmax]), [1, 0]))
      bboxes_to_draw = tf.reshape(bboxes_to_draw, [1, -1, 4])
      image_with_bboxes = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0), bboxes_to_draw)
      tf.image_summary('original_image', image_with_bboxes)
    
    # GVH: We need to ensure that the perturbed bbox still contains the parts...
    # Perturb the bounding box coordinates
    r = tf.random_uniform([], minval=0, maxval=1, dtype=tf.float32)
    do_perturb = tf.logical_and(tf.less(r, cfg.DO_RANDOM_BBOX_SHIFT), tf.greater(num_bboxes, 0))
    xmin, ymin, xmax, ymax = tf.cond(do_perturb, 
      lambda: distorted_shifted_bounding_box(xmin, ymin, xmax, ymax, num_bboxes, image_height, image_width, cfg.RANDOM_BBOX_SHIFT_EXTENT), 
      lambda: tf.tuple([xmin, ymin, xmax, ymax])
    )

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
    part_visibilities = tf.reshape(parts_v, tf.pack([num_bboxes, num_parts]))

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
      bboxes_to_draw = tf.cond(no_bboxes, lambda:  tf.constant([[0, 0, 1, 1]], tf.float32), lambda: tf.transpose(tf.concat(0, [ymin, xmin, ymax, xmax]), [1, 0]))
      bboxes_to_draw = tf.reshape(bboxes_to_draw, [1, -1, 4])
      image_with_bboxes = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0), bboxes_to_draw)
      tf.image_summary('flipped_distorted_image', image_with_bboxes)
    
    # Create the crops, the bounding boxes, the parts and heatmaps
    bboxes = tf.concat(0, [xmin, ymin, xmax, ymax])
    bboxes = tf.transpose(bboxes, [1, 0])
    parts = tf.concat(0, [parts_x, parts_y])
    parts = tf.transpose(parts, [1, 0])
    parts = tf.reshape(parts, [-1, num_parts * 2])
    
    image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
    params = [image, bboxes, parts, part_visibilities, cfg.PARTS.SIGMAS, areas, cfg.INPUT_SIZE, cfg.HEATMAP_SIZE, False, 0, cfg.PARTS.LEFT_RIGHT_PAIRS]
    cropped_images, heatmaps, parts, background_heatmaps = tf.py_func(build_heatmaps_etc, params, [tf.uint8, tf.float32, tf.float32, tf.float32]) 
    cropped_images = tf.image.convert_image_dtype(cropped_images, dtype=tf.float32)

    # Add a summary of the final crops
    if add_summaries:
      tf.image_summary('cropped_images', cropped_images)
    
    # Get the images in the range [-1, 1]
    cropped_images = tf.sub(cropped_images, 0.5)
    cropped_images = tf.mul(cropped_images, 2.0)

    # Set the shape of everything for the queue
    cropped_images.set_shape([None, cfg.INPUT_SIZE, cfg.INPUT_SIZE, 3])
    image_ids = tf.tile([[image_id]], [num_bboxes, 1])
    image_ids.set_shape([None, 1])
    
    heatmaps.set_shape([None, cfg.HEATMAP_SIZE, cfg.HEATMAP_SIZE, num_parts])
    bboxes.set_shape([None, 4])
    parts.set_shape([None, num_parts * 2])
    part_visibilities.set_shape([None, num_parts]) 
    background_heatmaps.set_shape([None, cfg.HEATMAP_SIZE, cfg.HEATMAP_SIZE, num_parts])

    if shuffle_batch:
      batched_images, batched_heatmaps, batched_parts, batched_part_visibilities, batched_image_ids, batched_background_heatmaps = tf.train.shuffle_batch(
        [cropped_images, heatmaps, parts, part_visibilities, image_ids, background_heatmaps],
        batch_size=batch_size,
        num_threads=num_threads,
        capacity= capacity, #batch_size * (num_threads + 2),
        # Ensures a minimum amount of shuffling of examples.
        min_after_dequeue= min_after_dequeue, # 3 * batch_size,
        seed = cfg.RANDOM_SEED,
        enqueue_many=True,
        name="shuffle_batch_queue"
      )

    else:
      batched_images, batched_heatmaps, batched_parts, batched_part_visibilities, batched_image_ids, batched_background_heatmaps = tf.train.batch(
        [cropped_images, heatmaps, parts, part_visibilities, image_ids, background_heatmaps],
        batch_size=batch_size,
        num_threads=num_threads,
        capacity= capacity, #batch_size * (num_threads + 2),
        enqueue_many=True
      )

  # return a batch of images and their labels
  return batched_images, batched_heatmaps, batched_parts, batched_part_visibilities, batched_image_ids, batched_background_heatmaps

