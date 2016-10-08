import numpy as np
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

def extract_crop(image, bbox, pad_percentage=0.25):
  """ Extract a bounding box crop from the image.
  Args:
    image : float32 image
    bbox : bbox in image coordinates
  Returns:
    np.array : The cropped region of the image
    np.array : The new upper left hand coordinate (x, y). This can be used to offset part locations.
  """

  image_height, image_width = image.shape[:2]

  cropped_images = []
  adjusted_keypoints = []

  x1, y1, x2, y2 = bbox
  w = x2 - x1
  h = y2 - y1

  center_x = int(np.round(x1 + w / 2.))
  center_y = int(np.round(y1 + h / 2.))

  if w > h:

    pad = np.round(pad_percentage * w / 2.)

    new_x1 = x1 - pad
    new_x2 = x2 + pad
    new_w = np.round(new_x2 - new_x1)
    new_h = new_w
    new_y1 = center_y - new_h / 2.
    new_y2 = center_y + new_h / 2.

  else:

    pad = np.round(pad_percentage * h / 2.)

    new_y1 = y1 - pad
    new_y2 = y2 + pad
    new_h = np.round(new_y2 - new_y1)
    new_w = new_h
    new_x1 = center_x - new_w / 2.
    new_x2 = center_x + new_w / 2.
  
  new_x1 = int(np.round(new_x1))
  new_x2 = int(np.round(new_x2))
  new_y1 = int(np.round(new_y1))
  new_y2 = int(np.round(new_y2))

  new_w = int(np.round(new_x2 - new_x1))
  new_h = int(np.round(new_y2 - new_y1))

  cropped_bbox = np.zeros([new_h, new_w, 3])

  cropped_idx_x1 = 0 if new_x1 >= 0 else np.abs(new_x1)
  cropped_idx_x2 = new_w if new_x2 <= image_width else new_w - (new_x2 - image_width)
  cropped_idx_y1 = 0 if new_y1 >= 0 else np.abs(new_y1)
  cropped_idx_y2 = new_h if new_y2 <= image_height else new_h - (new_y2 - image_height)

  image_idx_x1 = max(0, new_x1)
  image_idx_x2 = min(image_width, new_x2)
  image_idx_y1 = max(0, new_y1)
  image_idx_y2 = min(image_height, new_y2)

  cropped_bbox[cropped_idx_y1:cropped_idx_y2,cropped_idx_x1:cropped_idx_x2] = image[image_idx_y1:image_idx_y2, image_idx_x1:image_idx_x2]
  
  cropped_bbox = cropped_bbox.astype(np.float32)
  upper_left_x_y = np.array([new_x1, new_y1]).astype(np.float32)

  return [cropped_bbox, upper_left_x_y]

def build_heatmaps(parts, part_visibilities, area, part_sigmas,
  image, input_size=256, heatmap_size=64):
  """
  parts : np.array in image space
  part_visibilities : np.array 
  area : float, a measure of the size of the object. Could be bounding box area or segmentation area.
  part_sigmas : np.array, the std for the individual parts
  image_shape : the shape of the image that the parts are currently in
  input_size : the size of the input image to the network
  heatmap_size : the size of the heatmap to generate
  """

  image_shape = image.shape[:2]
  num_parts= parts.shape[0] / 2
  part_corr = 0

  # The heatmaps we will return
  preped_heat_maps = np.zeros((heatmap_size, heatmap_size, num_parts), dtype=np.float32)

  # these are the ground truth part locations (not normalized; these will be used by the testing and validation code)
  preped_part_locations = np.zeros(num_parts * 2, dtype=np.float32)
  
  image_height, image_width = image_shape[:2]

  # Determing the scaling from the original image size to the input image size
  im_scale = 1.
  if image_height > image_width:
    new_height = input_size
    height_factor = float(1.0)
    width_factor = new_height / float(image_height)
    new_width = int(np.round(image_width * width_factor))
    im_scale = width_factor
  else:
    new_width = input_size
    width_factor = float(1.0)
    height_factor = new_width / float(image_width)
    new_height = int(np.round(image_height * height_factor))
    im_scale = height_factor

  # Determine the scaling from the input image size to the heatmap size
  heat_map_to_target_ratio = float(heatmap_size) / input_size

  # we want to zero out the parts of the heat map that don't correspond to valid
  # image coordinates (the images are not squashed).
  #heat_map_edge_x = min(heatmap_size, int(np.ceil(image_width * heat_map_to_target_ratio)))
  #heat_map_edge_y = min(heatmap_size, int(np.ceil(image_height * heat_map_to_target_ratio)))

  
  # Compute the heat maps and add them to the heat map blob
  for j in range(num_parts):
    ind = j * 2
    x, y = parts[ind:ind+2]
    v = part_visibilities[j]

    if v > 0:
      # scale the part locations
      scaled_x = x * im_scale * heat_map_to_target_ratio
      scaled_y = y * im_scale * heat_map_to_target_ratio

      sigma_x = im_scale * heat_map_to_target_ratio * np.sqrt(area) * 2. * part_sigmas[j]
      sigma_y = sigma_x 
      heat_map = two_d_gaussian(scaled_x, scaled_y, sigma_x, sigma_y, part_corr, (heatmap_size, heatmap_size))

      # Zero out the non-valid image coordinates
      #heat_map[:, heat_map_edge_x:] = 0
      #heat_map[heat_map_edge_y:, :] = 0

      # Axis order: (batch elem, channel, height, width)
      preped_heat_maps[:, :, j] = heat_map
      preped_part_locations[ind] = scaled_x
      preped_part_locations[ind+1] = scaled_y        
    else:
      # the heat map blob is prefilled with zeros, so we are good to go.
      pass
  
  preped_heat_maps = preped_heat_maps.astype(np.float32)
  preped_part_locations = preped_part_locations.astype(np.float32)

  return [preped_heat_maps, preped_part_locations]


def apply_with_random_selector(x, func, num_cases):
  """Computes func(x, sel), with sel sampled from [0...num_cases-1].
  Args:
    x: input Tensor.
    func: Python function to apply.
    num_cases: Python int32, number of cases to sample sel from.
  Returns:
    The result of func(x, sel), where func receives the value of the
    selector as a python integer, but sel is sampled dynamically.
  """
  sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
  # Pass the real x only to one of the func calls.
  return control_flow_ops.merge([
      func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
      for case in range(num_cases)])[0]


def distort_color(image, color_ordering=0, fast_mode=True, scope=None):
  """Distort the color of a Tensor image.
  Each color distortion is non-commutative and thus ordering of the color ops
  matters. Ideally we would randomly permute the ordering of the color ops.
  Rather then adding that level of complication, we select a distinct ordering
  of color ops for each preprocessing thread.
  Args:
    image: 3-D Tensor containing single image in [0, 1].
    color_ordering: Python int, a type of distortion (valid values: 0-3).
    fast_mode: Avoids slower ops (random_hue and random_contrast)
    scope: Optional scope for name_scope.
  Returns:
    3-D Tensor color-distorted image on range [0, 1]
  Raises:
    ValueError: if color_ordering not in [0, 3]
  """
  with tf.name_scope(scope, 'distort_color', [image]):
    if fast_mode:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      else:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    else:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
      elif color_ordering == 2:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      elif color_ordering == 3:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
      else:
        raise ValueError('color_ordering must be in [0, 3]')

    # The random_* ops do not necessarily clamp.
    return tf.clip_by_value(image, 0.0, 1.0)


def distorted_shifted_bounding_box(xmin, ymin, xmax, ymax, num_bboxes, image_height, image_width, max_num_pixels_to_shift = 5):
  """ Distort the bounding box coordinates by a given maximum amount. 
  """


  image_width = tf.cast(image_width, tf.float32)
  image_height = tf.cast(image_height, tf.float32)
  one_pixel_width = 1. / image_width
  one_pixel_height = 1. / image_height
  max_width_shift = one_pixel_width * max_num_pixels_to_shift
  max_height_shift = one_pixel_height * max_num_pixels_to_shift
  
  xmin -= tf.random_uniform([1, num_bboxes], minval=0, maxval=max_width_shift, dtype=tf.float32)
  xmax += tf.random_uniform([1, num_bboxes], minval=0, maxval=max_width_shift, dtype=tf.float32)
  ymin -= tf.random_uniform([1, num_bboxes], minval=0, maxval=max_height_shift, dtype=tf.float32)
  ymax += tf.random_uniform([1, num_bboxes], minval=0, maxval=max_height_shift, dtype=tf.float32)
  
  # ensure that the coordinates are still valid
  ymin = tf.clip_by_value(ymin, 0.0, 1.)
  xmin = tf.clip_by_value(xmin, 0.0, 1.)
  ymax = tf.clip_by_value(ymax, 0.0, 1.)
  xmax = tf.clip_by_value(xmax, 0.0, 1.)
  
  return xmin, ymin, xmax, ymax


def two_d_gaussian(center_x, center_y, sigma_x, sigma_y, corr, shape):
  """
  Generate a 2d gaussian image.
  This can be optimized using vector notation rather than for loops....

  corr is the correlation between the two axes.
  shape is a tuple of (height, width)
  """

  output = np.empty(shape, dtype=np.float32)

  A = 1. / (2. * np.pi * sigma_x * sigma_y * np.sqrt(1. - corr ** 2))
  A = 1 # GVH : Get rid of the normalization for now
  B = -1. / (2. * (1. - corr**2))

  sigma_x_sq = sigma_x ** 2
  sigma_y_sq = sigma_y ** 2

  for y in range(shape[0]):
    for x in range(shape[1]):
      C1 = ((x - center_x) ** 2) / sigma_x_sq
      C2 = ((y - center_y) ** 2) / sigma_y_sq
      C3 = 2. * corr * (x - center_x) * (y - center_y) / (sigma_x * sigma_y)
      output[y,x] = A * np.exp(B * (C1 + C2 - C3))

  return output

def flip_parts_left_right(parts_x, parts_y, parts_v, left_right_pairs, num_parts):
  """Flip the parts horizontally. The parts are in normalized coordinates
  """
  
  flipped_parts = np.vstack([np.squeeze(parts_x), np.squeeze(parts_y), np.squeeze(parts_v)]).transpose([1, 0])
  flipped_parts[:,0] = 1. - flipped_parts[:,0]
  
  num_instances = flipped_parts.shape[0] / num_parts

  for i in range(num_instances):
    for left_idx, right_idx in left_right_pairs:
      l = i * num_parts + left_idx
      r = i * num_parts + right_idx
      x,y,v = flipped_parts[l]
      flipped_parts[l] = flipped_parts[r][:]
      flipped_parts[r] = [x,y,v]
      
  flipped_parts = flipped_parts.astype(np.float32)

  flipped_x = np.expand_dims(flipped_parts[:,0].ravel(), 0)
  flipped_y = np.expand_dims(flipped_parts[:,1].ravel(), 0)
  flipped_v = np.expand_dims(flipped_parts[:,2].ravel().astype(np.int32), 0)
  
  return [flipped_x, flipped_y, flipped_v]

def flip_heatmaps_left_right(heatmaps, left_right_pairs):
  heatmaps = np.fliplr(heatmaps)
  for left_idx, right_idx in left_right_pairs:
    l = np.copy(heatmaps[:,:,left_idx])
    heatmaps[:,:,left_idx] = heatmaps[:,:,right_idx]
    heatmaps[:,:,right_idx] = l[:,:]
  return heatmaps.astype(np.float32)
