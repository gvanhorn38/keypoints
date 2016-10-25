"""
Visualize the training inputs to the network. 
"""

import argparse
import logging
from matplotlib import pyplot as plt
import numpy as np
from scipy import interpolate
import tensorflow as tf

from config import parse_config_file
import train_inputs
import train_inputs_precomputed

def create_solid_rgb_image(shape, color):
  image = np.zeros(shape, np.uint8)
  image[:] = color
  return image

def visualize(tfrecords, cfg, precomputed=False):
  
  logger = logging.getLogger()
  logger.setLevel(logging.DEBUG)

  graph = tf.Graph()
  sess = tf.Session(graph = graph)
  
  num_parts = cfg.PARTS.NUM_PARTS

  # run a session to look at the images...
  with sess.as_default(), graph.as_default():

    if precomputed:
      input_nodes = train_inputs_precomputed.input_nodes
    else:
      input_nodes = train_inputs.input_nodes

    # Input Nodes
    batched_images, batched_heatmaps, batched_parts, batched_part_visibilities, batched_image_ids = input_nodes(
      tfrecords,
      num_parts,
      num_epochs=None,
      batch_size=cfg.BATCH_SIZE,
      num_threads=1,
      shuffle_batch = False,
      capacity = 10,
      min_after_dequeue = 0,
      add_summaries = True,
      cfg=cfg
    )

    image_to_convert = tf.placeholder(tf.float32)
    convert_to_uint8 = tf.image.convert_image_dtype(tf.add(tf.div(image_to_convert, 2.0), 0.5), tf.uint8)

    image_to_resize = tf.placeholder(tf.float32)
    resize_to_input_size = tf.image.resize_bilinear(image_to_resize, size=[cfg.INPUT_SIZE, cfg.INPUT_SIZE])

    coord = tf.train.Coordinator()
    tf.initialize_all_variables().run()
    tf.initialize_local_variables().run()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    num_part_cols = 3
    num_part_rows = int(np.ceil(num_parts / (num_part_cols * 1.)))

    plt.ion()
    r = ""
    while r == "":
      outputs = sess.run([batched_images, batched_heatmaps, batched_parts, batched_part_visibilities, batched_image_ids])
      
      for b in range(cfg.BATCH_SIZE):

        image_figure = plt.figure("Image")
        image_figure.clear()

        image = outputs[0][b]
        uint8_image = sess.run(convert_to_uint8, {image_to_convert : image})
        plt.imshow(uint8_image)
        
        parts = outputs[2][b]
        part_visibilities = outputs[3][b]

        for p in range(num_parts):
          if part_visibilities[p] > 0:
            idx = 2*p
            x, y = parts[idx:idx+2] * float(cfg.INPUT_SIZE) 
            plt.plot(x, y, color=cfg.PARTS.COLORS[p], marker=cfg.PARTS.SYMBOLS[p], label=cfg.PARTS.NAMES[p])
        
        heatmaps_figure = plt.figure("Heatmaps")
        heatmaps_figure.clear()

        heatmaps = outputs[1][b]
        for p in range(num_parts):
          heatmap = heatmaps[:,:,p]
          print "%s : max %0.3f, min %0.3f" % (cfg.PARTS.NAMES[p], np.max(heatmap), np.min(heatmap))

        heatmaps = np.clip(heatmaps, 0., 1.)
        heatmaps = np.expand_dims(heatmaps, 0)
        resized_heatmaps = sess.run(resize_to_input_size, {image_to_resize : heatmaps})
        resized_heatmaps = np.squeeze(resized_heatmaps)

        for p in range(num_parts):
          
          heatmap = resized_heatmaps[:,:,p]

          heatmaps_figure.add_subplot(num_part_rows, num_part_cols, p+1)
          plt.imshow(uint8_image)
 
          # rescale the values of the heatmap 
          f = interpolate.interp1d([np.min(heatmap), np.max(heatmap)], [0, 255])
          int_heatmap = f(heatmap).astype(np.uint8)

          # Add the heatmap as an alpha channel over the image
          blank_image = create_solid_rgb_image(image.shape, [255, 0, 0])
          heat_map_alpha = np.dstack((blank_image, int_heatmap))
          plt.imshow(heat_map_alpha)
          plt.axis('off')
          plt.title(cfg.PARTS.NAMES[p])

        plt.show()
        r = raw_input("push button")
        plt.clf()
        if r != "":
          break


def parse_args():

    parser = argparse.ArgumentParser(description='Visualize the inputs to the multibox detection system.')

    parser.add_argument('--tfrecords', dest='tfrecords',
                        help='paths to tfrecords files that contain the training data', type=str,
                        nargs='+', required=True)

    parser.add_argument('--config', dest='config_file',
                        help='Path to the configuration file',
                        required=True, type=str)
    
    parser.add_argument('--precomputed', dest='precomputed',
                        help='If True, then we assume that the tfrecords contain the precomputed heatmaps.',
                        action='store_true', default=False)

    args = parser.parse_args()
    return args

def main():
  args = parse_args()
  cfg = parse_config_file(args.config_file)
  visualize(
    tfrecords=args.tfrecords,
    cfg=cfg,
    precomputed=args.precomputed
  )

  
          
if __name__ == '__main__':
  main()