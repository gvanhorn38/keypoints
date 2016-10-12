"""
File for detecting parts on images without ground truth.
"""
import argparse
import json
from matplotlib import pyplot as plt
import numpy as np
import os
import pprint
from scipy import interpolate
from scipy.misc import imresize
import sys
import tensorflow as tf
from tensorflow.contrib import slim
import time

from config import parse_config_file
from detect import get_local_maxima
import detect_inputs as inputs
import model

def detect(tfrecords, checkpoint_path, save_dir, max_iterations, iterations_per_record, cfg):

  tf.logging.set_verbosity(tf.logging.DEBUG)

  graph = tf.Graph()
  
  with graph.as_default():
    
    batched_images, batched_bboxes, batched_scores, batched_image_ids, batched_labels, batched_image_height_widths, batched_upper_left_x_y = inputs.input_nodes(
      tfrecords=tfrecords,
      num_epochs=1,
      batch_size=cfg.BATCH_SIZE,
      num_threads=cfg.NUM_INPUT_THREADS,
      capacity = cfg.QUEUE_CAPACITY,
      cfg=cfg
    )
    
    batch_norm_params = {
        'decay': cfg.BATCHNORM_MOVING_AVERAGE_DECAY,
        'epsilon': 0.001,
        'variables_collections' : [tf.GraphKeys.MOVING_AVERAGE_VARIABLES],
        'is_training' : False
    }
    # Set activation_fn and parameters for batch_norm.
    with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params,
                        weights_regularizer=slim.l2_regularizer(0.00004),
                        biases_regularizer=slim.l2_regularizer(0.00004)) as scope:
      
      predicted_heatmaps = model.build(
        input = batched_images, 
        num_parts = cfg.PARTS.NUM_PARTS
      )
    
    ema = tf.train.ExponentialMovingAverage(
      decay=cfg.MOVING_AVERAGE_DECAY
    )   
    shadow_vars = {
      ema.average_name(var) : var
      for var in slim.get_model_variables()
    }

    saver = tf.train.Saver(shadow_vars, reshape=True)
    
    fetches = [batched_images, predicted_heatmaps[-1], batched_bboxes, batched_scores, batched_image_ids, batched_labels, batched_image_height_widths, batched_upper_left_x_y]

    # Now create a training coordinator that will control the different threads
    coord = tf.train.Coordinator()
    
    sess_config = tf.ConfigProto(
      log_device_placement=False,
      #device_filters = device_filters,
      allow_soft_placement = True,
      gpu_options = tf.GPUOptions(
          per_process_gpu_memory_fraction=cfg.SESSION_CONFIG.PER_PROCESS_GPU_MEMORY_FRACTION
      )
    )
    session = tf.Session(graph=graph, config=sess_config)
    
    with session.as_default():

      # make sure to initialize all of the variables
      tf.initialize_all_variables().run()
      tf.initialize_local_variables().run()
      
      # launch the queue runner threads
      threads = tf.train.start_queue_runners(sess=session, coord=coord)

      results = []
      #output_writer = None
      
      try:
        
        if tf.gfile.IsDirectory(checkpoint_path):
          checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
        
        if checkpoint_path is None:
          print "ERROR: No checkpoint file found."
          return

        # Restores from checkpoint
        saver.restore(session, checkpoint_path)
        # Assuming model_checkpoint_path looks something like:
        #   /my-favorite-path/cifar10_train/model.ckpt-0,
        # extract global_step from it.
        global_step = int(checkpoint_path.split('/')[-1].split('-')[-1])
        print "Found model for global step: %d" % (global_step,)
        
        # we will store results into a tfrecord file
        output_writer_iteration = 0
        #output_path = os.path.join(save_dir, 'heatmap_results-%d-%d.tfrecords' % (global_step, output_writer_iteration))
        #output_writer = tf.python_io.TFRecordWriter(output_path)
        
        plt.ion()

        image_to_convert = tf.placeholder(tf.float32)
        convert_to_uint8 = tf.image.convert_image_dtype(tf.add(tf.div(image_to_convert, 2.0), 0.5), tf.uint8)
        
        image_to_resize = tf.placeholder(tf.float32)
        resize_to_input_size = tf.image.resize_bilinear(image_to_resize, size=[cfg.INPUT_SIZE, cfg.INPUT_SIZE])

        num_part_cols = 3
        num_part_rows = int(np.ceil(cfg.PARTS.NUM_PARTS / (num_part_cols * 1.)))

        step = 0
        print_str = ', '.join([
          'Step: %d',
          'Time/image network (ms): %.1f',
          'Time/image post proc (ms): %.1f'
        ])
        while not coord.should_stop():
         
          outputs = session.run(fetches)
          
          for b in range(cfg.BATCH_SIZE):

            fig = plt.figure("heat maps")
            plt.clf() 
    
            image = outputs[0][b]
            heatmaps = outputs[1][b]
            bbox = outputs[2][b]
            score = outputs[3][b]
            image_id = outputs[4][b]
            
            int8_image = session.run(convert_to_uint8, {image_to_convert : image})

            heatmaps = np.clip(heatmaps, 0., 1.)
            heatmaps = np.expand_dims(heatmaps, 0)
            resized_heatmaps = session.run(resize_to_input_size, {image_to_resize : heatmaps})
            resized_heatmaps = np.squeeze(resized_heatmaps)

            for j in range(cfg.PARTS.NUM_PARTS):
            
              heatmap = resized_heatmaps[:,:,j]
              
              fig.add_subplot(num_part_rows, num_part_cols, j+1)
              plt.imshow(int8_image)

              # rescale the values of the heatmap 
              f = interpolate.interp1d([0., 1.], [0, 255])
              int_heatmap = f(heatmap).astype(np.uint8)

              # Add the heatmap as an alpha channel over the image
              blank_image = np.zeros(image.shape, np.uint8)
              blank_image[:] = [255, 0, 0]
              heat_map_alpha = np.dstack((blank_image, int_heatmap))
              plt.imshow(heat_map_alpha)
              plt.axis('off')
              plt.title(cfg.PARTS.NAMES[j])
              
              # Render the argmax point
              x, y = np.array(np.unravel_index(np.argmax(heatmap), heatmap.shape)[::-1])
              plt.plot(x, y, color=cfg.PARTS.COLORS[j], marker=cfg.PARTS.SYMBOLS[j], label=cfg.PARTS.NAMES[j])
              
              print "%s : max %0.3f, min %0.3f" % (cfg.PARTS.NAMES[j], np.max(heatmap), np.min(heatmap))
                
            plt.show()
            #plt.pause(0.0001)
            r = raw_input("Push button")
            if r != "":
              done = True
              break
          
          
          
      except Exception as e:
        # Report exceptions to the coordinator.
        coord.request_stop(e)
      
      # When done, ask the threads to stop. It is innocuous to request stop twice.
      coord.request_stop()
      # And wait for them to actually do it.
      coord.join(threads)
      


def parse_args():

    parser = argparse.ArgumentParser(description='Test an Inception V3 network')

    parser.add_argument('--tfrecords', dest='tfrecords',
                        help='paths to tfrecords files', type=str,
                        nargs='+', required=True)

    parser.add_argument('--checkpoint_path', dest='checkpoint_path',
                          help='path to directory where the checkpoint files are stored. The latest model will be tested against.', type=str,
                          required=False, default=None)
                          
    parser.add_argument('--save_dir', dest='save_dir',
                        help='Path to the directory where the results will be saved',
                        required=True, type=str)
                        
    parser.add_argument('--config', dest='config_file',
                        help='Path to the configuration file',
                        required=True, type=str)
    
    parser.add_argument('--max_iterations', dest='max_iterations',
                        help='Maximum number of iterations to run. Set to 0 to run on all records.',
                        required=False, type=int, default=0)
    
    parser.add_argument('--iterations_per_record', dest='iterations_per_record',
                        help='The number of iterations to store in a tfrecord file before creating another one.',
                        required=False, type=int, default=1000)

    args = parser.parse_args()
    
    return args

if __name__ == '__main__':

    args = parse_args()
    cfg = parse_config_file(args.config_file)

    print "Configurations:"
    print pprint.pprint(cfg)

    detect(
      tfrecords=args.tfrecords,
      checkpoint_path=args.checkpoint_path,
      save_dir = args.save_dir,
      max_iterations = args.max_iterations,
      iterations_per_record = args.iterations_per_record,
      cfg=cfg
    )