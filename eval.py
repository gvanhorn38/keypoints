"""
File for detecting parts on images without ground truth.
"""
import argparse
from cStringIO import StringIO
import json
import numpy as np
import os
import pprint
import sys
import tensorflow as tf
from tensorflow.contrib import slim
import time

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from config import parse_config_file
from detect import get_local_maxima
import eval_inputs as inputs
import model

def eval(tfrecords, checkpoint_path, summary_dir, max_iterations, cfg):

  tf.logging.set_verbosity(tf.logging.DEBUG)

  graph = tf.Graph()
  
  num_parts = cfg.PARTS.NUM_PARTS

  with graph.as_default():
    
    batched_images, batched_bboxes, batched_parts, batched_part_visibilities, batched_image_ids, batched_image_height_widths, batched_crop_bboxes = inputs.input_nodes(
      tfrecords=tfrecords,
      num_parts = num_parts,
      num_epochs=1,
      batch_size=cfg.BATCH_SIZE,
      num_threads=cfg.NUM_INPUT_THREADS,
      capacity = cfg.QUEUE_CAPACITY,
      shuffle_batch=True,
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
    
    fetches = [predicted_heatmaps[-1], batched_bboxes, batched_parts, batched_part_visibilities, batched_image_ids, batched_image_height_widths, batched_crop_bboxes]

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

      dataset_image_ids = set()
      gt_annotations = []
      pred_annotations = []
      gt_annotation_id = 1
      gt_image_id = 1
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
        
        step = 0
        print_str = ', '.join([
          'Step: %d',
          'Time/image network (ms): %.1f'
        ])
        while not coord.should_stop():
          t = time.time()
          outputs = session.run(fetches)
          dt = time.time() - t

          for b in range(cfg.BATCH_SIZE):

            heatmaps = outputs[0][b]
            bbox = outputs[1][b]
            parts = outputs[2][b]
            part_visibilities = outputs[3][b]
            image_id = outputs[4][b]
            image_height_widths = outputs[5][b]
            crop_bboxes = outputs[6][b]
            
            
            #heatmaps = np.clip(heatmaps, 0., 1.)
            
             # We need to transform the keypoints back to the original image space.
            image_height, image_width = image_height_widths
            crop_x1, crop_y1, crop_x2, crop_y2 = crop_bboxes 
            crop_w, crop_h = np.array([crop_x2 - crop_x1, crop_y2 - crop_y1]) * np.array([image_width, image_height], dtype=np.float32)

            restrict_to_bbox=True
            if restrict_to_bbox:
              # Crop out the portion of the heatmap that corresponds to the bounding box of the object

              bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox

              heatmap_bbox_x1 = int(np.round((bbox_x1 - crop_x1) * ( image_width / crop_w ) * cfg.HEATMAP_SIZE ))
              heatmap_bbox_y1 = int(np.round((bbox_y1 - crop_y1) * ( image_height / crop_h) * cfg.HEATMAP_SIZE ))
              heatmap_bbox_x2 = int(np.round((bbox_x2 - crop_x1) * ( image_width / crop_w ) * cfg.HEATMAP_SIZE ))
              heatmap_bbox_y2 = int(np.round((bbox_y2 - crop_y1) * ( image_height / crop_h) * cfg.HEATMAP_SIZE ))

              #print "%d:%d, %d:%d" % (heatmap_bbox_y1, heatmap_bbox_y2, heatmap_bbox_x1, heatmap_bbox_x2)

              heatmaps_bbox = heatmaps[heatmap_bbox_y1:heatmap_bbox_y2, heatmap_bbox_x1:heatmap_bbox_x2]

              bbox_w = (bbox_x2 - bbox_x1) * image_width 
              bbox_h = (bbox_y2 - bbox_y1) * image_height

              keypoints = get_local_maxima(heatmaps_bbox, bbox_x1, bbox_y1, bbox_w, bbox_h, image_width, image_height)

            else:

              keypoints = get_local_maxima(heatmaps, crop_x1, crop_y1, crop_w, crop_h, image_width, image_height)
            
            bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox
            pred_parts = []
            for k in keypoints:
              s_idx = np.argsort(k['score']).tolist()
              s_idx.reverse()
              
              if len(s_idx) == 0:
                x = 0
                y = 0
                v = 0
              else:
                x = k['x'][s_idx[0]]
                y = k['y'][s_idx[0]]
                v = 1
              
              pred_parts += [x, y, v]
               

            # Store the results   
            pred_annotations.append({
              'image_id' : gt_image_id,
              'keypoints' : pred_parts,
              'score' : 1.,
              'category_id' : 1
            })

            gt_parts_x = parts[0::2] * image_width
            gt_parts_y = parts[1::2] * image_height
            gt_parts = np.transpose(np.vstack([gt_parts_x, gt_parts_y, part_visibilities]), [1, 0])
            gt_parts = gt_parts.ravel().tolist()
            
            x1, y1, x2, y2 = bbox * np.array([image_width, image_height, image_width, image_height])
            w = x2 - x1
            h = y2 - y1
            gt_annotations.append({
              "id" : gt_annotation_id,
              "image_id" : gt_image_id,
              "category_id" : 1,
              "area" : w * h,
              "bbox" : [x1, y1, w, h],
              "iscrowd" : 0,
              "keypoints" : gt_parts,
              "num_keypoints" : np.sum(part_visibilities)
            })
            
            dataset_image_ids.add(gt_image_id)
            
            gt_annotation_id += 1
            gt_image_id += 1

            

          print print_str % (step, (dt / cfg.BATCH_SIZE) * 1000) 
          step += 1
          
          if max_iterations > 0 and step == max_iterations:
              break
          
      except Exception as e:
        # Report exceptions to the coordinator.
        coord.request_stop(e)
      
      # When done, ask the threads to stop. It is innocuous to request stop twice.
      coord.request_stop()
      # And wait for them to actually do it.
      coord.join(threads)
      
      gt_dataset = {
        'annotations' : gt_annotations,
        'images' : [{'id' : img_id} for img_id in dataset_image_ids],
        'categories' : [{ 'id' : 1 }]
      }

      gt_coco = COCO()
      gt_coco.dataset = gt_dataset
      gt_coco.createIndex()

      pred_coco = gt_coco.loadRes(pred_annotations)

      cocoEval = COCOeval(gt_coco, pred_coco, iouType='keypoints')

      #cocoEval.params.useCats = 0
      #cocoEval.params.areaRange = ("medium","large") # I just created a different gt annotation file
      cocoEval.evaluate()
      cocoEval.accumulate()

      old_stdout = sys.stdout
      sys.stdout = captured_stdout = StringIO()
      cocoEval.summarize()
      sys.stdout = old_stdout

      summary_op = tf.merge_all_summaries()
      summary_writer = tf.train.SummaryWriter(summary_dir)
      summary = tf.Summary()
      summary.ParseFromString(session.run(summary_op))

      for line in captured_stdout.getvalue().split('\n'):
        if line != "":
          description, score = line.rsplit("=", 1)
          description = description.strip()
          score = float(score)

          summary.value.add(tag=description, simple_value=score)

          print "%s: %0.3f" % (description, score)
      
      summary_writer.add_summary(summary, global_step)
      summary_writer.flush()
      summary_writer.close()

def parse_args():

    parser = argparse.ArgumentParser(description='Test an Inception V3 network')

    parser.add_argument('--tfrecords', dest='tfrecords',
                        help='paths to tfrecords files', type=str,
                        nargs='+', required=True)

    parser.add_argument('--checkpoint_path', dest='checkpoint_path',
                          help='path to directory where the checkpoint files are stored. The latest model will be tested against.', type=str,
                          required=False, default=None)
                          
    parser.add_argument('--summary_dir', dest='summary_dir',
                        help='Path to the directory where the results will be saved',
                        required=True, type=str)
                        
    parser.add_argument('--config', dest='config_file',
                        help='Path to the configuration file',
                        required=True, type=str)
    
    parser.add_argument('--max_iterations', dest='max_iterations',
                        help='Maximum number of iterations to run. Set to 0 to run on all records.',
                        required=False, type=int, default=0)

    args = parser.parse_args()
    
    return args

if __name__ == '__main__':

    args = parse_args()
    cfg = parse_config_file(args.config_file)

    print "Configurations:"
    print pprint.pprint(cfg)

    eval(
      tfrecords=args.tfrecords,
      checkpoint_path=args.checkpoint_path,
      summary_dir = args.summary_dir,
      max_iterations = args.max_iterations,
      cfg=cfg
    )