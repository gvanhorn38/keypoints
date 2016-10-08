import tensorflow as tf
slim = tf.contrib.slim

def add_heatmaps_loss(gt_heatmaps, pred_heatmaps):
  """
  Args:
    gt_heatmaps : 
    pred_heatmaps : an array of heatmaps with the same shape as gt_heatmaps
  """

  total_loss = 0
  for pred in pred_heatmaps:
    l = tf.nn.l2_loss(gt_heatmaps - pred)
    slim.losses.add_loss(l)
    total_loss += l
  return total_loss
  
