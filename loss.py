import tensorflow as tf
slim = tf.contrib.slim

def add_heatmaps_loss(gt_heatmaps, pred_heatmaps, background_heatmaps, add_summaries=True):
  """
  Args:
    gt_heatmaps : 
    pred_heatmaps : an array of heatmaps with the same shape as gt_heatmaps
  """
  # Should we also scale the background heatmaps? 
  # Shift the background_heatmaps up by 1 so that by default a unit is not penalized
  background_heatmaps += 1.

  total_loss = 0
  summaries = []
  for i, pred in enumerate(pred_heatmaps):
    # params: predictions, targets, weights
    l = tf.contrib.losses.mean_squared_error(pred, gt_heatmaps, background_heatmaps)
    #l = tf.nn.l2_loss(gt_heatmaps - pred)
    slim.losses.add_loss(l)
    total_loss += l

    if add_summaries:
      summaries.append(tf.scalar_summary('heatmap_loss_%d' % i, l))

  return total_loss, summaries
  
