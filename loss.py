import tensorflow as tf
slim = tf.contrib.slim

def add_heatmaps_loss(gt_heatmaps, pred_heatmaps, background_heatmaps, add_summaries, cfg):
  """
  Args:
    gt_heatmaps : 
    pred_heatmaps : an array of heatmaps with the same shape as gt_heatmaps
  """
  # Should we also scale the background heatmaps? 
  # Shift the background_heatmaps up by 1 so that by default a unit is not penalized
  shifted_background_heatmaps = background_heatmaps + 1.

  total_loss = 0
  summaries = []
  
  decay = None
  if cfg.LOSS.USE_SCALED_BACKGROUND:
    decay = cfg.LOSS.SCALE_FACTORS
    
  for i, pred in enumerate(pred_heatmaps):
    # params: predictions, targets, weights
    #l = tf.contrib.losses.mean_squared_error(pred, gt_heatmaps, background_heatmaps)
    #l = tf.nn.l2_loss(gt_heatmaps - pred)
    
    if cfg.LOSS.USE_BACKGROUND_AS_COST_MATRIX:
      # We want (x - y)W(x - y)
      cost_matrix = shifted_background_heatmaps * cfg.COST_MATRIX_SCALE
      diff = gt_heatmaps - pred
      squared_scaled_diff = diff * cost_matrix * diff
      l = tf.reduce_sum(squared_scaled_diff) / 2.
      #l = tf.nn.l2_loss((gt_heatmaps - pred) * shifted_background_heatmaps)
    
    elif cfg.LOSS.USE_SCALED_BACKGROUND:
      l = tf.nn.l2_loss((gt_heatmaps + (background_heatmaps * decay[i]))  - pred)

    
    elif cfg.LOSS.NO_BACKGROUND:
      l = tf.nn.l2_loss(gt_heatmaps - pred)
    
    else:
      raise Exception("Unknown loss type")

    slim.losses.add_loss(l)
    total_loss += l

    if add_summaries:
      summaries.append(tf.scalar_summary('heatmap_loss_%d' % i, l))

  return total_loss, summaries
  
