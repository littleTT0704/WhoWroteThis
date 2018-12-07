import tensorflow as tf
# OCR = [charmap] --> [vec 8]

def _variable_on_cpu(name, shape, initializer):
  with tf.device('/cpu:0'):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var

def _variable_with_weight_decay(name, shape, stddev, wd):
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def init_ocr(C):
    with tf.variable_scope('L0') as scope:
        F = _variable_with_weight_decay('weights', shape=[4, 4, 1, 2], stddev=5e-2, wd=None)
        L = tf.nn.max_pool(tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(C, F, [1, 1, 1, 1], padding='SAME'), _variable_on_cpu('biases', [2], tf.constant_initializer(0.0))), name='R0'),
            ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME', name='P0')

    with tf.variable_scope('L1') as scope:
        F = _variable_with_weight_decay('weights', shape=[4, 4, 2, 1], stddev=5e-2, wd=None)
        L = tf.nn.max_pool(tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(L, F, [1, 1, 1, 1], padding='SAME'), _variable_on_cpu('biases', [2], tf.constant_initializer(0.0))), name='R1'),
            ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='SAME', name='P1')

    with tf.variable_scope('DS') as scope:
        F = _variable_with_weight_decay('weights', shape=[2, 2, 1, 1], stddev=5e-2, wd=None)
        L = tf.nn.max_pool(tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(L, F, [1, 1, 1, 1], padding='SAME'), _variable_on_cpu('biases', [2], tf.constant_initializer(0.0))), name='R2'),
            ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='P2')

    R = tf.nn.relu(tf.matmul(tf.reshape(L, [4]), weight_variable([4, 4])) + bias_variable([4]), name='RF')
