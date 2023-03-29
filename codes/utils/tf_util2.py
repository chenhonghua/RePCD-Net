import numpy as np
import tensorflow as tf
from tensorflow.contrib.seq2seq.python.ops import attention_wrapper
import tensorflow.contrib.seq2seq as seq2seq

def lrelu(x, alpha=0.2):
  return tf.nn.relu(x) - alpha * tf.nn.relu(-x)


# def lrelu2(x, leak=0.2, name="lrelu"):
#     with tf.variable_scope(name):
#         f1 = 0.5 * (1 + leak)
#         f2 = 0.5 * (1 - leak)
#         return f1 * x + f2 * abs(x)

def instance_norm(net, train=True,weight_decay=0.00001):
    batch, rows, cols, channels = [i.value for i in net.get_shape()]
    var_shape = [channels]
    mu, sigma_sq = tf.nn.moments(net, [1, 2], keep_dims=True)

    shift = tf.get_variable('shift',shape=var_shape,
                            initializer=tf.zeros_initializer,
                            regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
    scale = tf.get_variable('scale', shape=var_shape,
                            initializer=tf.ones_initializer,
                            regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
    epsilon = 1e-3
    normalized = (net - mu) / tf.square(sigma_sq + epsilon)
    return scale * normalized + shift

def conv2d(inputs,
           num_output_channels,
           kernel_size,
           scope=None,
           stride=[1, 1],
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.00001,
           activation_fn=tf.nn.relu,
           bn=False,
           ibn = False,
           bn_decay=None,
           use_bias = True,
           is_training=None,
           reuse=None):
  """ 2D convolution with non-linear operation.

  Args:
    inputs: 4-D tensor variable BxHxWxC
    num_output_channels: int
    kernel_size: a list of 2 ints
    scope: string
    stride: a list of 2 ints
    padding: 'SAME' or 'VALID'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable

  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope,reuse=reuse) as sc:
      if use_xavier:
          initializer = tf.contrib.layers.xavier_initializer()
      else:
          initializer = tf.truncated_normal_initializer(stddev=stddev)

      outputs = tf.layers.conv2d(inputs,num_output_channels,kernel_size,stride,padding,
                                 kernel_initializer=initializer,
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                 bias_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                 use_bias=use_bias,reuse=None)
      
      assert not (bn and ibn)
      if bn:
          outputs = tf.layers.batch_normalization(outputs,momentum=bn_decay,training=is_training,renorm=False,fused=True)
          #outputs = tf.contrib.layers.batch_norm(outputs,is_training=is_training)
      if ibn:
          outputs = instance_norm(outputs,is_training)


      if activation_fn is not None:
        outputs = activation_fn(outputs)

      return outputs


def fully_connected(inputs,
                    num_outputs,
                    scope,
                    use_xavier=True,
                    stddev=1e-3,
                    weight_decay=0.00001,
                    activation_fn=tf.nn.relu,
                    bn=False,
                    bn_decay=None,
                    use_bias = True,
                    is_training=None):
    """ Fully connected layer with non-linear operation.

    Args:
      inputs: 2-D tensor BxN
      num_outputs: int

    Returns:
      Variable tensor of size B x num_outputs.
    """

    with tf.variable_scope(scope) as sc:
        if use_xavier:
            initializer = tf.contrib.layers.xavier_initializer()
        else:
            initializer = tf.truncated_normal_initializer(stddev=stddev)

        outputs = tf.layers.dense(inputs,num_outputs,
                                  use_bias=use_bias,kernel_initializer=initializer,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                  bias_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                  reuse=None)

        if bn:
            outputs = tf.layers.batch_normalization(outputs, momentum=bn_decay, training=is_training, renorm=False)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return outputs

def rnn_encoder(inputs,
        hidden_size,
        scope,
        activation_fn=tf.nn.relu,
        bn=False,
        bn_decay=None,
        is_training=None):
  """ RNN encoder with no-linear operation.
  Args:
    inputs: 4-D tensor variable BxNxTxD
    hidden_size: int
    scope: encoder
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable
  Return:
    Variable Tensor BxNxD
  """
  with tf.variable_scope(scope) as sc:
    batch_size = inputs.get_shape()[0].value
    npoint = inputs.get_shape()[1].value
    nstep = inputs.get_shape()[2].value
    in_size = inputs.get_shape()[3].value
    reshaped_inputs = tf.reshape(inputs, (-1, nstep, in_size))
    # cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)
    cell = tf.nn.rnn_cell.LSTMCell(hidden_size)
    # cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
    h0 = cell.zero_state(batch_size*npoint, np.float32)
    output, state = tf.nn.dynamic_rnn(cell, reshaped_inputs, initial_state=h0)
    outputs = tf.reshape(state.h, (-1, npoint, hidden_size))

    if bn:
      outputs = batch_norm_for_fc(outputs, is_training, bn_decay, 'bn')

    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return outputs

def seq2seq_without_attention(inputs,
        hidden_size,
        scope,
        activation_fn=tf.nn.relu,
        bn=False,
        bn_decay=None,
        is_training=None):
    """ sequence model without attention.
    Args:
      inputs: 4-D tensor variable BxNxTxD
      hidden_size: int
      scope: encoder
      activation_fn: function
      bn: bool, whether to use batch norm
      bn_decay: float or float tensor variable in [0,1]
      is_training: bool Tensor variable
    Return:
      Variable Tensor BxNxD
    """
    with tf.variable_scope(scope) as sc:
        batch_size = inputs.get_shape()[0].value
        npoint = inputs.get_shape()[1].value
        nstep = inputs.get_shape()[2].value
        in_size = inputs.get_shape()[3].value
        reshaped_inputs = tf.reshape(inputs, (-1, nstep, in_size))#(b*Nx4x128)

        with tf.variable_scope('encoder'):
            # build encoder
            encoder_cell = tf.nn.rnn_cell.LSTMCell(hidden_size)
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, reshaped_inputs,
                                                               sequence_length=tf.fill([batch_size * npoint], nstep),
                                                               dtype=tf.float32, time_major=False)
            #outputs(batch_size, max_time, cell.output_size), state(2,batch_size, cell.output_size)==(c,h)
            #outputs(b*Nx4x128),   state(2 x b*N x 128)
        with tf.variable_scope('decoder'):
            decoder_cell = tf.nn.rnn_cell.LSTMCell(hidden_size)
            decoder_inputs = tf.reshape(encoder_state.h, [batch_size * npoint, 1, hidden_size])#state.h(b*N x 1 x 128)

            # Helper to feed inputs for training: read inputs from dense ground truth vectors
            train_helper = seq2seq.TrainingHelper(inputs=decoder_inputs, sequence_length=tf.fill([batch_size * npoint], 1),
                                                  time_major=False)
            decoder_initial_state = decoder_cell.zero_state(batch_size=batch_size * npoint, dtype=tf.float32)
            train_decoder = seq2seq.BasicDecoder(cell=decoder_cell, helper=train_helper,
                                                 initial_state=decoder_initial_state, output_layer=None)
            decoder_outputs_train, decoder_last_state_train, decoder_outputs_length_train = seq2seq.dynamic_decode(
                decoder=train_decoder, output_time_major=False, impute_finished=True)
        outputs = tf.reshape(decoder_last_state_train.c, (-1, npoint, hidden_size))
        if bn:
            outputs = batch_norm_for_fc(outputs, is_training, bn_decay, 'bn')

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs

def seq2seq_with_attention(inputs,
        hidden_size,
        scope,
        activation_fn=tf.nn.relu,
        bn=False,
        bn_decay=None,
        is_training=None):
    """ sequence model with attention.
       Args:
         inputs: 4-D tensor variable BxNxTxD #bxNx4x128
         hidden_size: int
         scope: encoder
         activation_fn: function
         bn: bool, whether to use batch norm
         bn_decay: float or float tensor variable in [0,1]
         is_training: bool Tensor variable
       Return:
         Variable Tensor BxNxD
       """
    with tf.variable_scope(scope) as sc:
        batch_size = inputs.get_shape()[0].value
        npoint = inputs.get_shape()[1].value
        nstep = inputs.get_shape()[2].value
        in_size = inputs.get_shape()[3].value
        reshaped_inputs = tf.reshape(inputs, (-1, nstep, in_size))

        with tf.variable_scope('encoder'):
            #build encoder
            encoder_cell = tf.nn.rnn_cell.LSTMCell(hidden_size)
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, reshaped_inputs,
                                                               sequence_length=tf.fill([batch_size*npoint], nstep),
                                                               dtype=tf.float32, time_major=False)
        with tf.variable_scope('decoder'):
            #build decoder
            decoder_cell = tf.nn.rnn_cell.LSTMCell(hidden_size)
            decoder_inputs = tf.reshape(encoder_state.h, [batch_size*npoint, 1, hidden_size])

            # building attention mechanism: default Bahdanau
            # 'Bahdanau' style attention: https://arxiv.org/abs/1409.0473
            attention_mechanism = seq2seq.BahdanauAttention(num_units=hidden_size, memory=encoder_outputs)
            # 'Luong' style attention: https://arxiv.org/abs/1508.04025
            # attention_mechanism = seq2seq.LuongAttention(num_units=hidden_size, memory=encoder_outputs)

            # AttentionWrapper wraps RNNCell with the attention_mechanism
            decoder_cell = seq2seq.AttentionWrapper(cell=decoder_cell, attention_mechanism=attention_mechanism,
                                                              attention_layer_size=hidden_size)

            # Helper to feed inputs for training: read inputs from dense ground truth vectors
            train_helper = seq2seq.TrainingHelper(inputs=decoder_inputs, sequence_length=tf.fill([batch_size*npoint], 1),
                                                  time_major=False)
            decoder_initial_state = decoder_cell.zero_state(batch_size=batch_size*npoint, dtype=tf.float32)
            train_decoder = seq2seq.BasicDecoder(cell=decoder_cell, helper=train_helper, initial_state=decoder_initial_state, output_layer=None)
            decoder_outputs_train, decoder_last_state_train, decoder_outputs_length_train = seq2seq.dynamic_decode(
                decoder=train_decoder, output_time_major=False, impute_finished=True)

        outputs = tf.reshape(decoder_last_state_train[0].h, (-1, npoint, hidden_size))
        if bn:
          outputs = batch_norm_for_fc(outputs, is_training, bn_decay, 'bn')

        if activation_fn is not None:
          outputs = activation_fn(outputs)
        return outputs

def rnn_encoder_base(inputs,
        hidden_size,
        scope,
        activation_fn=tf.nn.relu,
        bn=False,
        bn_decay=None,
        is_training=None):
  """ RNN encoder with no-linear operation.
  Args:
    inputs: 4-D tensor variable BxNxTxD
    hidden_size: int
    scope: encoder
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable
  Return:
    Variable Tensor BxNxD
  """
  with tf.variable_scope(scope) as sc:
    batch_size = inputs.get_shape()[0].value
    npoint = inputs.get_shape()[1].value
    nstep = inputs.get_shape()[2].value
    in_size = inputs.get_shape()[3].value
    reshaped_inputs = tf.reshape(inputs, (-1, nstep, in_size))
    cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)
    #cell = tf.nn.rnn_cell.LSTMCell(hidden_size)
    # cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
    h0 = cell.zero_state(batch_size*npoint, np.float32)
    output, state = tf.nn.dynamic_rnn(cell, reshaped_inputs, initial_state=h0)
    #print(state)
    outputs = tf.reshape(state, (-1, npoint, hidden_size))

    if bn:
      outputs = batch_norm_for_fc(outputs, is_training, bn_decay, 'bn')

    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return outputs

def seq2seq_with_attention_base(inputs,
        hidden_size,
        scope,
        activation_fn=tf.nn.relu,
        bn=False,
        bn_decay=None,
        is_training=None):
    """ sequence model with attention.
       Args:
         inputs: 4-D tensor variable BxNxTxD #bxNx4x128
         hidden_size: int
         scope: encoder
         activation_fn: function
         bn: bool, whether to use batch norm
         bn_decay: float or float tensor variable in [0,1]
         is_training: bool Tensor variable
       Return:
         Variable Tensor BxNxD
       """
    with tf.variable_scope(scope) as sc:
        batch_size = inputs.get_shape()[0].value
        npoint = inputs.get_shape()[1].value
        nstep = inputs.get_shape()[2].value
        in_size = inputs.get_shape()[3].value
        reshaped_inputs = tf.reshape(inputs, (-1, nstep, in_size))

        with tf.variable_scope('encoder'):
            #build encoder
            encoder_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)
            #encoder_cell = tf.nn.rnn_cell.LSTMCell(hidden_size)
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, reshaped_inputs,
                                                               sequence_length=tf.fill([batch_size*npoint], nstep),
                                                               dtype=tf.float32, time_major=False)
        with tf.variable_scope('decoder'):
            #build decoder
            decoder_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)
            #decoder_cell = tf.nn.rnn_cell.LSTMCell(hidden_size)
            decoder_inputs = tf.reshape(encoder_state, [batch_size*npoint, 1, hidden_size])

            # building attention mechanism: default Bahdanau
            # 'Bahdanau' style attention: https://arxiv.org/abs/1409.0473
            attention_mechanism = seq2seq.BahdanauAttention(num_units=hidden_size, memory=encoder_outputs)
            # 'Luong' style attention: https://arxiv.org/abs/1508.04025
            # attention_mechanism = seq2seq.LuongAttention(num_units=hidden_size, memory=encoder_outputs)

            # AttentionWrapper wraps RNNCell with the attention_mechanism
            decoder_cell = seq2seq.AttentionWrapper(cell=decoder_cell, attention_mechanism=attention_mechanism,
                                                              attention_layer_size=hidden_size)

            # Helper to feed inputs for training: read inputs from dense ground truth vectors
            train_helper = seq2seq.TrainingHelper(inputs=decoder_inputs, sequence_length=tf.fill([batch_size*npoint], 1),
                                                  time_major=False)
            decoder_initial_state = decoder_cell.zero_state(batch_size=batch_size*npoint, dtype=tf.float32)
            train_decoder = seq2seq.BasicDecoder(cell=decoder_cell, helper=train_helper, initial_state=decoder_initial_state, output_layer=None)
            decoder_outputs_train, decoder_last_state_train, decoder_outputs_length_train = seq2seq.dynamic_decode(
                decoder=train_decoder, output_time_major=False, impute_finished=True)
            #print(decoder_last_state_train[0])
        outputs = tf.reshape(decoder_last_state_train[0], (-1, npoint, hidden_size))
        if bn:
          outputs = batch_norm_for_fc(outputs, is_training, bn_decay, 'bn')

        if activation_fn is not None:
          outputs = activation_fn(outputs)
        return outputs

def batch_norm_template(inputs, is_training, scope, moments_dims_unused, bn_decay, data_format='NHWC'):
  """ Batch normalization on convolutional maps and beyond...
  Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow

  Args:
      inputs:        Tensor, k-D input ... x C could be BC or BHWC or BDHWC
      is_training:   boolean tf.Varialbe, true indicates training phase
      scope:         string, variable scope
      moments_dims:  a list of ints, indicating dimensions for moments calculation
      bn_decay:      float or float tensor variable, controling moving average weight
      data_format:   'NHWC' or 'NCHW'
  Return:
      normed:        batch-normalized maps
  """
  bn_decay = bn_decay if bn_decay is not None else 0.9
  return tf.contrib.layers.batch_norm(inputs,
                                      center=True, scale=True,
                                      is_training=is_training, decay=bn_decay,updates_collections=None,
                                      scope=scope,
                                      data_format=data_format)

def batch_norm_for_fc(inputs, is_training, bn_decay, scope):
  """ Batch normalization on FC data.
  
  Args:
      inputs:      Tensor, 2D BxC input
      is_training: boolean tf.Varialbe, true indicates training phase
      bn_decay:    float or float tensor variable, controling moving average weight
      scope:       string, variable scope
  Return:
      normed:      batch-normalized maps
  """
  return batch_norm_template(inputs, is_training, scope, [0,], bn_decay)
