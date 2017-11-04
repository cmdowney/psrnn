import tensorflow as tf
import numpy as np

class PSRNNCell(tf.contrib.rnn.RNNCell):

  def __init__(self, num_units, params, reuse=None):
    super(PSRNNCell, self).__init__(_reuse=reuse)
    self._num_units = num_units
    self._params = params

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  @property
  def W(self):
      return self._W
  
  @property
  def W_FE_F(self):
      return self._W_FE_F
  
  @property
  def b_FE_F(self):
      return self._b_FE_F
  
  @property
  def state(self):
      return self._state

  def call(self, inputs, state):
      
    self._state = state
    
    
    weights_initializer = tf.constant(self._params.W_FE_F.T.astype(np.float32))
    weights = tf.get_variable(
        "weights",
        initializer=weights_initializer)
    
    bias_initializer = tf.constant(self._params.b_FE_F.T.astype(np.float32))
    biases = tf.get_variable(
        "bias1", #[self._num_units**2],
        initializer=bias_initializer)
    
    self._W_FE_F = weights
    self._b_FE_F = biases

    W = tf.add(tf.matmul(state, weights), biases)
    batchedW = tf.split(W, W.shape[0], 0)
    batchedInputs = tf.split(inputs, inputs.shape[0], 0)
    bached_output = []
    for W, input in zip(batchedW, batchedInputs):
      W_square = tf.transpose(tf.reshape(W, [self._num_units, self._num_units]))
      self._W = W_square
      new_s = tf.matmul(input,W_square)
      new_s_normalized = new_s/tf.norm(new_s)
      bached_output.append(new_s_normalized)
    output = tf.concat(bached_output,0)
    return output, output






