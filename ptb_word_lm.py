"""Example / benchmark for building a PTB PSRNN model for character prediction.

To run:

$ python ptb_word_lm.py --data_path=data/

python c:/Users/Carlton/Dropbox/psrnn_tensorflow/psrnn_code/ptb_psrnn_random/ptb_word_lm.py --data_path=c:/Users/Carlton/Dropbox/psrnn_tensorflow/git/psrnn/data/

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import time

import numpy as np
import tensorflow as tf

import reader
import psrnn_cell_impl
import two_stage_regression

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("data_path", None,
                    "Where the training/test data is stored.")

FLAGS = flags.FLAGS

class Config(object):
  # Two Stage Regression Parameters
  nRFF_Obs = 2000
  nRFF_P = 2000
  nRFF_F = 2000
  dim_Obs = 20
  dim_P = 20
  dim_F = 20
  reg_rate = 1*10**-3
  obs_window = 5
  kernel_width_Obs = 2
  kernel_width_P = 0.2
  kernel_width_F = 0.2
    
  # BPTT parameters
  init_scale = 0.0
  learning_rate = 1
  max_grad_norm = 0.25
  num_layers = 1
  num_steps = 20
  max_epoch = 10
  keep_prob = 1.0
  lr_decay = 1.0
  batch_size = 20
  vocab_size = 49
  seed = 0
  hidden_size = dim_Obs

def onehot(data, dim):
  data_onehot = np.zeros((dim, len(data)))
  data_onehot[np.array(data),np.arange(len(data))] = 1
  return data_onehot

def data_type():
  return tf.float32

class PTBInput(object):
  """The input data."""

  def __init__(self, config, data, name=None):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
    self.input_data, self.targets = reader.ptb_producer(
        data, batch_size, num_steps, name=name)

class PTBModel(object):
  """The PTB model."""

  def __init__(self, is_training, config, input_, params):
    self._input = input_

    batch_size = input_.batch_size
    num_steps = input_.num_steps
    size = config.hidden_size
    vocab_size = config.vocab_size

    def lstm_cell():
      if 'reuse' in inspect.getargspec(
          tf.contrib.rnn.BasicLSTMCell.__init__).args:
        return psrnn_cell_impl.PSRNNCell(
            size,
            params,
            reuse=tf.get_variable_scope().reuse)
      else:
        return psrnn_cell_impl.PSRNNCell(
            size,
            params)  
    attn_cell = lstm_cell
    if is_training and config.keep_prob < 1:
      def attn_cell():
        return tf.contrib.rnn.DropoutWrapper(
            lstm_cell(), output_keep_prob=config.keep_prob)
    psrnns = [attn_cell() for _ in range(config.num_layers)];
    cell = tf.contrib.rnn.MultiRNNCell(
        psrnns, state_is_tuple=True)

    self._initial_state = []
    for i in range(config.num_layers):
        self._initial_state.append(tf.constant(np.ones((batch_size,1)).dot(params.q_1.T), dtype=data_type()))
    self._initial_state = tuple(self._initial_state)
      
    # convert to one-hot encoding
    inputs = tf.one_hot(input_.input_data, config.vocab_size)

    # random fourier features
    W_rff = tf.get_variable('W_rff', initializer=tf.constant(params.W_rff.astype(np.float32)), dtype=data_type())
    b_rff = tf.get_variable('b_rff', initializer=tf.constant(params.b_rff.astype(np.float32)), dtype=data_type())
    
    self._W_rff = W_rff
    self._b_rff = b_rff

    z = tf.tensordot(inputs, W_rff,axes=[[2],[0]]) + b_rff
    inputs_rff = tf.cos(z)*np.sqrt(2.)/np.sqrt(config.nRFF_Obs)
    
    # dimensionality reduction
    U = tf.get_variable('U', initializer=tf.constant(params.U.astype(np.float32)),dtype=data_type())
    U_bias = tf.get_variable('U_bias',[config.hidden_size],initializer=tf.constant_initializer(0.0))
    inputs_embed = tf.tensordot(inputs_rff, U, axes=[[2],[0]]) + U_bias
    
    # update rnn state
    inputs_unstacked = tf.unstack(inputs_embed, num=num_steps, axis=1)
    outputs, state = tf.contrib.rnn.static_rnn(
        cell, inputs_unstacked, initial_state=self._initial_state)

    # reshape output
    output = tf.reshape(tf.stack(axis=1, values=outputs), [-1, size])
    
    # softmax
    initializer = tf.constant(params.W_pred.T.astype(np.float32))
    softmax_w = tf.get_variable("softmax_w", initializer=initializer, dtype=data_type())
    initializer = tf.constant(params.b_pred.T.astype(np.float32))
    softmax_b = tf.get_variable("softmax_b", initializer=initializer, dtype=data_type())
    logits = tf.matmul(output, softmax_w) + softmax_b    

    # Reshape logits to be 3-D tensor for sequence loss
    logits = tf.reshape(logits, [batch_size, num_steps, vocab_size])

    # use the contrib sequence loss and average over the batches
    loss = tf.contrib.seq2seq.sequence_loss(
        logits,
        input_.targets,
        tf.ones([batch_size, num_steps], dtype=data_type()),
        average_across_timesteps=False,
        average_across_batch=True
    )
    one_step_pred = tf.argmax(logits,axis=2)
    
    self._cost = cost = tf.reduce_sum(loss)
    self._num_correct_pred = tf.reduce_sum(tf.cast(tf.equal(tf.to_int32(one_step_pred),input_.targets), tf.float32))/batch_size
    self._final_state = state

    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self._lr)
    self._train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.contrib.framework.get_or_create_global_step())

    self._new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})    

  @property
  def input(self):
    return self._input

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def num_correct_pred(self):
      return self._num_correct_pred

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op
  
def run_epoch(session, model, eval_op=None, verbose=False, save_params=False):
  """Runs the model on the given data."""
  start_time = time.time()
  costs = 0.0
  correct_pred = 0.0
  iters = 0
  state = session.run(model.initial_state)

  fetches = {
      "cost": model.cost,
      "num_correct_pred": model.num_correct_pred,
      "final_state": model.final_state,
  }
  if eval_op is not None:
    fetches["eval_op"] = eval_op

  for step in range(model.input.epoch_size):
      
    feed_dict = {}
    for i, s in enumerate(model.initial_state):
      feed_dict[s] = state[i]

    vals = session.run(fetches, feed_dict)
    cost = vals["cost"]
    state = vals["final_state"]

    costs += cost
    iters += model.input.num_steps
    correct_pred += vals["num_correct_pred"]
    
    perplexity = np.exp(costs / iters)
    bpc = np.log2(perplexity)
    accuracy = correct_pred/iters

    if verbose and step % (model.input.epoch_size // 10) == 10:
      print("%.3f perplexity: %.3f bpc: %.3f accuracy: %.3f speed: %.0f wps" %
            (step * 1.0 / model.input.epoch_size, perplexity, bpc, accuracy,
             iters * model.input.batch_size / (time.time() - start_time)))
      
  return perplexity, bpc, accuracy

def main(_):
  if not FLAGS.data_path:#43493
    raise ValueError("Must set --data_path to PTB data directory")

  raw_data = reader.ptb_raw_data(FLAGS.data_path)
  train_data, valid_data, test_data, _, word_to_id = raw_data
  
  config = Config()
  eval_config = Config()
  eval_config.batch_size = 1
  eval_config.num_steps = 1
  
  # print the config
  for i in inspect.getmembers(config):
    # Ignores anything starting with underscore 
    # (that is, private and protected attributes)
    if not i[0].startswith('_'):
        # Ignores methods
        if not inspect.ismethod(i[1]):
            print(i)
            
  # reshape array of raw training labels
  raw_train_data = np.array(train_data).reshape([-1,len(train_data)])

  # convert characters to one-hot encoding
  train_data_onehot = onehot(train_data, config.vocab_size)

  # perform two stage regression to obtain initialization for PSRNN
  params = two_stage_regression.two_stage_regression(
          raw_train_data,
          train_data_onehot,
          config.kernel_width_Obs, config.kernel_width_P, config.kernel_width_F, 
          config.seed, 
          config.nRFF_Obs, config.nRFF_P, config.nRFF_F,
          config.dim_Obs, config.dim_P, config.dim_F, 
          config.reg_rate,
          config.obs_window)

  with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)

    with tf.name_scope("Train"):
      train_input = PTBInput(config=config, data=train_data, name="TrainInput")
      with tf.variable_scope("Model", reuse=None, initializer=initializer):
        m = PTBModel(is_training=True, config=config, input_=train_input, params=params)
      tf.summary.scalar("Training Loss", m.cost)
      tf.summary.scalar("Learning Rate", m.lr)

    with tf.name_scope("Valid"):
      valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mvalid = PTBModel(is_training=False, config=config, input_=valid_input, params=params)
      tf.summary.scalar("Validation Loss", mvalid.cost)

    with tf.name_scope("Test"):
      test_input = PTBInput(config=eval_config, data=test_data, name="TestInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mtest = PTBModel(is_training=False, config=eval_config,
                         input_=test_input, params=params)

    sv = tf.train.Supervisor()
    with sv.managed_session() as session:

      valid_perplexity_all = []        
      valid_bpc_all = []
      valid_acc_all = []
     
      
      valid_perplexity, valid_bpc, valid_acc = run_epoch(session, mvalid)
      print("Epoch: %d Valid Perplexity: %.3f Valid BPC: %.3f Valid Accuracy: %.3f"
            % (0, valid_perplexity, valid_bpc, valid_acc))
      valid_perplexity_all.append(valid_perplexity)
      valid_bpc_all.append(valid_bpc)
      valid_acc_all.append(valid_acc)
      
      for i in range(config.max_epoch):
        m.assign_lr(session, config.learning_rate * config.lr_decay)

        print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
        
        train_perplexity, train_bpc, train_acc = run_epoch(session, m, eval_op=m.train_op, verbose=True)
        print("Epoch: %d Train Perplexity: %.3f Train BPC: %.3f Train Accuracy: %.3f" 
              % (i + 1, train_perplexity, train_bpc, train_acc))
        
        valid_perplexity, valid_bpc, valid_acc = run_epoch(session, mvalid)
        print("Epoch: %d Valid Perplexity: %.3f Valid BPC: %.3f Valid Accuracy: %.3f" 
              % (i + 1, valid_perplexity, valid_bpc, valid_acc))
        valid_perplexity_all.append(valid_perplexity)
        valid_bpc_all.append(valid_bpc)
        valid_acc_all.append(valid_acc)

      test_perplexity, test_bpc, test_acc = run_epoch(session, mtest, save_params=True)
      print("Test Perplexity: %.3f Test BPC: %.3f Test Accuracy: %.3f" % (test_perplexity, test_bpc, test_acc))
      
      print("validation perplexity\n",valid_perplexity_all)
      print("validation bpc\n",valid_bpc_all)
      print("validation acc\n",valid_acc_all)

if __name__ == "__main__":
  tf.app.run()
