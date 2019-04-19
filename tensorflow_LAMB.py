"""LAMB for TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops
from tensorflow.python.util.tf_export import tf_export


class LAMBOptimizer(optimizer.Optimizer):

  def __init__(self, learning_rate=0.001, wd= 0.01, beta1=0.9, beta2=0.999, epsilon=1e-6,
               use_locking=False, name="LAMB"):

    super(LAMBOptimizer, self).__init__(use_locking, name)
    self._lr = learning_rate
    self._beta1 = beta1
    self._beta2 = beta2
    self._epsilon = epsilon
    self._wd = wd

    # Tensor versions of the constructor arguments, created in _prepare().
    self._lr_t = None
    self._beta1_t = None
    self._beta2_t = None
    self._epsilon_t = None
    self._wd_t = None

  def _get_beta_accumulators(self):
    with ops.init_scope():
      if context.executing_eagerly():
        graph = None
      else:
        graph = ops.get_default_graph()
      return (self._get_non_slot_variable("beta1_power", graph=graph),
              self._get_non_slot_variable("beta2_power", graph=graph))

  def _create_slots(self, var_list):
    # Create the beta1 and beta2 accumulators on the same device as the first
    # variable. Sort the var_list to make sure this device is consistent across
    # workers (these need to go on the same PS, otherwise some updates are
    # silently ignored).
    first_var = min(var_list, key=lambda x: x.name)
    self._create_non_slot_variable(initial_value=self._beta1,
                                   name="beta1_power",
                                   colocate_with=first_var)
    self._create_non_slot_variable(initial_value=self._beta2,
                                   name="beta2_power",
                                   colocate_with=first_var)

    # Create slots for the first and second moments.
    for v in var_list:
      self._zeros_slot(v, "m", self._name)
      self._zeros_slot(v, "v", self._name)

  def _prepare(self):
    lr = self._call_if_callable(self._lr)
    beta1 = self._call_if_callable(self._beta1)
    beta2 = self._call_if_callable(self._beta2)
    epsilon = self._call_if_callable(self._epsilon)
    wd = self._call_if_callable(self._wd)

    self._lr_t = ops.convert_to_tensor(lr, name="learning_rate")
    self._beta1_t = ops.convert_to_tensor(beta1, name="beta1")
    self._beta2_t = ops.convert_to_tensor(beta2, name="beta2")
    self._epsilon_t = ops.convert_to_tensor(epsilon, name="epsilon")
    self._wd_t = ops.convert_to_tensor(wd, name="wd")

  def _apply_dense(self, grad, var):
    lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
    beta1_power, beta2_power = self._get_beta_accumulators()
    beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
    beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
    eps = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)
    wd_lambda = math_ops.cast(self._wd_t, var.dtype.base_dtype)

    v = self.get_slot(var, "v")
    v_t = v.assign(beta2_t * v + (1. - beta2_t) * grad**2)
    m = self.get_slot(var, "m")
    m_t = m.assign(beta1_t * m + (1. - beta1_t) * grad)

    # add l2 normalizations and set ratio
    r1 = tf.sqrt(tf.reduce_sum(tf.square(var)))
    step = m_t / (tf.sqrt(v_t) + eps) + wd_lambda * var
    r2 = tf.sqrt(tf.reduce_sum(tf.square(step)))

    var_update = state_ops.assign_sub(var, lr_t * tf.minimum(r1 / r2, 10) * step) #set upper bound of 10 for r1/r2 / section 3.3.1 of LAMB paper
    return control_flow_ops.group(*[var_update, v_t, m_t])
