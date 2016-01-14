from tensorflow.models.rnn.rnn_cell import RNNCell,linear
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs

from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh

from tensorflow.python.ops import clip_ops


class SpaceOdisseyLSTM(RNNCell):

  def __init__(self, num_units, input_size,
            use_input_gate=True,
            use_forget_gate=True,
            use_output_gate=True,
            use_input_activation_function=True,
            use_output_activation_function=True,
            use_peepholes=True,
            couple_input_and_forget=False,
            cell_clip=None,
            initializer=None,
            forget_bias=1.0):

    self._num_units = num_units
    self._input_size = input_size
    self._use_input_gate = use_input_gate
    self._use_forget_gate = use_forget_gate
    self._use_output_gate = use_output_gate
    self._use_input_activation_function = use_input_activation_function
    self._use_output_activation_function = use_output_activation_function
    self._couple_input_and_forget = couple_input_and_forget

    self._use_peepholes = use_peepholes
    self._cell_clip = cell_clip
    self._initializer = initializer
    self._forget_bias = forget_bias

    self._state_size = 2 * num_units
    self._output_size = num_units

    if self.__couple_input_and_forget:
      self._use_input_gate = True

    if self._use_peepholes:
      self._use_output_gate = True
      self._use_input_gate = True
      self._use_forget_gate = True

  @property
  def input_size(self):
    return self._input_size

  @property
  def output_size(self):
    return self._output_size

  @property
  def state_size(self):
    return self._state_size

  def __call__(self, x_t, state, scope=None):
    """Run one step of LSTM."""

    # this is our state. For efficiency we store the two matrices c & h as one
    # single tensor `state`.
    # Parameters of gates are concatenated into one multiply for efficiency.
    c_t_1 = array_ops.slice(state, [0, 0], [-1, self._num_units])
    y_t_1 = array_ops.slice(state, [0, self._num_units], [-1, self._num_units])

    def var(name, shape, initializer=self._initializer, dtype=x_t.dtype):
        """ var is a utility function to create/fetch a variable for the current scope"""
        return vs.get_variable(name, shape, initializer=initializer, dtype=dtype)

    dtype = x_t.dtype

    # scope all the operations within this cell
    with vs.variable_scope(scope or type(self).__name__):
      # always have at least one result gate
      n_gates = 1 + sum([self._use_output_gate, self._use_input_gate, self._use_forget_gate])

      shape = [self.input_size, n_gates * self._num_units]

      W = var("W", shape)

      # bias has to have shape [ n_gates * input_size ]
      b = var("b", shape[1], array_ops.zeros_initializer)

      # here we do all the matrix multiplications of W * x + b in one go.
      x_and_y_concatenated = array_ops.concat(1, [x_t, y_t_1])

      lstm_matrix = nn_ops.bias_add(math_ops.matmul(x_and_y_concatenated, W), b)

      # i = input_gate, j = new_input, f = forget_gate, o = output_gate

      gates = array_ops.split(1, n_gates, lstm_matrix)

      idx = 0
      if self._use_input_gate:
        i = gates[idx]
        idx += 1

      j = gates[idx]
      idx += 1

      if self._use_forget_gate:
        f = gates[idx]
        idx += 1

      if self._use_output_gate:
        o = gates[-1]

      if self._use_peepholes:
        p_f = var("p_f", [self._num_units])
        p_i = var("p_i", [self._num_units])
        p_o = var("p_o", [self._num_units])

      z_t = tanh(j)

      if self._use_input_gate:
        if self._use_input_activation_function == False:
          sigma_i_t = lambda (a) : a
        else:
          sigma_i_t = sigmoid

        if self._use_peepholes:
          i_t = sigma_i_t(i + p_i * c_t_1)
        else:
          i_t = sigma_i_t(i)
      else:
        i_t = None

      if self._use_forget_gate:
        if self._use_peepholes:
          f_t = sigmoid(f + self._forget_bias + p_f * c_t_1)
        else:
          f_t = sigmoid(f + self._forget_bias)
      else:
        f_t = None

      if self._couple_input_and_forget:
        f_t = 1 - i_t

      c_t = (f_t * c_t_1 if f_t else c_t_1) + (i_t * z_t if i_t else z_t)

      # Clipping
      if self._cell_clip is not None:
        c_t = clip_ops.clip_by_value(c_t, -self._cell_clip, self._cell_clip)

      # Output
      if self._use_output_gate:

        if self._use_output_activation_function == False:
          sigma_o_t = lambda (a) : a
        else:
          sigma_o_t = sigmoid

        if self._use_peepholes:
          o_t = sigma_o_t(o + p_o * c_t)
        else:
          o_t = sigma_o_t(o)
      else:
        o_t = None

      y_t = o_t * tanh(c_t) if o_t else tanh(c_t)

    return y_t, array_ops.concat(1, [c_t, y_t])
