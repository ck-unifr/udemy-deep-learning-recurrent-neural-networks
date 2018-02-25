# The script is updated from
# https://github.com/lazyprogrammer/machine_learning_examples/blob/master/hmm_class/scan1.py

from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


import numpy as np
import theano
import theano.tensor as T


x = T.vector('x')

def square(x):
  return x*x

outputs, updates = theano.scan(
  fn=square,
  sequences=x,
  n_steps=x.shape[0],
)

square_op = theano.function(
  inputs=[x],
  outputs=[outputs],
)

o_val = square_op(np.array([1, 2, 3, 4, 5]))

print("output:", o_val)
