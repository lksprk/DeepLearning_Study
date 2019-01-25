## Understanding tf.argmax ##

import tensorflow as tf
import functions

a1 = tf.Variable([0.1, 0.3, 0.5])
functions.showOperation(tf.argmax(a1, 0))



import tensorflow as tf
import functions

a2 = tf.Variable([[0.1, 0.3, 0.5]])
functions.showOperation(tf.argmax(a2, 0))
functions.showOperation(tf.argmax(a2, 1))



import tensorflow as tf
import functions

a3 = tf.Variable([[[0.1, 0.3, 0.5],
                   [0.3, 0.5, 0.1]],
                  [[0.5, 0.1, 0.3],
                   [0.1, 0.3, 0.5]],
                  [[0.3, 0.5, 0.1],
                   [0.5, 0.1, 0.3]]])

functions.showOperation(tf.argmax(a3, 0))
functions.showOperation(tf.argmax(a3, 1))
functions.showOperation(tf.argmax(a3, 2))
