import tensorflow as tf

def mse(y_approximate-y_true):
	return tf.reduce_mean (tf.square (y_approximate-y_true))

def mse_and_trace (y_approximate, y_true):
	return tf.reduce_mean (tf.square (y_approximate-y_true)) + tf.reduce_mean (tf.square (tf.trace(y_approximate) - tf.trace(y_true)))
