import tensorflow as tf

X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

W = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

hypothesis = X * W + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(5001):
  cost_val, W_val, b_val, _ = sess.run([cost, W, b, train], feed_dict = {X:[5, 7], Y:[52, 72]})
  if step%500 == 0:
    print(step, cost_val, W_val, b_val)
    
print("예측 Y :", sess.run(hypothesis, feed_dict={X:[8]}))