!wget https://raw.githubusercontent.com/jaeyong1/deeplearning-forus/master/chap06/airplane_takeoff/testset_x.csv
!wget https://raw.githubusercontent.com/jaeyong1/deeplearning-forus/master/chap06/airplane_takeoff/testset_y.csv
!wget https://raw.githubusercontent.com/jaeyong1/deeplearning-forus/master/chap06/airplane_takeoff/trainset.csv
!ls -al

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def shuffle_data(x_train, y_train):
  temp_index = np.arange(len(x_train))
  np.random.shuffle(temp_index)
 
  x_temp = np.zeros(x_train.shape)
  y_tmep = np.zeros(y_train.shape)
  x_temp = x_train[temp_index]
  y_temp = y_train[temp_index]

  return x_temp, y_temp

def minmax_normalize(x):
  xmax, xmin = x.max(), x.min()
  return (x - xmin) / (xmax - xmin)

def minmax_get_norm(realx, arrx):
  xmax, xmin = arrx.max(), arrx.min()
  normx = (relax - xmin) / (xmax - xmin)
  return normx

def minma_get_denorm(normx, arrx):
  xmax, xmin = arrx.max(), arrx.min()
  relax = normx * (xmax - xmin) + xmin
  return relax

def main():
  traincsvdata = np.loadtxt('trainset.csv', unpack = True, delimiter=',', skiprows=1)
  num_points = len(traincsvdata[0])
  print("points : ", num_points)
  
  x1_data = traincsvdata[0]
  x2_data = traincsvdata[1]
  y_data = traincsvdata[2]
  
  plt.plot(x1_data, y_data, 'mo')
  plt.suptitle('Traing set(x1)', fontsize=16)
  plt.xlabel('speed to take off')
  plt.ylabel('disgtance')
  plt.show()
  
  plt.plot(x2_data, y_data, 'bo')
  plt.suptitle('Traing set(x2)', fontsize=16)
  plt.xlabel('weight')
  plt.ylabel('distance')
  plt.show()
  
  x1_data = minmax_normalize(x1_data)
  x2_data = minmax_normalize(x2_data)
  y_data = minmax_normalize(y_data)
  
  x_data = [[item for item in x1_data], [item for item in x2_data]]
  x_data = np.reshape(x_data, 600, order='F')
  x_data = np.reshape(x_data, (-1, 2))
  y_data = np.reshape(y_data, [len(y_data), 1])
  
  BATCH_SIZE = 5
  BATCH_NUM = int(len(x1_data)/BATCH_SIZE)
  
  input_data = tf.placeholder(tf.float32, shape=[None, 2])
  output_data = tf.placeholder(tf.float32, shape=[None, 1])
  
  W1 = tf.Variable(tf.random_uniform([2, 5], 0.0, 1.0))
  W2 = tf.Variable(tf.random_uniform([5, 3], 0.0, 1.0))
  W_out = tf.Variable(tf.random_uniform([3, 1], 0.0, 1.0))
  
  hidden1 = tf.nn.sigmoid(tf.matmul(input_data, W1))
  hidden2 = tf.nn.sigmoid(tf.matmul(hidden1, W2))
  output = tf.matmul(hidden2, W_out)
  
  loss = tf.reduce_mean(tf.square(output-output_data))
  optimizer = tf.train.AdamOptimizer(0.01)
  train = optimizer.minimize(loss)
  
  init = tf.global_variables_initializer()
  
  sess = tf.Session()
  sess.run(init)
  
  for step in range(1000):
    index = 0
    
    x_data, y_data = shuffle_data(x_data, y_data)
    
  for batch_iter in range(BATCH_NUM-1):
    feed_dict = {input_data: x_data[index: index+BATCH_SIZE], output_data: y_data[index: index+BATCH_SIZE]}
    sess.run(train, feed_dict = feed_dict)
    index += BATCH_SIZE
    
    print('#학습완료. 임의값으로 이륙거리 추정#')
    arr_ask_x = [[290, 210],
                [320, 210],
                [300, 300],
                [320, 300]
                ]
    
    for i in range(len(arr_ask_x)):
      ask_x = [arr_ask_x[i]]
      ask_norm_x = [[minmax_get_norm(ask_x[0][0], traincsvdata[0]), minmax_get_norm(ask_x[0][1], traincsvdata[1])]]
      answer_norm_y = sess.run(output, feed_dict = {input_data: ask_norm_x})
      answer_y = minmax_get_denorm(answer_norm_y, traincsvdata[2])
        
      print("이륙거리계산) 이륙속도(x1): ", ask_x[0][0], "km/h, ", "비행기무게(x2): ", ask_x[0][1], "ton, ", "이륙거리(y): ", answer_y[0][0], "m")