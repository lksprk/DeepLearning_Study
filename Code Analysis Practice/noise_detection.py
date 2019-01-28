import tensorflow as tf
import numpy as np
import os
from sklearn.metrics import confusion_matrix

def search(dirname):
    features_dir_list = []
    labels_dir_list = []
    try:
        filenames = os.listdir(dirname)
        for filename in filenames:
            full_filename = os.path.join(dirname, filename)
            if os.path.isdir(full_filename):
                search(full_filename)
            elif 'features' in filename:
                features_dir_list.append(full_filename)
            elif 'labels' in filename:
                labels_dir_list.append(full_filename)
    except PermissionError:
        pass

    return features_dir_list, labels_dir_list


def make_dir_to_npy(dirname):
    train_features_dir, train_labels_dir = search(dirname + 'training')
    test_features_dir, test_labels_dir = search(dirname + 'testing')
    
    train_features_dir.sort()
    train_labels_dir.sort()
    test_features_dir.sort()
    test_labels_dir.sort()

    train_features = np.ndarray([0, 60, 41, 2])
    train_labels = np.ndarray([0, 5])
    test_features = np.ndarray([0, 60, 41, 2])
    test_labels = np.ndarray([0, 5])

    for index, train_feature_dir in enumerate(train_features_dir):
        train_features = np.append(train_features, np.load(train_feature_dir), axis=0)

    for index, train_label_dir in enumerate(train_labels_dir):
        train_labels = np.append(train_labels, np.load(train_label_dir), axis=0)
    
    for index, test_feature_dir in enumerate(test_features_dir):
        test_features = np.append(test_features, np.load(test_feature_dir), axis=0)
    
    for index, test_label_dir in enumerate(test_labels_dir):
        test_labels = np.append(test_labels, np.load(test_label_dir), axis=0)

    return train_features, train_labels, test_features, test_labels
    

def shuffle_numpy(features_np, labels_np):

    index_array = np.arange(len(features_np))
    np.random.shuffle(index_array)

    shuffled_features_np = np.zeros(np.shape(features_np))
    shuffled_labels_np = np.zeros(np.shape(labels_np))

    for i in range(len(shuffled_features_np)):
        shuffled_features_np[i] = features_np[index_array[i]]
        shuffled_labels_np[i] = labels_np[index_array[i]]

    return shuffled_features_np, shuffled_labels_np


def weight_variable(name, shape):
    initial = tf.contrib.layers.xavier_initializer()
    return tf.get_variable(name=name, shape = shape, initializer=initial)


def bias_variable(name, shape):
    initial = tf.contrib.layers.xavier_initializer()
    return tf.get_variable(name=name, shape = shape, initializer=initial)


def conv2d(x, W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1], padding='SAME')


def apply_convolution(x,shape1, shape2, num_channels,depth):
    weights = weight_variable([shape1, shape2, num_channels, depth])
    biases = bias_variable([depth])
    return tf.nn.relu(tf.add(conv2d(x, weights),biases))
    # weight_variable에 input이 왜 shape밖에 없는지???
    # bias_variable도 마찬가지

def apply_max_pool(x,shape1,shape2,stride_size1, stride_size2):
    return tf.nn.max_pool(x, ksize=[1, shape1, shape2, 1], 
                          strides=[1, stride_size1, stride_size2, 1], padding='SAME')


bands = 60
frames = 41

num_labels = 5 # 5개의 특징으로 구분?
num_channels = 2 # 여기서의 channel은 60*41 data 하나와 이 data를 이용해 계산한 delta 하나까지 포함해서 2개의 channel

batch_size = 128 # Total number of training examples present in a single batch
                 # Can't pass the entire dataset into the nn at once, so divide dataset into Number of Batches.
                 # Number of batches is equal to number of iterations for one epoch.
                 # Divide the dataset of 2000 examples into batches of 500 -> take 4 iterations to complete 1 epoch.
depth = 64 # filter 갯수
num_hidden = 512

learning_rate = 0.001
dropout_rate = 0.8
beta = 0.001
training_iterations = 3000

X = tf.placeholder(tf.float32, shape=[None,bands,frames,num_channels], name="X")
Y = tf.placeholder(tf.float32, shape=[None,num_labels], name="Y")
is_train = tf.placeholder(tf.bool,name='is_train')

keep_prob = tf.placeholder("float")

conv1_weights = weight_variable('c1w', [57, 6, num_channels, depth]) # what does 57 and 6 mean? filter?
conv1_biases = bias_variable('c1b', [depth])

## Batch Normalization
conv1 = tf.add(conv2d(X, conv1_weights), conv1_biases)
conv1_bn = tf.contrib.layers.batch_norm(conv1,decay=0.9, center=True,scale=True, is_training=is_train, updates_collections=None)
conv1 = tf.nn.relu(conv1_bn)
##

# conv1 = tf.nn.relu(tf.add(conv2d(X, conv1_weights), conv1_biases))

conv1 = tf.nn.dropout(conv1, keep_prob)
conv1_max = apply_max_pool(conv1, 4, 3, 1, 3)

conv2_weights = weight_variable('c2w', [1, 3, depth, depth])
conv2_biases = bias_variable('c2b', [depth])

## Batch Normalization
conv2 = tf.add(conv2d(conv1_max, conv2_weights), conv2_biases)
conv2_bn = tf.contrib.layers.batch_norm(conv2,decay=0.9, center=True,scale=True, is_training=is_train, updates_collections=None)
conv2 = tf.nn.relu(conv2_bn)
##

# conv2 = tf.nn.relu(tf.add(conv2d(conv1_max, conv2_weights), conv2_biases))

conv2_max = apply_max_pool(conv2, 1, 3, 1, 3)

conv3_weights = weight_variable('c3w', [1, 3, depth, depth])
conv3_biases = bias_variable('c3b', [depth])


## Batch Normalization
conv3 = tf.add(conv2d(conv2_max, conv3_weights), conv3_biases)
conv3_bn = tf.contrib.layers.batch_norm(conv3,decay=0.9, center=True,scale=True, is_training=is_train, updates_collections=None)
conv3 = tf.nn.relu(conv3_bn)
##

# conv2 = tf.nn.relu(tf.add(conv2d(conv1_max, conv2_weights), conv2_biases))

conv3_max = apply_max_pool(conv3, 1, 2, 1, 2)

shape = conv3_max.get_shape().as_list()
cov_flat = tf.reshape(conv3_max, [-1, shape[1] * shape[2] * shape[3]])

f1_weights = weight_variable('f1w', [shape[1] * shape[2] * shape[3], num_hidden])
f1_biases = bias_variable('f1b', [num_hidden])

## Batch Normalization
z1 = tf.add(tf.matmul(cov_flat, f1_weights),f1_biases)
z1_bn = tf.contrib.layers.batch_norm(z1,decay=0.9, center=True,scale=True, is_training=is_train, updates_collections=None)
f1 = tf.nn.relu(z1_bn)
##


f1 = tf.nn.dropout(f1, keep_prob)

out_weights = weight_variable('ow', [num_hidden, num_labels])
out_biases = bias_variable('ob', [num_labels])

## Batch Normalization
z3 = tf.add(tf.matmul(f1, out_weights), out_biases)
z3_bn = tf.contrib.layers.batch_norm(z3,decay=0.9, center=True,scale=True, is_training=is_train, updates_collections=None)
y_ = tf.nn.softmax(z3_bn)
##




cross_entropy = -tf.reduce_sum(Y * tf.log(y_))

reg_conv1 = tf.nn.l2_loss(conv1_weights) + tf.nn.l2_loss(conv1_biases)
reg_conv2 = tf.nn.l2_loss(conv2_weights) + tf.nn.l2_loss(conv2_biases)
reg_conv3 = tf.nn.l2_loss(conv3_weights) + tf.nn.l2_loss(conv3_biases)
reg_f1 = tf.nn.l2_loss(f1_weights) + tf.nn.l2_loss(f1_biases)

reg_out = tf.nn.l2_loss(out_weights) + tf.nn.l2_loss(out_biases)
regularizers = reg_conv1 + reg_conv2 + reg_f1 + reg_out

cross_entropy = tf.reduce_mean(cross_entropy + beta*regularizers)
# optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# cost_history = np.empty(shape=[1],dtype=float)
# accuracy_history = []

saver = tf.train.Saver()

tf.summary.scalar('cross_entropy', cross_entropy)
merged = tf.summary.merge_all()




dirname = 'experiment/'
folder_name = '1sec_aug/'

train_features, train_labels, test_features, test_labels = make_dir_to_npy(dirname+folder_name)

log_dir = 'log/' + folder_name
board_dir = 'tf_board/' + folder_name

try:
    os.makedirs(log_dir)
except:
    pass

accuracy_history = []
cfm = []

train_x, train_y = shuffle_numpy(train_features, train_labels)
test_x, test_y = shuffle_numpy(test_features, test_labels)
test_x = test_x[:1280]
test_y = test_y[:1280]

with tf.Session() as session:

    tf.global_variables_initializer().run()

    train_writer = tf.summary.FileWriter(board_dir + '/train', session.graph)
    test_writer = tf.summary.FileWriter(board_dir + '/test', session.graph)
    
    #num_batch = train_y.shape[0]//128
    maxac = 0
    for i in range(50):
        for j in range(train_features.shape[0]//128):
            batch_x = train_x[128*j:128*(j+1),:,:,:]
            batch_y = train_y[128*j:128*(j+1),:]

            _, train_loss, summary = session.run([optimizer, cross_entropy, merged], feed_dict={X: batch_x, Y: batch_y, keep_prob: dropout_rate, is_train:True})

        #print(train_loss)
        #saver.save(session, log_dir + 'model.ckpt')
        ac = 0
        
        y___ = np.zeros((1280,num_labels))
        for j in range(10):

            y__, accuracy_, val_loss, summary_test = session.run([y_, accuracy, cross_entropy, merged], feed_dict={X: test_x[128*j:128*(j+1)], Y: test_y[128*j:128*(j+1)], keep_prob: 1.0,is_train:False})
            y___[128*j:128*(j+1)] = y__
            ac += accuracy_ / 10
        if ac > maxac:
            maxac = ac
            saver.save(session, log_dir + 'model.ckpt')         


        accuracy_history = np.append(accuracy_history, round(accuracy_, 5))
        print("val_acc: {:.3f} / val_loss: {:.3f}".format(ac, val_loss / 128))
        print(confusion_matrix(np.argmax(test_y, axis=1), np.argmax(y___, axis=1)))




    session.close()
