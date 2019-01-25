import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import librosa
import os
import librosa.display

dirname = 'D:\AMD\DrumSample'
classes = ['Clhat', 'Kick', 'Snare', 'Tom']
folders = ['training', 'testing']

dict_class = {'Clhat': [1., 0., 0., 0.],
             'Kick': [0., 1., 0., 0.],
             'Snare': [0., 0., 1., 0.],
             'Tom': [0., 0., 0., 1.]}

sr = 22050
bands = 128
# temp = []

for classe in classes:
    for folder in folders:
        file_lists = glob.glob(os.path.join(dirname, classe, folder, '*.wav'))
        log_spectrograms = []
        labels = []
        for file_list in file_lists:
            y, sr = librosa.load(file_list, sr)
            if np.size(y) < 20000:
                y = np.lib.pad(y, (0, 20000-np.size(y)), 'constant', constant_values = (0, 0)) # zero padding for y shorter than length 20000
            else:
                y = y[:20000]
            # temp.append(np.size(y))
                
            normalization_factor = 1/np.max(np.abs(y))
            y = y * normalization_factor
            mel_spec = librosa.feature.melspectrogram(y, sr=sr, n_fft = 1024, hop_length = 512, n_mels = bands)
            log_mel_spec = librosa.power_to_db(mel_spec, ref = np.max)
            # print(np.shape(log_mel_spec)) -> (128, 40)
            log_spectrograms.append(log_mel_spec)
            # labels.append(label)
            label = dict_class[classe]
            labels.append(label)
            
        np.save(os.path.join(dirname, classe, '%s_%s_features.npy' % (classe, folder)), log_spectrograms)
        np.save(os.path.join(dirname, classe, '%s_%s_labels.npy' % (classe, folder)), labels)



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
    initial = tf.contrib.layers.xavier_initializer() # weight initialized
    return tf.get_variable(name = name, shape = shape, initializer = initial)

def bias_variable(name, shape):
    initial = tf.contrib.layers.xavier_initializer() # bias initialized
    return tf.get_variable(name = name, shape = shape, initializer = initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = 'SAME')

def apply_convolution(x, shape1, shape2, num_channels, depth):
    weights = weight_variable([shape1, shape2, num_channels, depth])
    biases = bias_variable([depth])
    return tf.nn.relu(tf.add(conv2d(x, weights), biases))
    # convolution: y= Wx+b -> Activation Function

def apply_max_pool(x, shape1, shape2, stride_size1, stride_size2):
    return tf.nn.max_pool(x, ksize = [1, shape1, shape2, 1], strides = [1, stride_size1, stride_size2, 1], padding = 'SAME')
    # ksize(the kernel size) will typically be [1,2,2,1] if I have a 2x2 window over which I take the MAXimum.
    # On the batch size dimension and the channels dimension, ksize is 1 because we don't want to take the MAX over multiple examples, or over multiple channels.


#def model(X, W1, W2, W3, W4, W_o, keep_prob):
    #l1a = tf.nn.relu(tf.nn.conv2d(X, W1, strides = [1,1,1,1], padding = 'SAME'))
        # tf.nn.conv2d(input, filter, strides, padding) -> W1 acts like a filter
        # Within strides, [0] and [3] are always 1 / [1] and [2] are normally same value
        # padding = 'SAME' makes the output the same size as the input
        # l1a shape = (?, 128, 40, 32)
        #l1 = tf.nn.max_pool(l1a, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
        # ksize = [1,2,2,1] means moving 2 length at a time 
        # l1 shape = (?, 64, 20, 32)
        #l1 = tf.nn.dropout(l1, keep_prob)

        #l2a = tf.nn.relu(tf.nn.conv2d(l1, W2, strides = [1,1,1,1], padding = 'SAME')
        # l2a shape = (?, 64, 20, 64)
        #l2 = tf.nn.max_pool(l2a, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
        # l2 shape = (?, 32, 10, 64)
        #l2 = tf.nn.dropout(l2, keep_prob)

        #l3a = tf.nn.relu(tf.nn.conv2d(l2, W3, strides = [1,1,1,1], padding = 'SAME'))
        # l3a shape = (?, 32, 10, 128)
        #l3 = tf.nn.max_pool(l3a, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
        # l3 shape = (?, 16, 5, 128)
        #l3 = tf.reshape(l3, [-1, W4.get_shape().as_list()[0]])
        # reshape to (?, 16*5*128)
        #l3 = tf.nn.dropout(l3, keep_prob)

        #l4 = tf.nn.relu(tf.matmul(l3, W4))
        #l4 = tf.nn.dropout(l4, keep_prob)

        #pyx = tf.matmul(l4, W_o)
        #return pyx

# train_X = 
# train_Y = 
# test_X = 
# test_Y =      import npy files 

bands = 128
frames = 40

num_labels = 4 # 4 Characteristics
num_channels = 1 # If it contains delta value, it will become 2

batch_size = 20

filter_num = 64  # number of the filters
num_hidden = 512

learning_rate = 0.001
dropout_rate = 0.8
beta = 0.001

# input: 128 x 40 data
X = tf.placeholder(tf.float32, shape = [None, bands, frames, num_channels], name = "X")
Y = tf.placeholder(tf.float32, shape = [None, num_labels], name = "Y")
is_train = tf.placeholder(tf.bool, name = "is_train")
keep_prob = tf.placeholder("float")

conv1_weights = weight_variable('c1w', [64, 6, num_channels, filter_num])
conv1_biases = bias_variable('c1b', [filter_num])

### Batch Normalization ###
conv1 = tf.add(conv2d(X, conv1_weights), conv1_biases)
conv1 = tf.nn.relu(tf.contrib.layers.batch_norm(conv1, decay = 0.9, center = True, scale = True, is_training = is_train, updates_collection = None))
###########################
conv1 = tf.nn.dropout(conv1, keep_prob)
conv1_max = apply_max_pool(conv1, 2, 2, 2, 2)
###########################

conv2_weights = weight_variable('c2w', [1, 3, filter_num, filter_num])
conv2_biases = bias_variable('c2b', [filter_num])

### Batch Normalization ###
conv2 = tf.add(conv2d(conv1_max, conv2_weights), conv2_biases)
conv2 = tf.nn.relu(tf.contrib.layers.batch_norm(conv2, decay = 0.9, center = True, scale = True, is_training = is_train, updates_collection = None))
###########################
conv2 = tf.nn.dropout(conv2, keep_prob)
conv2_max = apply_max_pool(conv2, 2, 2, 2, 2)
###########################

conv3_weights = weight_variable('c3w', [1, 3, filter_num, filter_num])
conv3_biases = bias_variable('c3b', [filter_num])

### Batch Normalization ###
conv3 = tf.add(conv2d(conv2_max, conv3_weights), conv3_biases)
conv3 = tf.nn.relu(tf.contrib.layers.batch_norm(conv3, decay = 0.9, center = True, scale = True, is_training = is_train, updates_collection = None))
###########################
conv3 = tf.nn.dropout(conv3, keep_prob)
conv3_max = apply_max_pool(conv3, 2, 2, 2, 2)
###########################


shape = conv3_max.get_shape().as_list()
cov_flat = tf.reshape(conv3_max, [-1, shape[1]*shape[2]*shape[3]])

### Fully Connected ###
f1_weights = weight_variable('f1w', [shape[1]*shape[2]*shape[3], num_hidden])
f1_biases = bias_variable('f1b', [num_hidden])

### Batch Normalization ###
z1 = tf.add(tf.matumul(cov_flat, f1_weights), f1_biases) # matmul function at FC layer
f1 = tf.nn.relu(tf.contrib.layers.batch_norm(z1, decay = 0.9, center = True, scale = True, is_training = is_train, updates_collection = None))
###########################
f1 = tf.nn.dropout(f1, keep_prob)


out_weights = weight_variable('ow', [num_hidden, num_labels])
out_biases = bias_variable('ob', [num_labels])

### Batch Normalization ###
z2 = tf.add(tf.matmul(f1, out_weights), out_biases)
y_ = tf.nn.softmax(tf.contrib.layers.batch_norm(z2, decay = 0.9, center = True, scale = True, is_training = is_train, updates_collection = None))



###################################################################################
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


saver = tf.train.Saver()

tf.summary.scalar('cross_entropy', cross_entropy)
merged = tf.summary.merge_all()






###################################################################################
W1 = weight_variable('W1', [3, 3, 1, 32]) # (3x3x1) filter -> 32 filters
W2 = weight_variable('W2', [3, 3, 32, 64])
W3 = weight_variable('W3', [3, 3, 64, 128])
W4 = weight_variable('W4', [128 * 16 * 5, 625]) # FC 128*16*5 inputs, 625 outputs
W_o = weight_variable('W_o', [625, 4]) # 4 labels = kick / snare / hi-hat / tom
###################################################################################

train_x, train_y = shuffle_numpy(train_features, train_labels)
test_x, test_y = shuffle_numpy(test_features, test_labels)
test_x = test_x[:400]
test_y = test_y[:400]

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for i in range(10):
        for j in range(train_features.shape[0]//20):
            batch_x = train_x[20*j:20*(j+1),:,:,:]
            batch_y = train_y[20*j:20*(j+1),:]

            _, train_loss, summary = session.run([optimizer, cross_entropy, mer])



