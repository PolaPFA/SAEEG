import tensorflow as tf

def define_placeholders(input, output):
    inshape = input.shape[1]
    outshape = output.shape[1]
    X = tf.placeholder(tf.float32, [None, inshape, 1])
    Y = tf.placeholder(tf.float32, [None, outshape, 1])
    placeholders = {'X': X, 'Y': Y}
    return placeholders

def weight_init():
    W1 = tf.get_variable("W1", shape=[1, 5, 1, 32])
    W2 = tf.get_variable("W2", shape=[1, 3, 32, 24])
    W3 = tf.get_variable("W3", shape=[1, 3, 24, 16])
    W4 = tf.get_variable("W4", shape=[1, 5, 16, 8])
    return {'W1': W1, 'W2': W2, 'W3': W3, 'W4': W4}

#CONVBlock
def conv_block(input, weight, stride, padding = 'SAME'):
    Z1 = tf.nn.conv1d(input, weight, stride, padding)
    Z1 = tf.nn.batch_normalization(Z1)
    return tf.nn.relu(Z1)

def model():
    out1 = conv_block(X, W1, 3, 'VALID')
    out2 = conv_block(out1, W2, 2, 'VALID')
    out3 = conv_block(out2, W3, 2, 'VALID')
    out4 = conv_block(out3, W4, 3, 'VALID')
    FC1in = tf.contrib.layers.flatten(out4)
    DOin = tf.contrib.layers.fully_connected(FC1in, 20, activation_function='relu')
    FC2in = tf.nn.dropout(DOin, keep_prob=0.5)
    SMin = tf.contrib.layers.fully_connected(FC2in, 2, activation_function='relu')
    return tf.nn.softmax(SMin)

def compute_cost(X, Y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=X, labels=Y))
    return cost

output = model()
cost = compute_cost(output, Y)
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
init = tf.global_variables_initializer()
number_of_epochs = 200
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(number_of_epochs):
        _, com_cost = sess.run(fetches=[optimizer, cost], feed_dict=[X_train, Y_train])
        print("Epoch number "+ epoch + "\t Cost: " + com_cost)