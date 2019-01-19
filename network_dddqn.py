import tensorflow as tf

state_size = 195 * 3
action_size = 6
learning_rate =  0.0001

class DDDQNNet:
    def __init__(self, name):
        self.name = name

        # We use tf.variable_scope here to know which network we're using (DQN or target_net)
        # it will be useful when we will update our w- parameters (by copy the DQN parameters)
        with tf.variable_scope(self.name):
            self.inputs_ = tf.placeholder(tf.float32, [None, state_size], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, action_size], name="actions_")
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")

            layer_1 = tf.layers.dense(inputs=self.inputs_, units=512, activation=tf.nn.leaky_relu)
            layer_2 = tf.layers.dense(inputs=layer_1, units=256, activation=tf.nn.leaky_relu)
            ## Here we separate into two streams
            # The one that calculate V(s)
            self.value_fc = tf.layers.dense(inputs=layer_2,
                                            units=256,
                                            activation=tf.nn.leaky_relu,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                            name="value_fc")

            self.value = tf.layers.dense(inputs=self.value_fc,
                                         units=1,
                                         activation=None,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name="value")

            # The one that calculate A(s,a)
            self.advantage_fc = tf.layers.dense(inputs=layer_2,
                                                units=256,
                                                activation=tf.nn.leaky_relu,
                                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                name="advantage_fc")

            self.advantage = tf.layers.dense(inputs=self.advantage_fc,
                                             units=action_size,
                                             activation=None,
                                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                             name="advantages")

            # Agregating layer
            # Q(s,a) = V(s) + (A(s,a) - 1/|A| * sum A(s,a'))
            self.output = self.value + tf.subtract(self.advantage, tf.reduce_mean(self.advantage, axis=1, keepdims=True))

            # Q is our predicted Q value.
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)


            self.loss = tf.reduce_mean(tf.squared_difference(self.target_Q, self.Q))
            self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(self.loss)