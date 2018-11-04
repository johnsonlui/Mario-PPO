import gym
import numpy as np
import tensorflow as tf


class Policy_net:
    def __init__(self, name: str, input_shape, num_action, temp=0.1):
        """
        :param name: string
        :param env: gym env
        :param temp: temperature of boltzmann distribution
        """

        # ob_space = env.observation_space
        # act_space = env.action_space

        with tf.variable_scope(name):
            self.obs = tf.placeholder(dtype=tf.float32, shape=[None] + list(input_shape), name='obs')

            with tf.variable_scope('policy_net'):
                conv_1 = tf.layers.conv2d(self.obs, 32, 5, activation=tf.nn.relu)
                conv_1 = tf.layers.max_pooling2d(conv_1, 2, 2)
                conv_2 = tf.layers.conv2d(conv_1, 64, 4, activation=tf.nn.relu)
                conv_2 = tf.layers.max_pooling2d(conv_2, 2, 2)
                conv_3 = tf.layers.conv2d(conv_2, 64, 3, activation=tf.nn.relu)

                fc_1 = tf.layers.flatten(conv_3)
                fc_1 = tf.layers.dense(fc_1, 512)
                self.act_probs = tf.layers.dense(inputs=tf.divide(fc_1, temp), units=num_action, activation=tf.nn.softmax)

            with tf.variable_scope('value_net'):
                conv_1 = tf.layers.conv2d(self.obs, 32, 5, activation=tf.nn.relu)
                conv_1 = tf.layers.max_pooling2d(conv_1, 2, 2)
                conv_2 = tf.layers.conv2d(conv_1, 64, 4, activation=tf.nn.relu)
                conv_2 = tf.layers.max_pooling2d(conv_2, 2, 2)
                conv_3 = tf.layers.conv2d(conv_2, 64, 3, activation=tf.nn.relu)

                fc_1 = tf.layers.flatten(conv_3)
                fc_1 = tf.layers.dense(fc_1, 512)
                self.v_preds = tf.layers.dense(inputs=fc_1, units=1, activation=None)

            self.act_stochastic = tf.multinomial(tf.log(self.act_probs), num_samples=1)
            self.act_stochastic = tf.reshape(self.act_stochastic, shape=[-1])

            self.act_deterministic = tf.argmax(self.act_probs, axis=1)

            self.scope = tf.get_variable_scope().name

    def act(self, obs, stochastic=True):
        if stochastic:
            return tf.get_default_session().run([self.act_stochastic, self.v_preds], feed_dict={self.obs: obs})
        else:
            return tf.get_default_session().run([self.act_deterministic, self.v_preds], feed_dict={self.obs: obs})

    def get_action_prob(self, obs):
        return tf.get_default_session().run(self.act_probs, feed_dict={self.obs: obs})

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
