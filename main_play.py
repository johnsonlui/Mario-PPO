import gym
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

import numpy as np
import tensorflow as tf
from policy_net import Policy_net
from PPO import PPOTrain
import time

import cv2

#ITERATION = int(1e5)
ITERATION = int(1000000)

ENV = 'SuperMarioBros-v0'

IMAGE_SIZE_1 = 27 # remove life bar
IMAGE_SIZE_2 = 32
IMAGE_SIZE_3 = 3
IMAGE_SIZE = (IMAGE_SIZE_2, IMAGE_SIZE_2)

def resize_state(state):
    # Resizing and normalizing the state input
    return cv2.resize(state, IMAGE_SIZE)[IMAGE_SIZE_2 - IMAGE_SIZE_1:, :]/255


def main():
    #env = gym.make('CartPole-v0')
    #env.seed(0)
    env = gym_super_mario_bros.make(ENV)
    env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)
    print(env.action_space, 'action_space', env.observation_space, 'observation_space')

    Policy = Policy_net('policy', (IMAGE_SIZE_1, IMAGE_SIZE_2, IMAGE_SIZE_3), len(SIMPLE_MOVEMENT))
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        obs = env.reset()
        obs = resize_state(obs)

        saver.restore(sess, './model/model.ckpt')

        for iteration in range(ITERATION):  # episode
            # print(iteration)
            run_policy_steps = 0
            x_pos_max = 0
            x_pos_avg = []
            current_life = 3
            last_x_pos = 0
            last_x_pos_repeated_count = 0
            while True:  # run policy RUN_POLICY_STEPS which is much less than episode length
                run_policy_steps += 1
                obs = np.stack([obs]).astype(dtype=np.float32)  # prepare to feed placeholder Policy.obs

                if last_x_pos_repeated_count > 4:
                    act, v_pred = Policy.act(obs=obs, stochastic=True)
                    last_x_pos_repeated_count = 0
                else:
                    act, v_pred = Policy.act(obs=obs, stochastic=False)

                act = np.asscalar(act)

                next_obs, reward, done, info = env.step(act)

                env.render()
                time.sleep(0.01)

                if info['x_pos'] > x_pos_max:
                    x_pos_max = info['x_pos']

                if info['x_pos'] == last_x_pos:
                    last_x_pos_repeated_count += 1

                if current_life != info['life']:
                    current_life = info['life']
                    x_pos_avg.append(last_x_pos)
                last_x_pos = info['x_pos']

                #if done or info['flag_get']:
                if done:
                    print("Episode", iteration, ", Game Reward:", info['score'], ", Max Distance:", x_pos_max, ", Avg Distance:", np.mean(x_pos_avg))
                    obs = env.reset()
                    obs = resize_state(obs)
                    reward = -1
                    break
                else:
                    obs = next_obs
                    obs = resize_state(obs)


if __name__ == '__main__':
    main()