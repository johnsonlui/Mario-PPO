import gym
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

import numpy as np
import tensorflow as tf
from policy_net import Policy_net
from PPO import PPOTrain

import cv2

#ITERATION = int(1e5)
ITERATION = int(1000000)
GAMMA = 0.95

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
    Old_Policy = Policy_net('old_policy', (IMAGE_SIZE_1, IMAGE_SIZE_2, IMAGE_SIZE_3), len(SIMPLE_MOVEMENT))
    PPO = PPOTrain(Policy, Old_Policy, gamma=GAMMA)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        writer = tf.summary.FileWriter('./log/train', sess.graph)
        sess.run(tf.global_variables_initializer())
        obs = env.reset()
        obs = resize_state(obs)
        reward = 0
        success_num = 0
        game_reward_max = 0

        saver.restore(sess, './model/model.ckpt')

        for iteration in range(ITERATION):  # episode
            # print(iteration)
            observations = []
            actions = []
            v_preds = []
            rewards = []
            run_policy_steps = 0
            x_pos_max = 0
            x_pos_avg = []
            current_life = 3
            last_x_pos = 0
            get_flag_num = 0
            while True:  # run policy RUN_POLICY_STEPS which is much less than episode length
                run_policy_steps += 1
                obs = np.stack([obs]).astype(dtype=np.float32)  # prepare to feed placeholder Policy.obs
                act, v_pred = Policy.act(obs=obs, stochastic=True)

                act = np.asscalar(act)
                v_pred = np.asscalar(v_pred)

                observations.append(obs)
                actions.append(act)
                v_preds.append(v_pred)
                rewards.append(reward)

                next_obs, reward, done, info = env.step(act)
                # env.render()

                if info['x_pos'] > x_pos_max:
                    x_pos_max = info['x_pos']

                if current_life != info['life']:
                    current_life = info['life']
                    x_pos_avg.append(last_x_pos)
                last_x_pos = info['x_pos']




                if done:
                    v_preds_next = v_preds[1:] + [0]  # next state of terminate state has 0 state value
                    print("Episode", iteration, ", Game Reward:", info['score'], ", Max Distance:", x_pos_max, ", Avg Distance:", np.mean(x_pos_avg))
                    saver.save(sess, './model/model.ckpt')
                    if info['score'] > game_reward_max:
                        saver.save(sess, './model/model_best_score.ckpt')
                        game_reward_max = info['score']

                    obs = env.reset()
                    obs = resize_state(obs)
                    reward = -1
                    break
                else:
                    if info['flag_get']:
                        get_flag_num += 1
                        if get_flag_num > 2:
                            v_preds_next = v_preds[1:] + [0]  # next state of terminate state has 0 state value
                            print("Episode", iteration, ", Game Reward:", info['score'], ", Max Distance:", x_pos_max,
                                  ", Avg Distance:", np.mean(x_pos_avg))
                            saver.save(sess, './model/model.ckpt')
                            if info['score'] > game_reward_max:
                                saver.save(sess, './model/model_best_score.ckpt')
                                game_reward_max = info['score']

                            obs = env.reset()
                            obs = resize_state(obs)
                            reward = -1
                            break
                        obs = env.reset()
                        obs = resize_state(obs)
                    else:
                        obs = next_obs
                        obs = resize_state(obs)

            writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_length', simple_value=run_policy_steps)])
                               , iteration)
            writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_reward', simple_value=sum(rewards))])
                               , iteration)

            '''
            if sum(rewards) >= 195:
                success_num += 1
                if success_num >= 100:
                    saver.save(sess, './model/model.ckpt')
                    print('Clear!! Model saved.')
                    break
            else:
                success_num = 0
            '''

            gaes = PPO.get_gaes(rewards=rewards, v_preds=v_preds, v_preds_next=v_preds_next)

            # convert list to numpy array for feeding tf.placeholder
            observations = np.reshape(observations, newshape=[-1] + list((IMAGE_SIZE_1, IMAGE_SIZE_2, IMAGE_SIZE_3)))
            actions = np.array(actions).astype(dtype=np.int32)
            rewards = np.array(rewards).astype(dtype=np.float32)
            v_preds_next = np.array(v_preds_next).astype(dtype=np.float32)
            gaes = np.array(gaes).astype(dtype=np.float32)
            gaes = (gaes - gaes.mean()) / gaes.std()

            PPO.assign_policy_parameters()

            inp = [observations, actions, rewards, v_preds_next, gaes]

            # train
            for epoch in range(10):
                sample_indices = np.random.randint(low=0, high=observations.shape[0], size=64)  # indices are in [low, high)
                sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]  # sample training data
                PPO.train(obs=sampled_inp[0],
                          actions=sampled_inp[1],
                          rewards=sampled_inp[2],
                          v_preds_next=sampled_inp[3],
                          gaes=sampled_inp[4])

            summary = PPO.get_summary(obs=inp[0],
                                      actions=inp[1],
                                      rewards=inp[2],
                                      v_preds_next=inp[3],
                                      gaes=inp[4])[0]

            writer.add_summary(summary, iteration)
            # print("Sum Reward:", np.sum(rewards))
        writer.close()


if __name__ == '__main__':
    main()