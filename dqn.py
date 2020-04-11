import sys
import numpy as np
import tensorflow as tf
import random
import gym
from collections import deque


class DQN:
    def __init__(self, env, multistep=False):

        self.env = env
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.multistep = multistep  # Multistep(n-step) 구현 시 True로 설정, 미구현 시 False
        self.n_steps = 4            # Multistep(n-step) 구현 시 n 값, 수정 가능

    def _build_network(self, ):
        # Target 네트워크와 Local 네트워크를 설정
        self.scalarInput = tf.place
        pass

    def predict(self, state):
        # state를 넣어 policy에 따라 action을 반환
        return self.env.action_space.sample()

    def train_minibatch(self, ):
        # mini batch를 받아 policy를 update
        pass

    def update_epsilon(self, ):
        # Exploration 시 사용할 epsilon 값을 업데이트
        pass


    # episode 최대 횟수는 구현하는 동안 더 적게, 더 많이 돌려보아도 무방합니다.
    # 그러나 평가시에는 episode 최대 회수를 1500 으로 설정합니다.
    def learn(self, max_episode=1500):
        avg_step_count_list = []     # 결과 그래프 그리기 위해 script.py 로 반환
        last_100_episode_step_count = deque(maxlen=100)

        for episode in range(max_episode):
            done = False
            state = self.env.reset()
            step_count = 0


            # episode 시작
            while not done:
                action = self.predict(state)
                next_state, reward, done, _ = self.env.step(action)


                state = next_state
                step_count += 1

            last_100_episode_step_count.append(step_count)


            # 최근 100개의 에피소드 평균 step 횟수를 저장 (이 부분은 수정하지 마세요)
            if len(last_100_episode_step_count) == 100:
                avg_step_count = np.mean(last_100_episode_step_count)
                print("[Episode {:>5}]  episode step_count: {:>5} avg step_count: {}".format(episode, step_count, avg_step_count))

                avg_step_count_list.append(avg_step_count)

        return avg_step_count_list
