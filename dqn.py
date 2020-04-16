import sys
import numpy as np
import tensorflow as tf
import random
import gym
import copy
import os
from gym import wrappers
from statistics import mean
from collections import deque


class Network(tf.keras.Model):
    def __init__(self, num_states, hidden_units, num_actions):
        super(Network, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(num_states,))
        self.hidden_layers = []
        for i in hidden_units:
            self.hidden_layers.append(tf.keras.layers.Dense(
                i, activation='relu', kernel_initializer='RandomNormal'))
            self.output_layer = tf.keras.layers.Dense(
                num_actions, activation='linear', kernel_initializer='RandomNormal'
            )

    @tf.function
    def call(self, inputs):
        z = self.input_layer(inputs)
        for layer in self.hidden_layers:
            z = layer(z)
        output = self.output_layer(z)
        return output

class Model:
    def __init__(self, env, multistep, n_step):
        self.env = env
        self.multistep = multistep
        self.n_step = n_step
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.batch_size = 32
        self.learning_rate = 1e-4
        self.optimizer = tf.optimizers.Adam(self.learning_rate)
        self.gamma = 0.99
        self.hidden_units = [200, 200]
        self.network = Network(self.state_size, self.hidden_units, self.action_size)
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}
        self.max_experiences = 10000
        self.min_experiences = 100

    def predict(self, state):
        return self.network(np.atleast_2d(state.astype('float32')))


    def get_action(self, states, epsilon):
        if np.random.random() < epsilon:
            return np.random.choice(self.action_size)
        else:
            return np.argmax(self.predict(np.atleast_2d(states))[0])

    def train_minibatch(self, TargetNet):
        # mini batch를 받아 policy를 update

        if len(self.experience['s']) < self.min_experiences:
            return 0

        # sample experience

        if self.multistep:
            multi_ids = np.random.randint(low=0, high=max(len(self.experience['s'])-self.n_step+1, 0), size=self.batch_size)
            states = np.asarray([self.experience['s'][i] for i in multi_ids])
            actions = np.asarray([self.experience['a'][i] for i in multi_ids])

            end_step_states = np.asarray([self.experience['s2'][i+self.n_step-1] for i in multi_ids])
            end_step_value = np.max(TargetNet.predict(end_step_states), axis=1)
            rewards = np.asarray([self.experience['r'][i] for i in multi_ids])
            dones = np.asarray([self.experience['done'][i+self.n_step-1] for i in multi_ids])

            gamma = self.gamma
            for i in range(1, self.n_step):
                step_reward = copy.deepcopy(np.asarray([self.experience['r'][j+i] for j in multi_ids])) * gamma
                rewards += step_reward
                gamma *= self.gamma


            actual_values = np.where(dones, rewards, rewards+gamma*end_step_value)

        else:
            ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size)
            states = np.asarray([self.experience['s'][i] for i in ids])
            actions = np.asarray([self.experience['a'][i] for i in ids])

            rewards = np.asarray([self.experience['r'][i] for i in ids])
            states_next = np.asarray([self.experience['s2'][i] for i in ids])
            dones = np.asarray([self.experience['done'][i] for i in ids])
            value_next = np.max(TargetNet.predict(states_next), axis=1)
            actual_values = np.where(dones, rewards, rewards+self.gamma*value_next)

        with tf.GradientTape() as tape:
            selected_action_values = tf.math.reduce_sum(
                self.predict(states) * tf.one_hot(actions, self.action_size), axis=1)
            loss = tf.math.reduce_mean(tf.square(actual_values - selected_action_values))
        variables = self.network.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients((zip(gradients, variables)))

        return loss

    def add_experience(self, exp):
        if len(self.experience['s']) >= self.max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0)
        for key, value in exp.items():
            self.experience[key].append(value)

    def copy_weights(self, TrainNet):
        variables1 = self.network.trainable_variables
        variables2 = TrainNet.network.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())


class DQN:
    def __init__(self, env, multistep=False):
        self.env = env
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.multistep = multistep  # Multistep(n-step) 구현 시 True로 설정, 미구현 시 False
        self.n_steps = 3           # Multistep(n-step) 구현 시 n 값, 수정 가능

    def _build_network(self):
        # Target 네트워크와 Local 네트워크를 설정
        self.TargetNet = Model(self.env, self.multistep, self.n_steps)
        self.LocalNet = Model(self.env, self.multistep, self.n_steps)


    def play_game(self, epsilon, copy_step):
        rewards = 0
        step_count = 0
        done = False
        observations = self.env.reset()
        experience_buffer = deque(maxlen=self.n_steps)
        losses = list()

        while not done:
            action = self.LocalNet.get_action(observations, epsilon)
            prev_observations = observations
            observations, reward, done, _ = self.env.step(action)
            rewards += reward
            if done:
                reward = -250
                self.env.reset()

            exp = {'s': prev_observations, 'a': action, 'r':reward, 's2': observations, 'done': done}
            self.LocalNet.add_experience(exp)
            loss = self.LocalNet.train_minibatch(self.TargetNet)
            if isinstance(loss, int):
                losses.append(loss)
            else:
                losses.append(loss.numpy())
            step_count += 1
            if step_count % copy_step == 0:
                self.TargetNet.copy_weights(self.LocalNet)
        return rewards, mean(losses), step_count

    def make_video(self):
        env = wrappers.Monitor(self.env, os.path.join(os.getcwd(), "videos"), force=True)
        rewards = 0
        steps = 0
        done = False
        observation = self.env.reset()
        while not done:
            self.env.render()
            action = self.LocalNet.get_action(observation, 0)
            observation, reward, done, _ = self.env.step(action)
            steps += 1
            rewards += reward
        print("Testing steps: {} rewards {}: ".format(steps, rewards))


    # episode 최대 횟수는 구현하는 동안 더 적게, 더 많이 돌려보아도 무방합니다.
    # 그러나 평가시에는 episode 최대 회수를 1500 으로 설정합니다.
    def learn(self, max_episode=1500):
        avg_step_count_list = []     # 결과 그래프 그리기 위해 script.py 로 반환
        last_100_episode_step_count = deque(maxlen=100)

        _ = self._build_network()

        total_rewards = np.empty(max_episode)
        copy_step = 4
        epsilon = 0.1
        decay = 0.9999
        min_epsilon = 0

        for episode in range(max_episode):
            epsilon = max(min_epsilon, epsilon * decay)
            total_reward, losses, step_count = self.play_game(epsilon, copy_step)
            total_rewards[episode] = total_reward

            last_100_episode_step_count.append(step_count)

            # 최근 100개의 에피소드 평균 step 횟수를 저장 (이 부분은 수정하지 마세요)
            avg_step_count = np.mean(last_100_episode_step_count)
            print("[Episode {:>5}]  episode step_count: {:>5} avg step_count: {}".format(episode, step_count, avg_step_count))

            avg_step_count_list.append(avg_step_count)


            if avg_step_count >= 475:
                break

        return avg_step_count_list