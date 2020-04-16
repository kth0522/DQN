import numpy as np
import tensorflow as tf
from statistics import mean
from collections import deque


class Network(tf.keras.Model):
    '''
    Base network
    '''
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
    '''
    Model for the target network and local network
    '''
    def __init__(self, env, multistep, n_step):
        # Environment info
        self.env = env
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n

        # Parameter for multi step
        self.multistep = multistep
        self.n_step = n_step

        # Hyperparameter
        self.batch_size = 32
        self.learning_rate = 1e-4
        self.gamma = 0.99

        # Network
        self.optimizer = tf.optimizers.Adam(self.learning_rate)
        self.hidden_units = [200, 200]
        self.network = Network(self.state_size, self.hidden_units, self.action_size)

        # Experience replay buffer
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}
        self.max_experiences = 10000
        self.min_experiences = 100

    # Predict the value with state using network
    def predict(self, state):
        return self.network(np.atleast_2d(state.astype('float32')))

    # Get action with state by epsilon-greedy algorithm
    def get_action(self, states, epsilon):
        if np.random.random() < epsilon:
            return np.random.choice(self.action_size)
        else:
            return np.argmax(self.predict(np.atleast_2d(states))[0])

    # Take mini batch and update the policy
    def train_minibatch(self, TargetNet):
        # If there is not enough experiences in the experience replay buffer, just return 0
        if len(self.experience['s']) < self.min_experiences:
            return 0
        # Multistep DQN
        if self.multistep:
            # Sampling the mini batch index (maximum ids is smaller than full length, for preventing out of index error)
            multi_ids = np.random.randint(low=0, high=max(len(self.experience['s'])-self.n_step+1, 0),\
                                          size=self.batch_size)
            # Lists of each sample's state, action, and reward
            states = np.asarray([self.experience['s'][i] for i in multi_ids])
            actions = np.asarray([self.experience['a'][i] for i in multi_ids])
            rewards = np.asarray([self.experience['r'][i] for i in multi_ids])
            # Information for n-th forward state from above state
            end_step_states = np.asarray([self.experience['s2'][i+self.n_step-1] for i in multi_ids])
            end_step_value = np.max(TargetNet.predict(end_step_states), axis=1)
            dones = np.asarray([self.experience['done'][i+self.n_step-1] for i in multi_ids])
            # Temp gamma for iteration
            gamma = self.gamma
            # Calculate multi step rewards
            for i in range(1, self.n_step):
                step_reward = np.asarray([self.experience['r'][j+i] for j in multi_ids]) * gamma
                rewards += step_reward
                gamma *= self.gamma
            # Calculate q-target
            q_target = np.where(dones, rewards, rewards + gamma * end_step_value)
        # Original DQN (equivalent to n=1 step DQN)
        else:
            # Sampling the mini batch index
            ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size)
            # Lists of each sample's state, action, reward, and done
            states = np.asarray([self.experience['s'][i] for i in ids])
            actions = np.asarray([self.experience['a'][i] for i in ids])
            rewards = np.asarray([self.experience['r'][i] for i in ids])
            dones = np.asarray([self.experience['done'][i] for i in ids])
            # Information for next states
            states_next = np.asarray([self.experience['s2'][i] for i in ids])
            value_next = np.max(TargetNet.predict(states_next), axis=1)
            # Calculate q-target
            q_target = np.where(dones, rewards, rewards + self.gamma * value_next)
        # Q-prediction get from wS_t
        with tf.GradientTape() as tape:
            selected_action_values = tf.math.reduce_sum(
                self.predict(states) * tf.one_hot(actions, self.action_size), axis=1)
            # Calculate loss
            loss = tf.math.reduce_mean(tf.square(q_target - selected_action_values))
        # Updating gradients
        variables = self.network.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients((zip(gradients, variables)))

        return loss

    # Add new experience to the experience replay buffer
    def add_experience(self, exp):
        if len(self.experience['s']) >= self.max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0)
        for key, value in exp.items():
            self.experience[key].append(value)

    # Copy the network's weight to another network
    def copy_weights(self, LocalNet):
        variables1 = self.network.trainable_variables
        variables2 = LocalNet.network.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())


class DQN:
    '''
    DQN consists of target network and local network
    '''
    def __init__(self, env, multistep=False):
        # Environment info
        self.env = env
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n

        # Hyperparameter
        self.epsilon = 0.1
        self.min_epsilon = 0
        self.epsilon_decay = 0.9999
        self.copy_step = 3

        # Parameter for multi step
        self.multistep = multistep
        self.n_steps = 3

    # Build target network and local network
    def _build_network(self):
        self.TargetNet = Model(self.env, self.multistep, self.n_steps)
        self.LocalNet = Model(self.env, self.multistep, self.n_steps)

    # Play the one episode
    def play_game(self):
        rewards = 0
        step_count = 0
        done = False
        observations = self.env.reset()

        losses = list()

        while not done:
            action = self.LocalNet.get_action(observations, self.epsilon)
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
            if step_count % self.copy_step == 0:
                self.TargetNet.copy_weights(self.LocalNet)
        return rewards, mean(losses), step_count

    # Updating epsilon
    def update_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    # Training the DQN
    def learn(self, max_episode=1500):
        self._build_network()

        avg_step_count_list = []
        last_100_episode_step_count = deque(maxlen=100)
        total_rewards = np.empty(max_episode)

        for episode in range(max_episode):
            self.update_epsilon()

            total_reward, losses, step_count = self.play_game()
            total_rewards[episode] = total_reward
            last_100_episode_step_count.append(step_count)
            avg_step_count = np.mean(last_100_episode_step_count)

            # Print progress
            print("[Episode {:>5}]  episode step_count: {:>5} avg step_count: {}".format(episode, step_count, avg_step_count))

            avg_step_count_list.append(avg_step_count)

            # If average step count exceed 475, stop early
            if avg_step_count >= 475:
                break

        return avg_step_count_list