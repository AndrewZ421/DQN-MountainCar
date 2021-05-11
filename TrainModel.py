import gym
import os
import random
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam    
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt

FILE_PATH = 'dqn_300.h5'
ENV_NAME = 'MountainCar-v0'
NUM_EPISODES = 1000
GAMMA = 0.95
NUM_BATCH = 64
UPDATE_FREQ = 200
MEMORY_SIZE = 2000
STATE_DIM = 2
ACTION_SPACE = 3
EPSILON = 1
EPSILON_DCEY = 0.99
EPSILON_MIN = 0.01
LEARNING_RATE = 0.5

class Agent:
    def __init__(self):
        self.step_count = 0
        self.epsilon = EPSILON
        self.env = gym.make(ENV_NAME)
        self.update_freq = UPDATE_FREQ
        self.gamma = GAMMA
        self.memory_size = MEMORY_SIZE
        self.memory_buffer = deque(maxlen=self.memory_size)

        self.model = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        inputs = Input(shape=(STATE_DIM,))
        x = Dense(100, activation='relu')(inputs)
        x = Dense(ACTION_SPACE, activation='linear')(x)

        model = Model(inputs=inputs, outputs=x)
        model.compile(loss='mean_squared_error', optimizer=Adam(0.001))

        return model

    def update_epsilon(self):
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DCEY

    def run(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice([0, 1, 2])
        return np.argmax(self.model.predict(np.array([state]))[0])

    def save_model(self):
        self.model.save(FILE_PATH)
        print('Model Saved!')

    def remember(self, state, action, reward, next_state, done):
        item = (state, action, reward, next_state, done)
        self.memory_buffer.append(item)

    def process_batch(self):
        if len(self.memory_buffer) < self.memory_size:
            return
        
        self.step_count += 1

        if self.step_count % self.update_freq == 0:
            self.target_model.set_weights(self.model.get_weights())

        data = random.sample(self.memory_buffer, NUM_BATCH)
        states = np.array([d[0] for d in data])
        next_states = np.array([d[3] for d in data])

        y = self.model.predict(states)
        q = self.target_model.predict(next_states)

        for i, (_, action, reward, _, done) in enumerate(data):
            if not done:
                y[i][action] = (1-LEARNING_RATE) * y[i][action] + LEARNING_RATE * (reward + self.gamma * np.amax(q[i]))
        
        self.update_epsilon()
        self.model.fit(states, y, verbose=0)

    def train(self):
        score_list = []
        for i in range(NUM_EPISODES):

            state = self.env.reset()
            score = 0
            done = False

            while not done:
                # self.env.render()
                action = self.run(state)
                next_state, reward, done, _ = self.env.step(action)
                reward = abs(next_state[0] - (-0.5))
                if next_state[0] >= 0.5:
                    reward += 300 
                score += reward
                self.remember(state, action, reward, next_state, done)
                self.process_batch()
                state = next_state

            score_list.append(score)
            print('episode:', i, 'score:', score)
            # print('episode:', i, 'score:', score, 'max:', max(score_list))

            if np.min(score_list[-10:]) > 300:
                self.save_model()
                break
        # plt.plot(score_list,'o-',color = 'orange')
        # plt.xlabel("Episode")
        # plt.ylabel("Score")
        # plt.show()


if __name__ == '__main__':
    model = Agent()
    model.train()