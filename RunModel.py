import time
import gym
import numpy as np
from tensorflow.keras import models

# create env
env = gym.make('MountainCar-v0')
# load trained model
model = models.load_model('dqn_300.h5')

# reset env
s = env.reset()
while True:
    env.render()
    time.sleep(0.01)
    a = np.argmax(model.predict(np.array([s]))[0])
    s, reward, done, _ = env.step(a)
    if done:
        break
env.close()