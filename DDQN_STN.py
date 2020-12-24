import random
import gym
import numpy as np
from collections import deque
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

INFO_MESSAGE = "Episode: {a}/{b}, Score: {c}, Reward: {d}"
GAMMA = 0.95
TRAINING_DATA_SIZE = 1000
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.999
EPISODES = 1000
LEARNING_RATE = 0.0001
BATCH_SIZE = 32
TAU = 0.1
INIT = 'he_uniform'


class DDQNAgent:
    def __init__(self, state_size, action_size, mode=None):
        self.mode = mode
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.epsilon = 1.0
        self.ddqn = True
        self.soft_target_network = False
        self.model = create_model((self.state_size,), self.action_size)
        self.target_model = create_model((self.state_size,), self.action_size)

    def update_target_model(self):
        if not self.mode:
            self.target_model.set_weights(self.model.get_weights())
            return

        if self.mode == 'soft':
            q_model_theta = self.model.get_weights()
            target_model_th = self.target_model.get_weights()
            counter = 0
            for q_weight, target_weight in zip(q_model_theta, target_model_th):
                target_weight = target_weight * (1 - TAU) + q_weight * TAU
                target_model_th[counter] = target_weight
                counter += 1
            self.target_model.set_weights(target_model_th)

    def act(self, state):
        return (random.randrange(self.action_size)
                if np.random.random() < self.epsilon
                else np.argmax(self.model.predict(state)))

    def play_with_sample(self):
        if TRAINING_DATA_SIZE >= len(self.memory):
            return

        batch = random.sample(self.memory, BATCH_SIZE)
        state = np.zeros((BATCH_SIZE, self.state_size))
        next_state = np.zeros((BATCH_SIZE, self.state_size))
        action, reward, done = [], [], []

        for i in range(BATCH_SIZE):
            state[i] = batch[i][0]
            action.append(batch[i][1])
            reward.append(batch[i][2])
            next_state[i] = batch[i][3]
            done.append(batch[i][4])

        target = self.model.predict(state)
        target_next = self.model.predict(next_state)
        target_val = self.target_model.predict(next_state)

        for i in range(len(batch)):
            Bellman_equation = reward[i] + GAMMA * (target_val[i][np.argmax(target_next[i])])
            target[i][action[i]] = reward[i] if done[i] else Bellman_equation

        self.model.fit(state, target, batch_size=BATCH_SIZE, verbose=0)
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)


def create_model(input_dim, action_space):
    current_input = Input(input_dim)
    n = current_input
    n = Dense(512, input_shape=input_dim, activation="relu",
              kernel_initializer=INIT)(n)
    n = Dense(256, activation="relu", kernel_initializer=INIT)(n)
    n = Dense(64, activation="relu", kernel_initializer=INIT)(n)
    n = Dense(action_space, activation="linear", kernel_initializer=INIT)(n)

    model = Model(inputs=current_input, outputs=n)
    model.compile(loss="mean_squared_error", optimizer=Adam(lr=LEARNING_RATE),
                  metrics=["accuracy"])
    model.summary()

    return model


def print_msg(*args):
    a, b, c, d = args[0], args[1], args[2], args[3]
    print(INFO_MESSAGE.format(a=a + 1, b=b, c=c, d=d))


if __name__ == "__main__":
    # Задаем среду
    env = gym.make('CartPole-v1')

    # Задаем константы
    total_steps = 4000
    states = env.observation_space.shape[0]
    actions = env.action_space.n

    # Создаем агента
    agent = DDQNAgent(state_size=states, action_size=actions, mode='soft')

    for episode in range(EPISODES):
        total_reward = 0
        state = np.reshape(env.reset(), [1, states])

        for ep_step in range(total_steps):
            env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)

            if not done or ep_step == total_steps - 1:
                reward = reward
            else:
                reward = -100

            total_reward += reward
            next_state = np.reshape(next_state, [1, states])
            data = state, action, reward, next_state, done
            agent.memory.append(data)
            state = next_state

            if done:
                agent.update_target_model()
                print_msg(episode + 1, EPISODES, ep_step, total_reward)
                break

            agent.play_with_sample()
    env.close()
