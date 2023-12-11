import cliff_walking
import numpy as np
env = cliff_walking.CliffWalking(render_mode="human")
observation, info = env.reset(seed=30)
max_iter_number = 1000
value_function = np.zeros(48)
gamma = 0.9
theta = 1e-8
delta = 1
while delta >= theta:
    delta = 0
    for s in range(48):
        v = value_function[s]
        q_values = []
        for a in range(4):
            rtmp = 0
            for i in range(-1, 2):
                _, next_state, reward, _ = env.P[s][(a + i) % 4][0]
                if next_state == 47:
                    reward = 100
                rtmp += 1 / 3 * (reward + gamma * value_function[next_state])
            q_values.append(rtmp)
        value_function[s] = max(q_values)
        delta = max(delta, abs(v - value_function[s]))
optimal_policy = np.zeros(48, dtype=int)
for s in range(48):
    q_values = []
    for a in range(4):
        rtmp = 0
        for i in range(-1, 2):
            _, next_state, reward, _ = env.P[s][(a + i) % 4][0]
            if next_state == 47:
                reward = 100
            rtmp += 1 / 3 * (reward + gamma * value_function[next_state])
        q_values.append(rtmp)
    optimal_policy[s] = np.argmax(q_values)

state = 36
for __ in range(max_iter_number):
    action = optimal_policy[state]
    next_state, reward, done, truncated, info = env.step(action)
    state = next_state
    if done or truncated:
        observation, info = env.reset()
env.close()