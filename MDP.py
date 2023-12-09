import cliff_walking
import numpy as np
env = cliff_walking.CliffWalking(render_mode="human")
observation, info = env.reset(seed=30)
max_iter_number = 1000
value_function = np.zeros(48)
gamma = 0.9
theta = 1e-8
while True:
    delta = 0
    for s in range(48):
        v = value_function[s]
        q_values = []
        for a in range(4):
            _, next_state, reward, _ = env.P[s][a][0]
            if next_state == 46:
                reward = 100
            q_values.append(reward + gamma * value_function[next_state])
        value_function[s] = max(q_values)
        delta = max(delta, abs(v - value_function[s]))
    if delta < theta:
        break
print("Optimal Value Function:")
print(value_function)
optimal_policy = np.zeros(48, dtype=int)
for s in range(48):
    q_values = []
    for a in range(4):
        _, next_state, reward, _ = env.P[s][a][0]
        q_values.append(reward + gamma * value_function[next_state])
    optimal_policy[s] = np.argmax(q_values)

state = 36
for __ in range(max_iter_number):
    action = optimal_policy[state]
    next_state, reward, done, truncated, info = env.step(action)
    state = next_state
    if done or truncated:
        observation, info = env.reset()
env.close()