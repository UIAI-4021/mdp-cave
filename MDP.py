
import cliff_wakling
import numpy as np

A = [0, 1, 2, 3]
S = [i for i in range(36)]
T = np.zeros((len(S), len(A), len(S)))
for i in range(len(S)):
    for j in range(len(A)):
        T[i, j, i - 1] = 1 / 3
        T[i, j, i + 1] = 1 / 3
        T[i, j, i - 12] = 1 / 3
        T[i, j, i + 12] = 1 / 3
    T[i, 0, i + 12] = 0
    T[i, 1, i - 1] = 0
    T[i, 2, i - 12] = 0
    T[i, 3, i + 1] = 0
R = np.zeros((len(S), len(A)))
for i in range(len(S)):
    for j in range(len(A)):
        R[i, j] = -1

def choose_action(state, policy):
    return np.random.choice(actions, p=[policy[state][a] for a in actions])

# Create an environment
env = CliffWalking(render_mode="human")
observation, info = env.reset(seed=30)

# Define the maximum number of iterations
max_iter_number = 1000

policy = {}
for i in range(0,36):
    dic = {}
    for j in range(0,4):
        dic[j] = 0.25
    policy[i] = dic
for __ in range(max_iter_number):

    
    # Note: .sample() is used to sample random action from the environment's action space

    # Choose an action (Replace this random action with your agent's policy)
    action = choose_action(next_state, policy)

    # Perform the action and receive feedback from the environment
    next_state, reward, done, truncated, info = env.step(action)

    if done or truncated:
        observation, info = env.reset()

# Close the environment
env.close()