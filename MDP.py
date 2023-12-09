import cliff_walking
env = cliff_walking.CliffWalking(render_mode="human")
observation, info = env.reset(seed=30)
max_iter_number = 1000
def Policy(state):
    min = 3
    for i in range(0, 4):
        if env.P[state][min][0][2] > env.P[state][i][0][2]:
            min = i
    print(env.P[state][min][0])
    if env.P[state][min][0][2] == -100:
        return (min + 2) % 4
    else:
        return 1
state = 36
for __ in range(max_iter_number):
    action = Policy(state)
    next_state, reward, done, truncated, info = env.step(action)
    state = next_state
    if done or truncated:
        observation, info = env.reset()
env.close()