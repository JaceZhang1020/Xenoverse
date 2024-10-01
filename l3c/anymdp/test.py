if __name__=="__main__":
    import gym
    from l3c.anymdp import AnyMDPEnv, AnyMDPSolver, AnyMDPTaskSampler, Resampler

    env = gym.make("anymdp-v0", max_steps=100)
    task = AnyMDPTaskSampler(10, 4)
    env.set_task(Resampler(task))

    # Test Random Policy
    state, info = env.reset()
    acc_reward = 0
    done = False
    while not done:
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        acc_reward += reward
        print("State: {}, Action: {}, Reward: {}".format(state, action, reward))
    print("Accumulated Reward For Random Policy: {}".format(acc_reward))

    # Test AnyMDPSolver
    solver = AnyMDPSolver(env)
    state, info = env.reset()
    done = False
    acc_reward = 0
    while not done:
        action = solver.policy(state)
        state, reward, done, info = env.step(action)
        acc_reward += reward
        print("State: {}, Action: {}, Reward: {}".format(state, action, reward))
    print("Accumulated Reward with AnyMDP Solver: {}".format(acc_reward))

    print("Test Passed")