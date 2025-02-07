if __name__=="__main__":
    import gym
    from l3c.anymdpv2 import AnyMDPv2TaskSampler

    task = AnyMDPv2TaskSampler(state_dim=128, 
                             action_dim=16)
    max_steps = 32000
    prt_freq = 1000

    # Test Random Policy
    env = gym.make("anymdp-v2")
    env.set_task(task)
    state, info = env.reset()
    acc_reward = 0
    epoch_reward = 0
    done = False

    steps = 0
    while steps < max_steps:
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        acc_reward += reward
        epoch_reward += reward
        steps += 1
        if(steps % prt_freq == 0 and steps > 0):
            print("Step:{}\tEpoch Reward: {}".format(steps, epoch_reward))
            epoch_reward = 0
        if(done):
            state, info = env.reset()
    print("Random Policy Summary: {}".format(acc_reward))

    print("Test Passed")