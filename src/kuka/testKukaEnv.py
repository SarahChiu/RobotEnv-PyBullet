import numpy as np

def test(env, num_epi, use_good):
    env_file = env[0].lower() + env[1:]
    exec('from kuka.' + env_file + ' import ' + env)
    env = eval(env + '(renders=True)')

    for epi in range(num_epi):
        if use_good: 
            ob, _ = env.getGoodInitState()
        else:
            ob = env.reset()
        ep_r = 0.0
        while True:
            a = np.random.normal(0.0, 0.02, size=env.action_space.shape[0])
            ob, r, t, _ = env.step(a)
            ep_r += r
            if t:
                print('Episode reward: ', ep_r)
                break

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', type=str, default='KukaContiGraspEnv')
    parser.add_argument('--num_epi', help='number of episodes', type=int, default=100)
    parser.add_argument('--use_good', help='use good init state', type=bool, default=False)
    args = parser.parse_args()

    test(args.env, args.num_epi, args.use_good)

if __name__ == '__main__':
    main()
