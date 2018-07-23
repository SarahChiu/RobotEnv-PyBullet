# Robotics\_Env\_in\_PyBullet

This is a github project containing multiple simulation environments for robotics. These environments can be used to test different reinforcement learning algorithms.  
The environments are modified from the examples provided in **Bullet Physics SDK** (see [here](https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet/gym/pybullet_envs/examples) for more examples).

## Getting Started

### Prerequisites

1. Install OpenAI Gym (see [here](https://github.com/openai/gym) for the installation and more information)

2. Install PyBullet (see [here](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#heading=h.778da594xyte) for the installation and more information)

### Installation

1. Clone the repository.

```bash
git clone https://github.com/SarahChiu/Robotics_Env_in_PyBullet.git
cd Robotics_Env_in_PyBullet/src
```

2. Install the package.

```bash
pip install -e .
```

3. Add the path of the installed package to *PYTHONPATH*.

```bash
export PYTHONPATH=/your_path_to_this_project/src
```

### Usage
Here is an example code for running the simulation environment.

```python
import numpy as np
from kuka.kukaContiGraspEnv import KukaContiGraspEnv as grasp

#Set renders to true if you want to show the UI
env = grasp(renders=True)

#To get an initial observation, you can try one of the following functions
ob = env.reset()
ob, _ = env.getGoodInitState()
ob = env.getMidInitState()

#Run an episode and output the episode reward
ep_r = 0.0
while True:
    a = np.random.normal(0.0, 0.02, size=env.action_space.shape[0]) 
    ob, r, t, _ = env.step(a)
    ep_r += r
    if t:
        print('Episode reward: ', ep_r)
        break
```

### Available Environments
Please checkout [src](src/) for more information.

## Acknowledgement
More environments are under development. If you have any problem or suggestion, please feel free to contact me ([email](mailto:z.y.sarah.chiu@gmail.com), [Twitter](https://twitter.com/zihyunchiu), [LinkedIn](https://www.linkedin.com/in/zihyun-chiu/)).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

