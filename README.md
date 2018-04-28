# comp_767_project
Huiwen You, huiwen.you@mail.mcgill.ca 260798462

Zilun Peng, zilun.peng@mail.mcgill.ca 260529763

### Install gym

### Install gym-lavaland
```
cd gym-lavaland
pip3 install -e .
```

Posterior calculation is implemented in IRD.py

Linear programming risk-averse planner and runner for experiment 4.1 4.2 is under Agent_planner.py

experiment 4.3 can be run using Agent_planner_reward_hacking.py

mdp environment setup code is under ./gym-lavaland/env

Lavaland_spec.py contains preparation code for risk-averse planner.

policy_iteration.py contains the PI implementation.

baseline.py contains another baseline method i.e. q_learning
