# Inverse reinforcement learning for highway driving

Two inverse-RL approaches to the same question: given demonstrations of an expert driving on a highway, recover
*why* it drives that way (a reward function) and learn a policy that drives like it. Built for a course project and
lecture on IRL (Spring 2025), on [highway-env](https://github.com/Farama-Foundation/HighwayEnv).

**Setup (both methods):** `highway-v0`, 3 lanes, discrete meta-actions (lane left, idle, lane right, faster, slower).
The observation is the kinematics of the ego car and the 4 nearest cars (presence, x, y, vx, vy).

```bash
pip install -r requirements.txt
```

## 1. Maximum-entropy IRL + DQN (`maxent_irl/`)
- **Expert:** a rule-based driver (keep speed in the centre lane, change lanes or slow down when a car is close ahead).
- **Features** of each state: lane position, gap and relative speed to the car ahead, gap to the car behind, lateral
  speed (a proxy for lane changes).
- **IRL loop:** a reward network over those features is trained with the maximum-entropy objective (expert feature
  expectations vs. the current policy's, with L2 regularisation on the reward), and a DQN learns a policy on the learned reward. Iterate.
- **Outputs:** the learned reward plotted against lane position, the training curves, and the learned policy
  compared against the rule-based expert and a random policy.

```bash
cd maxent_irl && python irl_autonomous_vehicle.py   # generates expert demos on first run
```

## 2. Successor Feature Matching (`sfm/`)
Non-adversarial IRL from [Jain et al. 2024](https://arxiv.org/abs/2409.01369): instead of learning a reward and
running RL on it in an inner loop, match the expert's *successor features* (expected discounted future features)
directly.
- **Expert:** my own driving, recorded with the keyboard (`utils/collect_human_demos.py`, S/D/F = left/idle/right,
  E/Q = faster/slower), in `data/`.
- **Model:** a learned feature map phi(s), a successor-feature network trained by TD on phi, and a policy network.
- **Checkpoints and training plots:** `checkpoints/`.

```bash
cd sfm && python main.py
python utils/collect_human_demos.py   # record new demonstrations
```

**Status:** exploratory, and the results in `checkpoints/` are not converged. Known issue: the policy loss is
computed from features of replay-buffer states only, so no gradient reaches the policy network and it doesn't
improve. The fix is to match successor features under the policy's own action distribution (e.g. the expected SF
over `softmax(policy(s))`), as in the paper.

## References
- A. Ng and S. Russell, *Algorithms for Inverse Reinforcement Learning*, ICML 2000.
- B. Ziebart et al., *Maximum Entropy Inverse Reinforcement Learning*, AAAI 2008.
- S. Arora and P. Doshi, *A Survey of Inverse Reinforcement Learning*, [arXiv:1806.06877](https://arxiv.org/abs/1806.06877).
- A. Jain et al., *Non-Adversarial Inverse Reinforcement Learning via Successor Feature Matching*, [arXiv:2409.01369](https://arxiv.org/abs/2409.01369).

Slides from the IRL lecture: [`docs/IRL_presentation_spring25.pptx`](docs/IRL_presentation_spring25.pptx).
