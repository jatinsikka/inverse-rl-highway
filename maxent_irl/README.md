# Inverse Reinforcement Learning for Autonomous Vehicles

This project implements Inverse Reinforcement Learning (IRL) for autonomous vehicle behavior in a simulated highway environment using PyTorch and Highway-Env.

## Features
- Maximum Entropy IRL implementation
- DQN-based policy learning
- Highway environment simulation
- Expert demonstration collection and replay
- Visualization of training progress

## Requirements
- Python 3.8+
- PyTorch
- Highway-Env
- Gymnasium
- NumPy
- Matplotlib
- tqdm

## Installation
1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/irl-autonomous-vehicle.git
cd irl-autonomous-vehicle
```

2. Create and activate virtual environment:
```bash
python -m venv irl_venv
source irl_venv/bin/activate  # On Windows: irl_venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
Run the main training script:
```bash
python irl_project/irl_autonomous_vehicle.py
```

## Project Structure
- `irl_autonomous_vehicle.py`: Main implementation
- `data/`: Expert demonstrations
- `models/`: Saved model checkpoints
- `requirements.txt`: Project dependencies

## License
[MIT License](LICENSE)