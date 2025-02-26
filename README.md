# Dataset-crowd-simulation
This is a simulator that leverages pedestrian datasets to simulate pedestrians.

# Setup
- Go to base directory
```
git clone https://github.com/allanwangliqian/dataset-crowd-simulation.git
cd dataset-crowd-simulation
```

- Set up a virtual environment
```
python3 -m venv .myvenv
source .myvenv/bin/activate
python3 -m venv .
```

- Install dependencies
```
pip install -r requirements.txt
```

- Set up [RVO2](https://github.com/sybrenstuvel/Python-RVO2). If there are any issues, please go to [RVO2](https://github.com/sybrenstuvel/Python-RVO2) for troubleshooting.
```bash
git clone https://github.com/sybrenstuvel/Python-RVO2.git
cd Python-RVO2
pip install Cython
python setup.py build
python setup.py install
cd ..
```

- Set up [SFMpy](https://github.com/yuxiang-gao/PySocialForce). 
```bash
git clone https://github.com/yuxiang-gao/PySocialForce.git
cd PySocialForce
pip install -e '.[test,plot]'
cd ..
```

- Go to sim `cd sim`. Download the ETH, UCY and ATC(sample) datasets tar file from [here](https://drive.google.com/file/d/1Q_5xG4CmC69oEvriss0psAsr4CGtl8yZ/view?usp=sharing) and then extract.
```
tar -xvf datasets.tar
```

# Usage
- Make a data folder
```
mkdir data
```
- Use `test_case_helper.py` to check dataset pedestrian flows. Experiment with check regions and robot start and goal locations.
- Copy `check_regions` and `start_end_pos` to `test_case_generation.py`.
- Use `test_case_generation.py` to generate test cases.
- Initialize the simulator using the generated test files. See `main.py` for an example.



## for RL
pip install tensorboard
pip install gymnasium