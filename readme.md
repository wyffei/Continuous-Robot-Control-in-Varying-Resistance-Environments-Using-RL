# File Description

## 1. arm4addforce1104.xml
The geometric model, mechanical structure, and physical parameters of a robotic arm, such as density, color, material, and joint type.
## 2. environment.yml
Necessary environments for running the code.
## 3. gym_twofluid_1104.py
This gym environment is built upon the MuJoCo physics simulator to model the ten-joint octopus-inspired robotic arm operating in a fluid domain. It provides a continuous control interface for reinforcement learning algorithms and includes full visualization, logging, and hydrodynamic modeling.
## 4. hydro_forces1104.py
This file constructs the fluid resistance and torque experienced by the robotic arm underwater, mainly based on the Morison equation, and is imported into the environment at each step of Mujoco's calculations
## 5. fluid_field1104.py
This file constructs two fluid fields, namely air and water, simulates their respective features (density, viscosity, etc.), and divides them by the X-Y plane.
## 6.link1104.stl
This is a CAD model file for the robotic arm, used for importing into an XML file
## 7.pid_controller_twofluid.py
This is the low-level PID controller file used to control the joints of the robotic arm to reach the specified angle in each step of the calculation. 
## 8.train_twofluid.py
The training script initializes the custom Gym environment, configures the PPO algorithm along with selected parameters, and defines a multi-layer policy network with separate actor and critic branches. Customed callback (TrainLoggerCallback) periodically saves model checkpoints and training metadata to ensure reproducibility and continuous monitoring.EpisodeEnergyPlotCallback records total energy cost of each episode. EntropyDecayCallback realizes the decaying of entropy along time.
## 9.reward_plot_callback.py
This file enables visualizing of the reward as well as its components during trainging, which allow us to monitor.
## 10. Inverse Kinematic
gym_ik.py is the gym environment for ik, which is the same as twofluid training.

qp_ik.py is used to compute inverse kinematic by iteration.

train_twofluid_ik.py is just for ik computing.

## 11. Demo videos
All demo videos including fixed goal underwater, random goal underwater, fixed goal in two fluid, random goal in two fluid.

# Usage Instructions
## 1.Create a Virtual Environment with Conda 
```python
conda env create -f group7.yml -n group7
```
## 2.Place the folder in the created environment
## 3.Run file train_twofluid.py to start new training
To start new training, disable line 162 of train_twofluid.py:
```
model = RecurrentPPO.load("checkpoint_2025-11-17_16-03-30_step_30000.zip", env=env, tensorboard_log=log_dir)
```
by commenting. 
run:
```python
python train_twofluid.py
```
Results will be saved in the following folder:
```
./results/run_{year_month_date_time}/
-checkpoints
```
## 4. Visualize
### 4.1 Visualize Inverse Kinematics (IK) computed results
To visualize IK computed results, run:
```
python train_twofluid_ik.py
```

### 4.2 Visualize trained checkpoint
To visualize trained checkpoint results, just run train_twofluid.py:
```
python train_twofluid.py
```




