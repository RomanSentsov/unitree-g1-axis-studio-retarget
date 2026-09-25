TODO: complete the README with all the necessary information

This repo contains code to:

1. retarget hands and arms to unitree g1 robot using Axis Studio costume

2. inference retargeted angles on robot and dftp hands

Communication between different nodes is handled via Ros2

## Structure:

'''scripts/retarget_online_hands.py'''
Receives data from Axis Studio App () and retargets on unitree g1 and DFTP hands.

Publishes joint_states topic for robot arms and waist, and hand_state/r and /l for hand


'''scripts/teleop_ros2.py'''

Subscribes on mentioned topics and levereges unitree_sdk2_py (unitree_g1/unitree_g1 UnitreeG1 class) to controll the robot.
When program is launched, robot slowly (5s) transitions into your current state. When Ctrl+C is pressed, 
it similarly transitions back to idle.

For visualization, check this repo:

TODO

