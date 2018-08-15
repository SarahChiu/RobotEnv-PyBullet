# Kuka Related Environments

## KukaContiGraspEnv
KukaContiGraspEnv is an environment with a Kuka robot arm grasping an object on a table. The goal is to make the robot arm successfully grasp the object and pick it up.

**State**
The state consists of 
1. the 7 joint angles of the robot arm,
2. the position as well as euler orientation of the end effector,
3. the relative position as well as euler orientation between the end effector and the object.

**Action**
The action is to apply 7 joint angle variations to the current joint angles. The dimension is 7, and the range of each variation is [-0.2, 0.2].

**Termination Condition**
An episode will terminate if
1. the end effector is lower than 0.1 and will try to grasp the object, 
2. the step number is over 10 (the robot arm should try to grasp the object in 10 steps).

## KukaContiStackInHandEnv
KukaContiStackInHandEnv is an environment with a Kuka robot arm stacking an object in hand onto another object. The goal is to make the robot arm successfully stack the objects.

**State**
The state consists of 
1. the 7 joint angles of the robot arm,
2. the position as well as euler orientation of the end effector,
3. the relative position as well as euler orientation between the end effector and the object on the table.

**Action**
The action is to apply 7 joint angle variations to the current joint angles. The dimension is 7, and the range of each variation is [-0.2, 0.2].

**Termination Condition**
An episode will terminate if
1. the end effector is lower than 0.15 and will try to release the object in hand, 
2. the step number is over 10 (the robot arm should try to stack the objects in 10 steps).

## KukaContiOpenDoorEnv
KukaContiOpenDoorEnv is an environment with a Kuka robot arm opening the door. The goal is to make the robot arm successfully open the door to over 15 degrees.

**State**
The state consists of 
1. the 7 joint angles of the robot arm,
2. the position as well as euler orientation of the end effector,
3. the relative position as well as euler orientation between the end effector and the door knob.

**Action**
The action is to apply 7 joint angle variations to the current joint angles. The dimension is 7, and the range of each variation is [-0.2, 0.2].

**Termination Condition**
An episode will terminate if
1. the x distance between the end effector and the door is smaller than 0.3, and the end effector will try to pull the door, 
2. the step number is over 10 (the robot arm should try to open the door in 10 steps).
