import os, inspect

import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import pybullet as p
from . import kuka
import random
import pybullet_data

class KukaContiLineUpEnv(gym.Env):
  metadata = {
      'render.modes': ['human', 'rgb_array'],
      'video.frames_per_second' : 50
  }

  def __init__(self,
               urdfRoot=pybullet_data.getDataPath(),
               actionRepeat=1,
               isEnableSelfCollision=True,
               renders=False):
    self._timeStep = 1./240.
    self._urdfRoot = urdfRoot
    self._actionRepeat = actionRepeat
    self._isEnableSelfCollision = isEnableSelfCollision
    self._observation = []
    self._envStepCounter = 0
    self._renders = renders
    self.terminated = 0
    self.gripper_closed = 1
    self._p = p
    if self._renders:
      cid = p.connect(p.SHARED_MEMORY)
      if (cid<0):
         cid = p.connect(p.GUI)
      p.resetDebugVisualizerCamera(1.3,180,-41,[0.52,-0.2,-0.33])
    else:
      p.connect(p.DIRECT)

    self._seed()
    self.reset()
    observationDim = len(self.getExtendedObservation())
    
    observation_high = np.array([np.finfo(np.float32).max] * observationDim)    
    action_high = 0.2 + np.zeros(7)
    self.action_space = spaces.Box(-action_high, action_high) #continuous action
    self.observation_space = spaces.Box(-observation_high, observation_high)
    self.viewer = None

  def reset(self):
    self.terminated = 0
    self.gripper_closed = 1
    p.resetSimulation()
    p.setPhysicsEngineParameter(numSolverIterations=150)
    p.setTimeStep(self._timeStep)
    p.loadURDF(os.path.join(self._urdfRoot,"plane.urdf"),[0,0,-1])
    
    p.loadURDF(os.path.join(self._urdfRoot,"table/table.urdf"), 0.5000000,0.00000,-.820000,0.000000,0.000000,0.0,1.0)
    p.loadURDF(os.path.join(self._urdfRoot,"tray/tray.urdf"), 0.640000,0.075000,-0.190000,0.000000,0.000000,1.000000,0.000000)

    #Load a block for the gripper to grasp in hand
    xpos1 = 0.537
    ypos1 = 0.025
    ang1 = 1.570796
    orn1 = p.getQuaternionFromEuler([0,0,ang1])
    
    p.setGravity(0,0,-10)
    jInitPos=[ 0.006418, 1.134464, -0.011401, -1.589317, 0.005379, 0.436332, -0.006539, \
            0.000048, -0.016000, 0.000000, -0.000043, 0.016000, 0.000000, -0.000200 ]
    self._kuka = kuka.Kuka(baseInitPos=[-0.1,0.0,0.07], jointInitPos=jInitPos, gripperInitOrn=[orn1[0],orn1[1],orn1[2],orn1[3]], \
            fingerAForce=60, fingerBForce=55, fingerTipForce=60, \
            urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
    self.block1Uid =p.loadURDF(os.path.join(self._urdfRoot,"block.urdf"), xpos1,ypos1,-0.18,orn1[0],orn1[1],orn1[2],orn1[3])

    tempJPosDiff = [0, -0.808546, 0, 0, 0, 0.788618, 0]
    self._kuka.applyPosDiffAction(tempJPosDiff, self._renders)

    xpos2 = 0.5 +0.05*random.random()
    ypos2 = 0 +0.05*random.random()
    ang2 = 3.1415925438*random.random()
    orn2 = p.getQuaternionFromEuler([0,0,ang2])
    self.block2Uid =p.loadURDF(os.path.join(self._urdfRoot,"block.urdf"), xpos2,ypos2,-0.1,orn2[0],orn2[1],orn2[2],orn2[3])

    self._envStepCounter = 0
    p.stepSimulation()
    self._observation = self.getExtendedObservation()
    return np.array(self._observation)

  def __del__(self):
    p.disconnect()

  def _seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def getExtendedObservation(self):
     self._observation = self._kuka.getObservation()
     eeState  = p.getLinkState(self._kuka.kukaUid,self._kuka.kukaEndEffectorIndex)
     endEffectorPos = eeState[0]
     endEffectorOrn = eeState[1]
     blockPos,blockOrn = p.getBasePositionAndOrientation(self.block2Uid)

     invEEPos,invEEOrn = p.invertTransform(endEffectorPos,endEffectorOrn)
     blockPosInEE,blockOrnInEE = p.multiplyTransforms(invEEPos,invEEOrn,blockPos,blockOrn)
     blockEulerInEE = p.getEulerFromQuaternion(blockOrnInEE)
     self._observation.extend(list(blockPosInEE))
     self._observation.extend(list(blockEulerInEE))

     return self._observation

  def getGoodInitState(self):
    self.reset()

    _, block1Orn = p.getBasePositionAndOrientation(self.block1Uid)
    block1Orn = list(p.getEulerFromQuaternion(block1Orn))
    _, block2Orn = p.getBasePositionAndOrientation(self.block2Uid)
    block2Orn = list(p.getEulerFromQuaternion(block2Orn))

    goodJointPos=[ 0.006418, 0.922665, -0.011401, -1.589317, 0.005379, 0.648132, -0.006539]
    self._kuka.initState(goodJointPos, self._renders)

    tempJPosDiff = [0, 0, 0, 0, 0, 0, -block2Orn[2]+block1Orn[2]+0.006539]
    self._kuka.applyPosDiffAction(tempJPosDiff, self._renders)
    goodJointPos[-1] = -block2Orn[2]+block1Orn[2]

    self._observation = self.getExtendedObservation()

    return np.array(self._observation), goodJointPos[0:7]

  def getMidInitState(self):
    self.reset()
    midJointPos=[ 0.006418, 0.785398, -0.011401, -1.589317, 0.005379, 0.785398, -0.006539]
    self._kuka.initState(midJointPos, self._renders)
    self._observation = self.getExtendedObservation()

    return np.array(self._observation)

  def setGoodInitState(self, ob, jointPoses, extra=None):
    self.reset()
    tempJoint7 = jointPoses[-1]
    jointPoses[-1] = -0.006539
    self._kuka.setGoodInitStateEE(jointPoses, self._renders)
    tempJPosDiff = [0, 0, 0, 0, 0, 0, tempJoint7+0.006539]
    self._kuka.applyPosDiffAction(tempJPosDiff, self._renders)
    #Get pos and orn for the gripper
    linkState = p.getLinkState(self._kuka.kukaUid, self._kuka.kukaEndEffectorIndex)
    gripperPos = list(linkState[0])
    gripperOrn = list(linkState[1])
    #Set pos and orn for the block
    blockOrnInEE = p.getQuaternionFromEuler(ob[16:19])
    blockPos, blockOrn = p.multiplyTransforms(gripperPos, gripperOrn, ob[13:16], blockOrnInEE)
    p.resetBasePositionAndOrientation(self.block2Uid, blockPos, blockOrn)

    p.stepSimulation()
    self._observation = self.getExtendedObservation()

  def getCurrentJointPos(self):
    jointStates = list(p.getJointStates(self._kuka.kukaUid, range(self._kuka.kukaEndEffectorIndex+1)))
    jointPoses = []
    for state in jointStates:
        jointPoses.append(list(state)[0])

    return jointPoses

  def getExtraInfo(self):
    return None

  def step(self, action):
    return self.stepPosDiff(action)

  def step2(self, action):
    action = np.clip(action, self.action_space.low, self.action_space.high)
    self._kuka.applyAction2(action, self._renders)

    self._observation = self.getExtendedObservation()
    self._envStepCounter += 1

    done = self._termination()
    reward = self._reward()

    return np.array(self._observation), reward, done, {}

  #directly apply position difference commends
  def stepPosDiff(self, action):
    action = np.clip(action, self.action_space.low, self.action_space.high)
    self._kuka.applyPosDiffAction(action, self._renders)
    self._observation = self.getExtendedObservation()
    self._envStepCounter += 1
    
    done = self._termination()
    reward = self._reward()
    
    return np.array(self._observation), reward, done, {}

  def _render(self, mode='human', close=False):
      return

  def _termination(self):
    state = p.getLinkState(self._kuka.kukaUid,self._kuka.kukaEndEffectorIndex)
    actualEndEffectorPos = state[0]
 
    if (self.terminated or self._envStepCounter > 10):
      self._observation = self.getExtendedObservation()
      return True
    
    if (actualEndEffectorPos[2] <= 0.20): #0.10
      self.terminated = 1
      
      #print("opening gripper")
      self.gripper_closed = 0
      fingerAngle = 0
      
      for i in range (1000):
        p.setJointMotorControl2(self._kuka.kukaUid, 8, p.POSITION_CONTROL, targetPosition=-fingerAngle, force=self._kuka.fingerAForce)
        p.setJointMotorControl2(self._kuka.kukaUid, 11, p.POSITION_CONTROL, targetPosition=fingerAngle, force=self._kuka.fingerBForce)
        p.setJointMotorControl2(self._kuka.kukaUid, 10, p.POSITION_CONTROL, targetPosition=0, force=self._kuka.fingerTipForce)
        p.setJointMotorControl2(self._kuka.kukaUid, 13, p.POSITION_CONTROL, targetPosition=0, force=self._kuka.fingerTipForce)
        p.stepSimulation()
        fingerAngle = fingerAngle+(0.03/100.)
        if (fingerAngle>0.3):
          fingerAngle=0.3

      self._observation = self.getExtendedObservation()
      return True

    return False
 
  def _reward(self):
    
    #rewards is height of target object and the xy distance as well as yaw between two blocks
    block1Pos, block1Orn=p.getBasePositionAndOrientation(self.block1Uid)
    block2Pos, block2Orn=p.getBasePositionAndOrientation(self.block2Uid)
    block1Orn = list(p.getEulerFromQuaternion(block1Orn))
    block2Orn = list(p.getEulerFromQuaternion(block2Orn))

    heightDiff = abs(block1Pos[2] - block2Pos[2])
    xyDis = np.linalg.norm(np.array(block1Pos[:2])-np.array(block2Pos[:2]))
    ornDiff = abs(block1Orn[2] - block2Orn[2])

    reward = 0.0

    if (heightDiff < 0.01 and xyDis < 0.03 and ornDiff < 0.01 \
            and self.terminated and not self.gripper_closed):
      #print("line up!!!")
      #print("self._envStepCounter")
      #print(self._envStepCounter)
      #reward = reward+1000
      reward = 1.0

    return reward

  def internalReward(self):
    #rewards is the distance as well as orn diff between block1 and block2
    closestPoints = p.getClosestPoints(self.block1Uid,self.block2Uid,1000)
    reward = -1000
    numPt = len(closestPoints)
    if (numPt>0):
      reward = -closestPoints[0][8]*10

      _, block1Orn=p.getBasePositionAndOrientation(self.block1Uid)
      _, block2Orn=p.getBasePositionAndOrientation(self.block2Uid)
      block1Orn = list(p.getEulerFromQuaternion(block1Orn))
      block2Orn = list(p.getEulerFromQuaternion(block2Orn))
      ornDiff = abs(block1Orn[2] - block2Orn[2])

      reward -= ornDiff

    return reward
