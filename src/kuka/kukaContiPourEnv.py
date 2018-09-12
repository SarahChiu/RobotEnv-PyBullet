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

class KukaContiPourEnv(gym.Env):
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
    box_xpos = 0.85 + 0.05 * random.random()
    box_ypos = 0 + 0.05 * random.random()
    box_ang = 3.141593*random.random()
    box_orn = p.getQuaternionFromEuler([1.570796,0,box_ang])
    self.boxUid = p.loadURDF(os.path.join(os.environ['URDF_DATA'],"cardboard_box.urdf"), [box_xpos,box_ypos,0], box_orn)

    #Load a cup for the gripper to grasp in hand
    xpos = 0.25
    ypos = 0.0
    ang = 1.570796
    orn = p.getQuaternionFromEuler([0,-ang,0])
    
    p.setGravity(0,0,-10)
    jInitPos=[ 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, -1.570796, \
            0.000048, -0.020000, 0.000000, -0.000043, 0.020000, 0.000000, -0.000200 ]
    self._kuka = kuka.Kuka(baseInitPos=[-0.1,0.0,0.07], jointInitPos=jInitPos, gripperInitOrn=[orn[0],orn[1],orn[2],orn[3]], \
            fingerAForce=60, fingerBForce=55, fingerTipForce=60, \
            urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
    self.cupUid =p.loadURDF(os.path.join(os.environ['URDF_DATA'],"cup.urdf"),xpos,ypos,1.63,orn[0],orn[1],orn[2],orn[3])

    resetInitPos = [0.006418, 0, -0.011401, -1.070796, 0.005379, 0.5, -1.570796]
    tempJPosDiff = np.array(resetInitPos) - np.array(jInitPos[0:7])
    self._kuka.applyPosDiffAction(tempJPosDiff, self._renders)

    #Load 5 cubes in the cup
    cup_pos, _ = p.getBasePositionAndOrientation(self.cupUid)
    self.cubeUids = []
    for i in range(2):
        self.cubeUids.append(p.loadURDF(os.path.join(self._urdfRoot,"cube_small.urdf"),\
                [cup_pos[0],cup_pos[1],(i+1)*0.1+cup_pos[2]]))

    self._envStepCounter = 0
    for i in range(100):
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
     boxPos,boxOrn = p.getBasePositionAndOrientation(self.boxUid)

     invEEPos,invEEOrn = p.invertTransform(endEffectorPos,endEffectorOrn)
     boxPosInEE,boxOrnInEE = p.multiplyTransforms(invEEPos,invEEOrn,boxPos,boxOrn)
     boxEulerInEE = p.getEulerFromQuaternion(boxOrnInEE)
     self._observation.extend(list(boxPosInEE))
     self._observation.extend(list(boxEulerInEE))

     return self._observation

  def getGoodInitState(self):
    self.reset()
    goodJointPos=[ 0.250000, 0.250000, -0.011401, -1.589317, 0.005379, 0.400000, -1.570796]
    self._kuka.initState(goodJointPos, self._renders)

    tempJPosDiff = [0, 0, 0, 0, 0, 0, -0.5+1.570796]
    self._kuka.applyPosDiffAction(tempJPosDiff, self._renders)
    goodJointPos[-1] = -0.5

    self._observation = self.getExtendedObservation()

    return np.array(self._observation), goodJointPos[0:7]

  def getMidInitState(self):
    self.reset()
    midJointPos=[ 0.128209, 0.125000, -0.011401, -1.330057, 0.005379, 0.450000, -1.035398]
    self._kuka.initState(midJointPos, self._renders)
    self._observation = self.getExtendedObservation()

    return np.array(self._observation)

  def getGoodMidInitState(self):
    self.reset()
    goodMidJointPos=[ 0.189105, 0.187500, -0.011401, -1.459687, 0.005379, 0.425000, -0.767699]
    self._kuka.initState(goodMidJointPos, self._renders)
    self._observation = self.getExtendedObservation()

    return np.array(self._observation)

  def setGoodInitState(self, ob, jointPoses, extra=None):
    self.reset()
    tempJoint7 = jointPoses[-1]
    jointPoses[-1] = -1.570796
    self._kuka.setGoodInitStateEE(jointPoses, self._renders)

    tempJPosDiff = [0, 0, 0, 0, 0, 0, tempJoint7+1.570796]
    self._kuka.applyPosDiffAction(tempJPosDiff, self._renders)

    #Get pos and orn for the gripper
    linkState = p.getLinkState(self._kuka.kukaUid, self._kuka.kukaEndEffectorIndex)
    gripperPos = list(linkState[0])
    gripperOrn = list(linkState[1])
    #Set pos and orn for the box
    boxOrnInEE = p.getQuaternionFromEuler(ob[16:19])
    boxPos, boxOrn = p.multiplyTransforms(gripperPos, gripperOrn, ob[13:16], boxOrnInEE)
    p.resetBasePositionAndOrientation(self.boxUid, boxPos, boxOrn)

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
    
    if (actualEndEffectorPos[2] <= 0.57): 
      self.terminated = 1
      
      #print("pour the cubes")
      self.gripper_closed = 1
      curJointPos = self.getCurrentJointPos()
      tempJPosDiff = [0, 0, 0, 0, 0, 0, 0.75-curJointPos[-1]]
      self._kuka.applyPosDiffAction(tempJPosDiff, self._renders)

      for i in range(1000):
        p.stepSimulation()
        
      self._observation = self.getExtendedObservation()
      return True

    return False

  def _reward(self):
    
    #reward is if the cubes are placed into the box
    contact1Pts = p.getContactPoints(self.cubeUids[0], self.cupUid)
    contact2Pts = p.getContactPoints(self.cubeUids[1], self.cupUid)
    cube1Pos, _ = p.getBasePositionAndOrientation(self.cubeUids[0])
    cube2Pos, _ = p.getBasePositionAndOrientation(self.cubeUids[1])
    cupPos, _ = p.getBasePositionAndOrientation(self.cupUid)

    reward = 0.0

    if (len(contact1Pts)==0 and len(contact2Pts)==0 \
            and cube1Pos[2]<0.1 and cube1Pos[2]>-0.1 \
            and cube2Pos[2]<0.1 and cube2Pos[2]>-0.1 \
            and cupPos[2] > 0.15 \
            and self.terminated and self.gripper_closed):
      #print("pour!!!")
      #print("self._envStepCounter")
      #print(self._envStepCounter)
      #reward = reward+1000
      reward = 1.0

    return reward

  def internalReward(self):
    #rewards is the larger distance between cubes and box
    closest1Points = p.getClosestPoints(self.cubeUids[0],self.boxUid,1000)
    closest2Points = p.getClosestPoints(self.cubeUids[1],self.boxUid,1000)
    reward = -1000
    numPt1 = len(closest1Points)
    numPt2 = len(closest2Points)
    if (numPt1>0 and numPt2>0):
      reward = min(-closest1Points[0][8]*10, -closest2Points[0][8]*10)
    return reward
