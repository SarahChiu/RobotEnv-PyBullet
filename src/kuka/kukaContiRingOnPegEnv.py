import os
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

class KukaContiRingOnPegEnv(gym.Env):
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

  def reset(self, finalJPos=[0.006418, 0, -0.011401, -0.785398, 0.005379, 0, -0.006539]):
    self.terminated = 0
    self.gripper_closed = 1
    p.resetSimulation()
    p.setPhysicsEngineParameter(numSolverIterations=150)
    p.setTimeStep(self._timeStep)
    p.loadURDF(os.path.join(self._urdfRoot,"plane.urdf"),[0,0,-1])
    
    p.loadURDF(os.path.join(self._urdfRoot,"table/table.urdf"), 0.5000000,0.00000,-.820000,0.000000,0.000000,0.0,1.0)

    #Load a ring for the gripper to grasp in hand
    xpos1 = 0.0
    ypos1 = 0.0
    ang1 = 1.570796
    orn1 = p.getQuaternionFromEuler([ang1,0,-ang1])
    
    p.setGravity(0,0,-10)
    jInitPos=[ 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, \
            0.000048, -0.040000, 0.000000, -0.000043, 0.040000, 0.000000, -0.000200 ]
    self._kuka = kuka.Kuka(baseInitPos=[-0.1,0.0,0.07], jointInitPos=jInitPos, gripperInitOrn=[orn1[0],orn1[1],orn1[2],orn1[3]], \
            fingerAForce=60, fingerBForce=55, fingerTipForce=60, \
            urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
    self.ringUid =p.loadURDF(os.path.join(os.environ['URDF_DATA'],"ring.urdf"), xpos1,ypos1,1.7,orn1[0],orn1[1],orn1[2],orn1[3])

    tempJPosDiff = np.array(finalJPos) - np.array(jInitPos[0:7])
    self._kuka.applyPosDiffAction(tempJPosDiff, self._renders)

    pegOrientation = p.getQuaternionFromEuler([0,0,0])
    xpos2 = 0.8 + 0.05*random.random()
    ypos2 = 0 + 0.05*random.random()
    self.pegUid =p.loadURDF(os.path.join(os.environ['URDF_DATA'],"peg.urdf"), [xpos2,ypos2,0.0], pegOrientation)

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
     pegPos,pegOrn = p.getBasePositionAndOrientation(self.pegUid)

     invEEPos,invEEOrn = p.invertTransform(endEffectorPos,endEffectorOrn)
     pegPosInEE,pegOrnInEE = p.multiplyTransforms(invEEPos,invEEOrn,pegPos,pegOrn)
     pegEulerInEE = p.getEulerFromQuaternion(pegOrnInEE)
     self._observation.extend(list(pegPosInEE))
     self._observation.extend(list(pegEulerInEE))

     return self._observation

  def getGoodInitState(self):
    self.reset()
    goodJointPos=[ 0.006418, -0.500000, -0.011401, -2.070796, 0.005379, 0.250000, -0.006539]
    self._kuka.initState(goodJointPos, self._renders)
    self._observation = self.getExtendedObservation()

    return np.array(self._observation), goodJointPos[0:7]

  def getMidInitState(self):
    midJointPos=[ 0.006418, -0.250000, -0.011401, -1.428097, 0.005379, 0.125000, -0.006539]
    self.reset(finalJPos=midJointPos)
    self._observation = self.getExtendedObservation()

    return np.array(self._observation)

  def getGoodMidInitState(self):
    goodMidJointPos=[ 0.006418, -0.375000, -0.011401, -1.749447, 0.005379, 0.187500, -0.006539]
    self.reset(finalJPos=goodMidJointPos)
    self._observation = self.getExtendedObservation()

    return np.array(self._observation)

  def setGoodInitState(self, ob, jointPoses, extra=None):
    self.reset()
    self._kuka.setGoodInitStateEE(jointPoses, self._renders)
    #Get pos and orn for the gripper
    linkState = p.getLinkState(self._kuka.kukaUid, self._kuka.kukaEndEffectorIndex)
    gripperPos = list(linkState[0])
    gripperOrn = list(linkState[1])
    #Set pos and orn for the peg
    pegOrnInEE = p.getQuaternionFromEuler(ob[16:19])
    pegPos, pegOrn = p.multiplyTransforms(gripperPos, gripperOrn, ob[13:16], pegOrnInEE)
    p.resetBasePositionAndOrientation(self.pegUid, pegPos, pegOrn)

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
    ringPos, _ = p.getBasePositionAndOrientation(self.ringUid)
 
    if (self.terminated or self._envStepCounter > 10):
      self._observation = self.getExtendedObservation()
      return True
    
    if (ringPos[2] <= 0.55):
      self.terminated = 1
      
      #print("opening gripper")
      self.gripper_closed = 0
      fingerAngle = 0

      for i in range(1000):
        p.setJointMotorControl2(self._kuka.kukaUid, 8, p.POSITION_CONTROL, targetPosition=-fingerAngle, force=self._kuka.fingerAForce)
        p.setJointMotorControl2(self._kuka.kukaUid, 11, p.POSITION_CONTROL, targetPosition=fingerAngle, force=self._kuka.fingerBForce)
        p.setJointMotorControl2(self._kuka.kukaUid, 10, p.POSITION_CONTROL, targetPosition=0, force=self._kuka.fingerTipForce)
        p.setJointMotorControl2(self._kuka.kukaUid, 13, p.POSITION_CONTROL, targetPosition=0, force=self._kuka.fingerTipForce)

        p.stepSimulation()
        fingerAngle = fingerAngle+(0.03/100.)
        if (fingerAngle>0.3):
          fingerAngle=0.3
        
      tempJPosDiff = [0, -0.5, 0, 0, 0, 0, 0]
      self._kuka.applyPosDiffAction(tempJPosDiff, self._renders)

      self._observation = self.getExtendedObservation()
      return True

    return False

  def _reward(self):
    
    #reward is the height of ring and xy distance between ring and peg
    ringPos,_ = p.getBasePositionAndOrientation(self.ringUid)
    pegPos,_ = p.getBasePositionAndOrientation(self.pegUid)
    dis = np.linalg.norm(np.array(ringPos[:2])-np.array(pegPos[:2]))

    reward = 0.0

    if (ringPos[2] < 0.1 and dis < 0.14 and self.terminated and not self.gripper_closed):
      #print("ring on peg!!!")
      #print("self._envStepCounter")
      #print(self._envStepCounter)
      #reward = reward+1000
      reward = 1.0

    return reward

  def internalReward(self):
    #reward is the distance between ring and peg
    closestPoints = p.getClosestPoints(self.ringUid, self.pegUid, 1000, linkIndexB=0)
    reward = -1000
    numPt = len(closestPoints)
    if (numPt>0):
      reward = -closestPoints[0][8]*10
    return reward
