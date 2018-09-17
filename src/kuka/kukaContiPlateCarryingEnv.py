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

class KukaContiPlateCarryingEnv(gym.Env):
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

  def reset(self, finalJPos=[0.006418, 0, -0.011401, -1.070796, 0.005379, 0.5, 0]):
    self.terminated = 0
    self.gripper_closed = 1
    p.resetSimulation()
    p.setPhysicsEngineParameter(numSolverIterations=150)
    p.setTimeStep(self._timeStep)
    p.loadURDF(os.path.join(self._urdfRoot,"plane.urdf"),[0,0,-1])
    
    p.loadURDF(os.path.join(self._urdfRoot,"table/table.urdf"), 0.5000000,0.00000,-.820000,0.000000,0.000000,0.0,1.0)
    table_xpos = 0.85 + 0.05 * random.random()
    table_ypos = 0 + 0.2 * random.random()
    table_ang = 3.141593*random.random()
    table_orn = p.getQuaternionFromEuler([0,0,table_ang])

    p.setGravity(0,0,-10)
    jInitPos=[ 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, \
            0.000048, -0.035000, 0.000000, -0.000043, 0.035000, 0.000000, -0.000200 ]
    self._kuka = kuka.Kuka(baseInitPos=[-0.1,0.0,-0.12], jointInitPos=jInitPos, \
            gripperInitOrn=[table_orn[0],table_orn[1],table_orn[2],table_orn[3]], \
            fingerAForce=60, fingerBForce=55, fingerTipForce=60, \
            urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
    ang1 = 1.570796
    orn1 = p.getQuaternionFromEuler([ang1,0,-ang1])
    self.plateUid =p.loadURDF(os.path.join(os.environ['URDF_DATA'],"plate.urdf"),0,0,1.41,orn1[0],orn1[1],orn1[2],orn1[3])

    tempJPosDiff = np.array(finalJPos) - np.array(jInitPos[0:7])
    self._kuka.applyPosDiffAction(tempJPosDiff, self._renders)

    platePos, _ = p.getBasePositionAndOrientation(self.plateUid)
    cup1 = p.loadURDF(os.path.join(os.environ['URDF_DATA'], 'cup2.urdf'),[platePos[0]+0.075,platePos[1]+0.075,platePos[2]+0.015],[0,0,0,1])
    cup2 = p.loadURDF(os.path.join(os.environ['URDF_DATA'], 'cup2.urdf'),[platePos[0]-0.075,platePos[1]+0.075,platePos[2]+0.015],[0,0,0,1])
    cup3 = p.loadURDF(os.path.join(os.environ['URDF_DATA'], 'cup2.urdf'),[platePos[0]-0.075,platePos[1]-0.075,platePos[2]+0.015],[0,0,0,1])
    cup4 = p.loadURDF(os.path.join(os.environ['URDF_DATA'], 'cup2.urdf'),[platePos[0]+0.075,platePos[1]-0.075,platePos[2]+0.015],[0,0,0,1])
    self.cupUids = [cup1, cup2, cup3, cup4]
    self.tableUid = p.loadURDF(os.path.join(self._urdfRoot,"table/table.urdf"), \
            [table_xpos,table_ypos,0], table_orn, globalScaling=.4)

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
     tablePos,tableOrn = p.getBasePositionAndOrientation(self.tableUid)

     invEEPos,invEEOrn = p.invertTransform(endEffectorPos,endEffectorOrn)
     tablePosInEE,tableOrnInEE = p.multiplyTransforms(invEEPos,invEEOrn,tablePos,tableOrn)
     tableEulerInEE = p.getEulerFromQuaternion(tableOrnInEE)
     self._observation.extend(list(tablePosInEE))
     self._observation.extend(list(tableEulerInEE))

     return self._observation

  def getGoodInitState(self):
    goodJointPos=[0.006418, 0.400000, -0.011401, -1.570796, 0.005379, -0.400000, 0.000000]
    self.reset(finalJPos=goodJointPos)
    self._observation = self.getExtendedObservation()

    return np.array(self._observation), goodJointPos[0:7]

  def getMidInitState(self):
    midJointPos=[0.006418, 0.200000, -0.011401, -1.320796, 0.005379, 0.050000, 0.000000]
    self.reset(finalJPos=midJointPos)
    self._observation = self.getExtendedObservation()

    return np.array(self._observation)

  def setGoodInitState(self, ob, jointPoses, extra=None):
    self.reset(finalJPos=jointPoses)
    #Get pos and orn for the gripper
    linkState = p.getLinkState(self._kuka.kukaUid, self._kuka.kukaEndEffectorIndex)
    gripperPos = list(linkState[0])
    gripperOrn = list(linkState[1])
    #Set pos and orn for the table
    tableOrnInEE = p.getQuaternionFromEuler(ob[16:19])
    tablePos, tableOrn = p.multiplyTransforms(gripperPos, gripperOrn, ob[13:16], tableOrnInEE)
    p.resetBasePositionAndOrientation(self.tableUid, tablePos, tableOrn)

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
    actualEndEffectorPos = list(state[0])
    actualEndEffectorOrn = list(state[1])
    platePos, _ = p.getBasePositionAndOrientation(self.plateUid)
 
    if (self.terminated or self._envStepCounter > 10):
      self._observation = self.getExtendedObservation()
      return True
    
    if (platePos[2] <= 0.33):
      self.terminated = 1
      
      #print("opening gripper")
      self.gripper_closed = 0
      fingerAngle = 0
      
      for i in range (1000):
        p.setJointMotorControl2(self._kuka.kukaUid, 8, p.POSITION_CONTROL, targetPosition=-fingerAngle, force=self._kuka.fingerAForce)
        p.setJointMotorControl2(self._kuka.kukaUid, 11, p.POSITION_CONTROL, targetPosition=fingerAngle, force=self._kuka.fingerBForce)
        p.setJointMotorControl2(self._kuka.kukaUid, 10, p.POSITION_CONTROL, targetPosition=0, force=self._kuka.fingerTipForce)
        p.setJointMotorControl2(self._kuka.kukaUid, 13, p.POSITION_CONTROL, targetPosition=0, force=self._kuka.fingerTipForce)

        actualEndEffectorPos[0] -= 0.00025
        jPos = p.calculateInverseKinematics(self._kuka.kukaUid, self._kuka.kukaEndEffectorIndex, \
                actualEndEffectorPos, actualEndEffectorOrn, \
                self._kuka.ll, self._kuka.ul, self._kuka.jr, self._kuka.rp)
        for j in range(self._kuka.kukaEndEffectorIndex+1):
            p.setJointMotorControl2(bodyIndex=self._kuka.kukaUid, jointIndex=j, controlMode=p.POSITION_CONTROL, \
                    targetPosition=jPos[j], targetVelocity=0, force=self._kuka.maxForce, positionGain=0.03, velocityGain=1)

        p.stepSimulation()
        fingerAngle = fingerAngle+(0.03/100.)
        if (fingerAngle>0.3):
          fingerAngle=0.3
        
      self._observation = self.getExtendedObservation()
      return True

    return False

  def _reward(self):
    
    #reward is if the plate and cups are on the table
    platePos, plateOrn = p.getBasePositionAndOrientation(self.plateUid)
    plateOrn = p.getEulerFromQuaternion(plateOrn)
    cupPos = []
    cupOrn = []
    for i in range(len(self.cupUids)):
        pos, orn = p.getBasePositionAndOrientation(self.cupUids[i])
        orn = p.getEulerFromQuaternion(orn)
        cupPos.append(pos)
        cupOrn.append(orn)

    reward = 0.0

    plateCond = True
    if (platePos[2] >= 0.35 or platePos[2] <= 0.25 or not plateOrn[0] < .01 or not plateOrn[1] < .01):
        plateCond = False

    cupCond = True
    for i in range(len(self.cupUids)):
        contactPts = p.getContactPoints(self.cupUids[i], self.plateUid)
        if (len(contactPts) == 0):
            cupCond = False
            break
        elif (cupPos[i][2] >= 0.35 or cupPos[i][2] <= 0.25 or not cupOrn[i][0] < .01 or not cupOrn[i][1] < .01):
            cupCond = False
            break

    if (plateCond and cupCond and self.terminated and not self.gripper_closed):
      #print("plate carrying!!!")
      #print("self._envStepCounter")
      #print(self._envStepCounter)
      #reward = reward+1000
      reward = 1.0

    return reward

  def internalReward(self):
    #rewards is the distance between plate and table
    closestPoints = p.getClosestPoints(self.plateUid,self.tableUid,1000)
    reward = -1000
    numPt = len(closestPoints)
    if (numPt>0):
      reward = -closestPoints[0][8]*10
    return reward
