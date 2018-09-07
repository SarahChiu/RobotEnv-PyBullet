import os,  inspect
#currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#parentdir = os.path.dirname(os.path.dirname(currentdir))
#os.sys.path.insert(0,parentdir)

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

class KukaContiOpenDoorEnv(gym.Env):
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
    self.gripper_closed = 0
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
    self.gripper_closed = 0
    p.resetSimulation()
    p.setPhysicsEngineParameter(numSolverIterations=150)
    p.setTimeStep(self._timeStep)
    p.loadURDF(os.path.join(self._urdfRoot,"plane.urdf"),[0,0,-1])
    
    p.loadURDF(os.path.join(self._urdfRoot,"table/table.urdf"), 0.5000000,0.00000,-.820000,0.000000,0.000000,0.0,1.0)
    
    doorOrientation = p.getQuaternionFromEuler([0,0,1.570796])
    xpos = 0.9 + 0.05 * random.random()
    ypos = -0.25 + 0.05 * random.random()
    self.doorUid = p.loadURDF(os.path.join(os.environ['URDF_DATA'],"door.urdf"), [xpos, ypos, 0.0], doorOrientation)

    p.setGravity(0,0,-10)
    orn = p.getQuaternionFromEuler([0,0,0])
    jInitPos = [0.006418, 0.413184, -0.011401, -1.589317, 0.005379, 1.137684, -0.006539, \
            0.000048, -0.299912, 0.000000, -0.000043, 0.299960, 0.000000, -0.000200]
    self._kuka = kuka.Kuka(baseInitPos=[-0.1,0.0,0.07], jointInitPos = jInitPos, gripperInitOrn=[orn[0],orn[1],orn[2],orn[3]], \
            fingerAForce=60, fingerBForce=55, fingerTipForce=60, \
            urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
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

     doorKnobState = p.getLinkState(self.doorUid, 2)
     doorKnobPos,doorKnobOrn = doorKnobState[0], doorKnobState[1]

     invEEPos,invEEOrn = p.invertTransform(endEffectorPos,endEffectorOrn)
     doorKnobPosInEE,doorKnobOrnInEE = p.multiplyTransforms(invEEPos,invEEOrn,doorKnobPos,doorKnobOrn)
     doorKnobEulerInEE = p.getEulerFromQuaternion(doorKnobOrnInEE)
     self._observation.extend(list(doorKnobPosInEE))
     self._observation.extend(list(doorKnobEulerInEE))

     return self._observation

  def getGoodInitState(self):
    self.reset()
    goodJointPos=[ 0.610865, 0.523599, -0.011401, -1.308997, 0.005379, 0.000000, -0.006539]
    self._kuka.initState(goodJointPos, self._renders)
    self._observation = self.getExtendedObservation()

    return np.array(self._observation), goodJointPos[0:7]

  def getMidInitState(self):
    self.reset()
    midJointPos=[ 0.308642, 0.468392, -0.011401, -1.449157, 0.005379, 0.568842, -0.006539]
    self._kuka.initState(midJointPos, self._renders)
    self._observation = self.getExtendedObservation()

    return np.array(self._observation)

  def getGoodMidInitState(self):
    self.reset()
    goodMidJointPos=[ 0.459754, 0.495996, -0.011401, -1.404077, 0.005379, 0.284421, -0.006539]
    self._kuka.initState(goodMidJointPos, self._renders)
    self._observation = self.getExtendedObservation()

    return np.array(self._observation)

  def setGoodInitState(self, ob, jointPoses, extra=None): #extra --> door: [pos, orn]
    self.reset()
    self._kuka.setGoodInitStateEE(jointPoses, self._renders)
    #Set pos, orn, and joint angle for the door
    p.resetBasePositionAndOrientation(self.doorUid, extra[0], extra[1])

    p.stepSimulation()
    self._observation = self.getExtendedObservation()

  def getCurrentJointPos(self):
    jointStates = list(p.getJointStates(self._kuka.kukaUid, range(self._kuka.kukaEndEffectorIndex+1)))
    jointPoses = []
    for state in jointStates:
        jointPoses.append(list(state)[0])

    return jointPoses

  def getExtraInfo(self): #Current door info
    doorPos, doorOrn = p.getBasePositionAndOrientation(self.doorUid)

    return [doorPos, doorOrn]

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
    doorPos, _ = p.getBasePositionAndOrientation(self.doorUid)
 
    if (self.terminated or self._envStepCounter > 10):
      self._observation = self.getExtendedObservation()
      return True
    
    if (abs(doorPos[0]-actualEndEffectorPos[0]) <= 0.32):
      self.terminated = 1
      
      #print("closing gripper, attempting holding door knob")
      self.gripper_closed = 1
      #start grasp and terminate
      fingerAngle = 0.3
      
      for i in range (1000):
        p.setJointMotorControl2(self._kuka.kukaUid, 8, p.POSITION_CONTROL, targetPosition=-fingerAngle, force=self._kuka.fingerAForce)
        p.setJointMotorControl2(self._kuka.kukaUid, 11, p.POSITION_CONTROL, targetPosition=fingerAngle, force=self._kuka.fingerBForce)
        p.setJointMotorControl2(self._kuka.kukaUid, 10, p.POSITION_CONTROL, targetPosition=0, force=self._kuka.fingerTipForce)
        p.setJointMotorControl2(self._kuka.kukaUid, 13, p.POSITION_CONTROL, targetPosition=0, force=self._kuka.fingerTipForce)

        #pull the door
        actualEndEffectorPos[0] -= 0.00025
        actualEndEffectorPos[1] -= 0.00025
        jPos = p.calculateInverseKinematics(self._kuka.kukaUid, self._kuka.kukaEndEffectorIndex, \
                actualEndEffectorPos, actualEndEffectorOrn, \
                self._kuka.ll, self._kuka.ul, self._kuka.jr, self._kuka.rp)
        for j in range(self._kuka.kukaEndEffectorIndex+1):
            p.setJointMotorControl2(bodyIndex=self._kuka.kukaUid, jointIndex=j, controlMode=p.POSITION_CONTROL, \
                    targetPosition=jPos[j], targetVelocity=0, force=self._kuka.maxForce, positionGain=0.03, velocityGain=1)

        p.stepSimulation()
        fingerAngle = fingerAngle-(0.3/100.)
        if (fingerAngle<0):
          fingerAngle=0
        
      self._observation = self.getExtendedObservation()
      return True

    return False
  
  def _reward(self):
    
    #rewards is rotation of the door
    doorJointPos = p.getJointState(self.doorUid, 1)[0]

    reward = 0.0

    if (doorJointPos > 0.261799 and self.terminated and self.gripper_closed):
      #print("open the door!!!")
      #print("self._envStepCounter")
      #print(self._envStepCounter)
      #reward = reward+1000
      reward = 1.0

    return reward

  def internalReward(self):
    #rewards is the distance between gripper and door knob
    closestPoints = p.getClosestPoints(self.doorUid, self._kuka.kukaUid, 1000, \
            linkIndexA=2, linkIndexB=self._kuka.kukaEndEffectorIndex)
    reward = -1000
    numPt = len(closestPoints)
    if (numPt>0):
      reward = -closestPoints[0][8]*10
    return reward
