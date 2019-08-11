import os
import numpy as np
import pybullet as p
from . import kuka
import random
from kuka.kukaContiEnv import KukaContiEnv

class KukaContiStackInHandEnv(KukaContiEnv):
  def __init__(self, renders=False):
    super(KukaContiStackInHandEnv, self).__init__(renders=renders)
    self.gripper_closed = 1

  def reset(self, block1Pos=[0.51, 0.02766, 0.275], \
                  finalJPos=[0.006418, 0.325918, -0.011401, -1.589317, 0.005379, 1.224950, -0.006539, \
                             0.000048, -0.100000, 0.000000, -0.000043, 0.100000, 0.000000, -0.000200]):
    self.terminated = 0
    self.gripper_closed = 1
    p.resetSimulation()
    p.setPhysicsEngineParameter(numSolverIterations=150)
    p.setTimeStep(self._timeStep)
    p.loadURDF(os.path.join(self._urdfRoot,"plane.urdf"),[0,0,-1])
    
    p.loadURDF(os.path.join(self._urdfRoot,"table/table.urdf"), 0.5000000,0.00000,-.820000,0.000000,0.000000,0.0,1.0)
    p.loadURDF(os.path.join(self._urdfRoot,"tray/tray.urdf"), 0.640000,0.075000,-0.190000,0.000000,0.000000,1.000000,0.000000)

    p.setGravity(0,0,-10)
    ang1 = 1.570796
    orn1 = p.getQuaternionFromEuler([0,0,ang1])
    self.block1Uid =p.loadURDF(os.path.join(self._urdfRoot,"cube_small.urdf"),block1Pos[0],block1Pos[1],block1Pos[2],orn1[0],orn1[1],orn1[2],orn1[3])
    self._kuka = kuka.Kuka(baseInitPos=[-0.1,0.0,0.07], jointInitPos=finalJPos, gripperInitOrn=[orn1[0],orn1[1],orn1[2],orn1[3]], \
            fingerAForce=60, fingerBForce=55, fingerTipForce=60, \
            urdfRootPath=self._urdfRoot, timeStep=self._timeStep)

    xpos2 = 0.5 +0.05*random.random()
    ypos2 = 0 +0.05*random.random()
    ang2 = 3.1415925438*random.random()
    orn2 = p.getQuaternionFromEuler([0,0,ang2])
    self.block2Uid =p.loadURDF(os.path.join(self._urdfRoot,"cube_small.urdf"), xpos2,ypos2,-0.1,orn2[0],orn2[1],orn2[2],orn2[3])

    self._envStepCounter = 0
    p.stepSimulation()
    self._observation = self.getExtendedObservation()
    return np.array(self._observation)

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
    block1Pos = [0.5675, 0.02766, -0.03]
    goodJointPos=[0.006418, 0.872665, -0.011401, -1.589317, 0.005379, 0.698132, -0.006539, \
            0.000048, -0.100000, 0.000000, -0.000043, 0.100000, 0.000000, -0.000200]
    self.reset(block1Pos=block1Pos, finalJPos=goodJointPos)
    self._observation = self.getExtendedObservation()

    return np.array(self._observation), goodJointPos[0:7]

  def getMidInitState(self):
    block1Pos = [0.568, 0.02766, 0.02]
    midJointPos=[ 0.006418, 0.785398, -0.011401, -1.589317, 0.005379, 0.785398, -0.006539, \
            0.000048, -0.100000, 0.000000, -0.000043, 0.100000, 0.000000, -0.000200]
    self.reset(block1Pos=block1Pos, finalJPos=midJointPos)
    self._observation = self.getExtendedObservation()

    return np.array(self._observation)

  def setGoodInitState(self, ob, jointPoses, extra=None):
    self.reset()
    self._kuka.setGoodInitStateEE(jointPoses, self._renders)
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

  def _termination(self):
    state = p.getLinkState(self._kuka.kukaUid,self._kuka.kukaEndEffectorIndex)
    actualEndEffectorPos = state[0]
 
    if (self.terminated or self._envStepCounter > 10):
      self._observation = self.getExtendedObservation()
      return True
    
    if (actualEndEffectorPos[2] <= 0.20):
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
    
    #rewards is height of target object and the xy distance between two blocks
    block1Pos,_=p.getBasePositionAndOrientation(self.block1Uid)
    block2Pos,_=p.getBasePositionAndOrientation(self.block2Uid)
    dis = np.linalg.norm(np.array(block1Pos[:2])-np.array(block2Pos[:2]))

    reward = 0.0

    if (block1Pos[2] > -0.125 and dis < 0.070711 and self.terminated and not self.gripper_closed):
      #print("stacked a block!!!")
      #print("self._envStepCounter")
      #print(self._envStepCounter)
      #reward = reward+1000
      reward = 1.0

    return reward

  def internalReward(self):
    #rewards is the distance between block1 and block2
    closestPoints = p.getClosestPoints(self.block1Uid,self.block2Uid,1000)
    reward = -1000
    numPt = len(closestPoints)
    if (numPt>0):
      reward = -closestPoints[0][8]*10
    return reward
