import os
import numpy as np
import pybullet as p
from . import kuka
import random
from kuka.kukaContiEnv import KukaContiEnv

class KukaContiKeyInsertionEnv(KukaContiEnv):
  def __init__(self,
               renders=False):
    super(KukaContiKeyInsertionEnv, self).__init__(renders=renders)
    self.gripper_closed = 1

  def reset(self, finalJPos=[0.006418, 0, -0.011401, -0.785398, 0.005379, 0, 1.570796]):
    self.terminated = 0
    self.gripper_closed = 1
    p.resetSimulation()
    p.setPhysicsEngineParameter(numSolverIterations=150)
    p.setTimeStep(self._timeStep)
    p.loadURDF(os.path.join(self._urdfRoot,"plane.urdf"),[0,0,-1])
    
    p.loadURDF(os.path.join(self._urdfRoot,"table/table.urdf"), 0.5000000,0.00000,-.820000,0.000000,0.000000,0.0,1.0)

    #Load a key for the gripper to grasp in hand
    xpos1 = 0.0
    ypos1 = 0.0
    ang1 = 1.570796
    orn1 = p.getQuaternionFromEuler([0,-ang1,0])
    
    p.setGravity(0,0,-10)
    jInitPos=[ 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.570796, \
            0.000048, -0.040000, 0.000000, -0.000043, 0.040000, 0.000000, -0.000200 ]
    self._kuka = kuka.Kuka(baseInitPos=[-0.1,0.0,0.07], jointInitPos=jInitPos, gripperInitOrn=[orn1[0],orn1[1],orn1[2],orn1[3]], \
            fingerAForce=60, fingerBForce=55, fingerTipForce=60, \
            urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
    self.keyUid =p.loadURDF(os.path.join(os.environ['URDF_DATA'],"key.urdf"), xpos1,ypos1,1.5,orn1[0],orn1[1],orn1[2],orn1[3])

    tempJPosDiff = np.array(finalJPos) - np.array(jInitPos[0:7])
    self._kuka.applyPosDiffAction(tempJPosDiff, self._renders)

    xpos2 = 0.9 + 0.05*random.random()
    ypos2 = 0 + 0.05*random.random()
    keyholeOrientation = p.getQuaternionFromEuler([0,0,0])
    self.keyholeUid =p.loadURDF(os.path.join(os.environ['URDF_DATA'],"keyhole.urdf"), [xpos2,ypos2,0.0], keyholeOrientation)

    self._envStepCounter = 0
    p.stepSimulation()
    self._observation = self.getExtendedObservation()
    return np.array(self._observation)

  def getExtendedObservation(self):
     self._observation = self._kuka.getObservation()
     eeState  = p.getLinkState(self._kuka.kukaUid,self._kuka.kukaEndEffectorIndex)
     endEffectorPos = eeState[0]
     endEffectorOrn = eeState[1]
     keyholePos,keyholeOrn = p.getBasePositionAndOrientation(self.keyholeUid)

     invEEPos,invEEOrn = p.invertTransform(endEffectorPos,endEffectorOrn)
     keyholePosInEE,keyholeOrnInEE = p.multiplyTransforms(invEEPos,invEEOrn,keyholePos,keyholeOrn)
     keyholeEulerInEE = p.getEulerFromQuaternion(keyholeOrnInEE)
     self._observation.extend(list(keyholePosInEE))
     self._observation.extend(list(keyholeEulerInEE))

     return self._observation

  def getGoodInitState(self):
    self.reset()
    goodJointPos=[0.006418, 0.713184, -0.011401, -1.589317, 0.005379, -0.713184, 1.570796]
    self._kuka.initState(goodJointPos, self._renders)
    self._observation = self.getExtendedObservation()

    return np.array(self._observation), goodJointPos[0:7]

  def getMidInitState(self):
    midJointPos=[ 0.006418, 0.356592, -0.011401, -1.187358, 0.005379, -0.356592, 1.570796]
    self.reset(finalJPos=midJointPos)
    self._observation = self.getExtendedObservation()

    return np.array(self._observation)

  def getGoodMidInitState(self):
    goodMidJointPos=[ 0.006418, 0.534888, -0.011401, -1.388338, 0.005379, -0.534888, 1.570796]
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
    #Set pos and orn for the keyhole
    keyholeOrnInEE = p.getQuaternionFromEuler(ob[16:19])
    keyholePos, keyholeOrn = p.multiplyTransforms(gripperPos, gripperOrn, ob[13:16], keyholeOrnInEE)
    p.resetBasePositionAndOrientation(self.keyholeUid, keyholePos, keyholeOrn)

    p.stepSimulation()
    self._observation = self.getExtendedObservation()

  def _termination(self):
    keyPos, _ = p.getBasePositionAndOrientation(self.keyUid)
 
    if (self.terminated or self._envStepCounter > 10):
      self._observation = self.getExtendedObservation()
      return True
    
    if (keyPos[2] <= 0.3):
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
        
      self._observation = self.getExtendedObservation()
      return True

    return False

  def _reward(self):
    
    #reward is the height of key and xy distance between key and keyhole
    keyPos,_ = p.getBasePositionAndOrientation(self.keyUid)
    keyholePos,_ = p.getBasePositionAndOrientation(self.keyholeUid)
    dis = np.linalg.norm(np.array(keyPos[:2])-np.array(keyholePos[:2]))

    reward = 0.0

    if (keyPos[2] < 0.1 and dis < 0.05 and self.terminated and not self.gripper_closed):
      #print("key inserted!!!")
      #print("self._envStepCounter")
      #print(self._envStepCounter)
      #reward = reward+1000
      reward = 1.0

    return reward

  def internalReward(self):
    #reward is the distance between key and keyhole
    closestPoints = p.getClosestPoints(self.keyUid, self.keyholeUid, 1000)
    reward = -1000
    numPt = len(closestPoints)
    if (numPt>0):
      reward = -closestPoints[0][8]*10
    return reward
