import os
import numpy as np
import pybullet as p
from . import kuka
import random
from kuka.kukaContiCamEnv import KukaContiCamEnv

class KukaContiCamGraspEnv(KukaContiCamEnv):
  def __init__(self, renders=False):
    super(KukaContiCamGraspEnv, self).__init__(renders=renders)
    '''
    self.viewMat = [-1.0, 5.31451931351512e-08, -6.113655359740733e-08, 0.0, -8.100672488353666e-08, -0.6560590267181396, \
            0.7547096014022827, 0.0, 0.0, 0.7547096014022827, 0.6560590267181396, 0.0, 0.5199999809265137, 0.11784233152866364, \
            -0.6075014472007751, 1.0]
    self.projMatrix = [0.69921875, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0, 0.0, \
            -0.02000020071864128, 0.0]
    '''
    self.viewMat = [1.0, 0.0, -0.0, 0.0, -0.0, 0.9998477101325989, -0.017452415078878403, 0.0, 0.0, 0.017452415078878403, \
            0.9998477101325989, 0.0, -0.7200000286102295, 0.20572884380817413, -1.6235408782958984, 1.0]
    self.projMatrix = [0.69921875, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0, 0.0, \
            -0.02000020071864128, 0.0]

  def reset(self, finalJPos=None):
    self.terminated = 0
    self.gripper_closed = 0
    p.resetSimulation()
    p.setPhysicsEngineParameter(numSolverIterations=150)
    p.setTimeStep(self._timeStep)
    p.loadURDF(os.path.join(self._urdfRoot,"plane.urdf"),[0,0,-1])
    
    p.loadURDF(os.path.join(self._urdfRoot,"table/table.urdf"), 0.5000000,0.00000,-.820000,0.000000,0.000000,0.0,1.0)
    p.loadURDF(os.path.join(self._urdfRoot,"tray/tray.urdf"), 0.640000,0.075000,-0.190000,0.000000,0.000000,1.000000,0.000000)
    
    xpos = 0.5 + 0.2*random.random() #TODO
    ypos = 0 + 0.25*random.random() #TODO
    ang = 3.1415925438*random.random()
    orn = p.getQuaternionFromEuler([0,0,ang])
    self.blockUid =p.loadURDF(os.path.join(self._urdfRoot,"block.urdf"), xpos,ypos,-0.1,orn[0],orn[1],orn[2],orn[3])

    p.setGravity(0,0,-10)
    jInitPos = [0.006418, 0.413184, -0.011401, -1.589317, 0.005379, 1.137684, -0.006539, \
            0.000048, -0.299912, 0.000000, -0.000043, 0.299960, 0.000000, -0.000200]
    self._kuka = kuka.Kuka(baseInitPos=[-0.1,0.0,0.07], jointInitPos=jInitPos, gripperInitOrn=[orn[0],orn[1],orn[2],orn[3]], \
            urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
    self._envStepCounter = 0
    p.stepSimulation()
    self._observation = self.getExtendedObservation()
    return np.array(self._observation)

  def getGoodInitState(self):
    self.reset()
    goodJointPos=[ 0.006418, 1.047197, -0.011401, -1.589317, 0.005379, 0.523598, -0.006539]
    self._kuka.initState(goodJointPos, self._renders)
    self._observation = self.getExtendedObservation()

    return np.array(self._observation), goodJointPos[0:7]

  def getMidInitState(self):
    self.reset()
    midJointPos=[ 0.006418, 0.785398, -0.011401, -1.589317, 0.005379, 0.785398, -0.006539]
    self._kuka.initState(midJointPos, self._renders)
    self._observation = self.getExtendedObservation()

    return np.array(self._observation)

  def getGoodMidInitState(self):
    self.reset()
    goodMidJointPos=[ 0.006418, 0.916298, -0.011401, -1.589317, 0.005379, 0.654498, -0.006539]
    self._kuka.initState(goodMidJointPos, self._renders)
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
    p.resetBasePositionAndOrientation(self.blockUid, blockPos, blockOrn)

    p.stepSimulation()
    self._observation = self.getExtendedObservation()

  def _termination(self):
    state = p.getLinkState(self._kuka.kukaUid,self._kuka.kukaEndEffectorIndex)
    actualEndEffectorPos = state[0]
 
    if (self.terminated or self._envStepCounter > 10):
      self._observation = self.getExtendedObservation()
      return True
    
    if (actualEndEffectorPos[2] <= 0.10):
      self.terminated = 1
      
      #print("closing gripper, attempting grasp")
      self.gripper_closed = 1
      #start grasp and terminate
      fingerAngle = 0.3
      
      for i in range (1000):
        graspAction = [0,0,0.001,0,fingerAngle]
        self._kuka.applyAction(graspAction)
        p.stepSimulation()
        fingerAngle = fingerAngle-(0.3/100.)
        if (fingerAngle<0):
          fingerAngle=0
        
      self._observation = self.getExtendedObservation()
      return True

    return False
   
  def _reward(self):
    
    #rewards is height of target object
    blockPos,_=p.getBasePositionAndOrientation(self.blockUid)

    reward = 0.0

    if (blockPos[2] >0.2 and self.terminated and self.gripper_closed):
      #print("grasped a block!!!")
      #print("self._envStepCounter")
      #print(self._envStepCounter)
      #reward = reward+1000
      reward = 1.0

    return reward

  def internalReward(self):
    #rewards is the distance between gripper and target object
    closestPoints = p.getClosestPoints(self.blockUid, self._kuka.kukaUid, 1000, linkIndexB=self._kuka.kukaEndEffectorIndex)
    reward = -1000
    numPt = len(closestPoints)
    if (numPt>0):
      reward = -closestPoints[0][8]*10
    return reward
