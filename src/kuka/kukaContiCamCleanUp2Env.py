import os
import numpy as np
import pybullet as p
from . import kuka
import random
from kuka.kukaContiCamEnv import KukaContiCamEnv

class KukaContiCamCleanUp2Env(KukaContiCamEnv):
  def __init__(self,
               renders=False):
    super(KukaContiCamCleanUp2Env, self).__init__(renders=renders)
    self.gripper_closed = 1
    '''
    self.viewMat = [-1.0, 1.3315725766460673e-07, -4.5849727570157484e-08, 0.0, -1.4082986865560088e-07, -0.9455186128616333, \
            0.32556819915771484, 0.0, 0.0, 0.32556819915771484, 0.9455186128616333, 0.0, 0.5199999809265137, -0.08166629076004028, \
            -0.9228652715682983, 1.0]
    self.projMatrix = [0.69921875, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0, 0.0, \
            -0.02000020071864128, 0.0]
    '''
    self.viewMat = [1.0, 0.0, -0.0, 0.0, -0.0, 0.9998477101325989, -0.017452415078878403, 0.0, 0.0, 0.017452415078878403, \
            0.9998477101325989, 0.0, -0.7200000286102295, 0.20572884380817413, -1.6235408782958984, 1.0]
    self.projMatrix = [0.69921875, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0, 0.0, \
            -0.02000020071864128, 0.0]

  def reset(self, finalJPos=[0.006418, 0.325918, -0.011401, -1.589317, 0.005379, 1.224950, -0.006539]):
    self.terminated = 0
    self.gripper_closed = 1
    p.resetSimulation()
    p.setPhysicsEngineParameter(numSolverIterations=150)
    p.setTimeStep(self._timeStep)
    p.loadURDF(os.path.join(self._urdfRoot,"plane.urdf"),[0,0,-1])
    
    p.loadURDF(os.path.join(self._urdfRoot,"table/table.urdf"), 0.5000000,0.00000,-.820000,0.000000,0.000000,0.0,1.0)
    box_xpos = 0.8 + 0.15 * random.random() #TODO
    box_ypos = 0 + 0.15 * random.random() #TODO
    box_ang = -1.570796
    box_orn = p.getQuaternionFromEuler([0,0,box_ang])
    self.boxUid = p.loadURDF(os.path.join(os.environ['URDF_DATA'],"box.urdf"), [box_xpos,box_ypos,0], box_orn)

    #Load a block for the gripper to grasp in hand
    xpos = 0.525
    ypos = 0.025
    ang = 1.570796
    orn = p.getQuaternionFromEuler([0,0,ang])
    
    p.setGravity(0,0,-10)
    jInitPos=[ 0.006418, 1.134464, -0.011401, -1.589317, 0.005379, 0.436332, -0.006539, \
            0.000048, -0.299912, 0.000000, -0.000043, 0.299960, 0.000000, -0.000200 ]
    self._kuka = kuka.Kuka(baseInitPos=[-0.1,0.0,0.07], jointInitPos=jInitPos, gripperInitOrn=[orn[0],orn[1],orn[2],orn[3]], \
            fingerAForce=60, fingerBForce=55, fingerTipForce=60, \
            urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
    self.blockUid =p.loadURDF(os.path.join(self._urdfRoot,"cube_small.urdf"), xpos,ypos,-0.1,orn[0],orn[1],orn[2],orn[3])

    fingerAngle = 0.3
      
    for i in range (1000):
        graspAction = [0,0,0,0,fingerAngle]
        self._kuka.applyAction(graspAction)
        p.stepSimulation()
        fingerAngle = fingerAngle-(0.3/100.)
        if (fingerAngle<0):
            fingerAngle=0

    tempJPosDiff = np.array(finalJPos) - np.array(jInitPos[0:7])
    self._kuka.applyPosDiffAction(tempJPosDiff, self._renders)

    self._envStepCounter = 0
    p.stepSimulation()
    self._observation = self.getExtendedObservation()
    return np.array(self._observation)

  def getGoodInitState(self):
    goodJointPos=[ 0.006418, 0.850000, -0.011401, -1.589317, 0.005379, 0.350000, -0.006539]
    self.reset(finalJPos=goodJointPos)
    self._observation = self.getExtendedObservation()

    return np.array(self._observation), goodJointPos[0:7]

  def getMidInitState(self):
    midJointPos=[ 0.006418, 0.587959, -0.011401, -1.589317, 0.005379, 0.787475, -0.006539]
    self.reset(finalJPos=midJointPos)
    self._observation = self.getExtendedObservation()

    return np.array(self._observation)

  def setGoodInitState(self, ob, jointPoses, extra=None):
    self.reset()
    self._kuka.setGoodInitStateEE(jointPoses, self._renders)
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

  def _termination(self):
    boxJointPos = p.getJointState(self.boxUid, 1)[0]
 
    if (self.terminated or self._envStepCounter > 10):
      self._observation = self.getExtendedObservation()
      return True
    
    if (boxJointPos >= 0.523599): 
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
    
    #reward is if the block is placed into the box
    contactPts = p.getContactPoints(self.blockUid, self.boxUid)
    blockPos, _ = p.getBasePositionAndOrientation(self.blockUid)

    reward = 0.0

    if (len(contactPts)>0 and blockPos[2]<0.05 and blockPos[2]>-0.05 \
            and self.terminated and not self.gripper_closed):
      #print("clean up!!!")
      #print("self._envStepCounter")
      #print(self._envStepCounter)
      #reward = reward+1000
      reward = 1.0

    return reward

  def internalReward(self):
    #rewards is the distance between block and box
    closestPoints = p.getClosestPoints(self.blockUid,self.boxUid,1000)
    reward = -1000
    numPt = len(closestPoints)
    if (numPt>0):
      reward = -closestPoints[0][8]*10
    return reward
