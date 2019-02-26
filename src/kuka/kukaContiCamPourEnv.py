import os
import numpy as np
import pybullet as p
from . import kuka
import random
from kuka.kukaContiCamEnv import KukaContiCamEnv

class KukaContiCamPourEnv(KukaContiCamEnv):
  def __init__(self,
               renders=False):
    super(KukaContiCamPourEnv, self).__init__(renders=renders)
    self.gripper_closed = 1
    self.viewMat = [1.0, 0.0, -0.0, 0.0, -0.0, 0.9998477101325989, -0.017452415078878403, 0.0, 0.0, 0.017452415078878403, \
            0.9998477101325989, 0.0, -0.7200000286102295, 0.20572884380817413, -1.6235408782958984, 1.0]
    self.projMatrix = [0.69921875, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0, 0.0, \
            -0.02000020071864128, 0.0]

  def reset(self, finalJPos=[0.006418, 0, -0.011401, -1.070796, 0.005379, 0.5, -1.570796]):
    self.terminated = 0
    self.gripper_closed = 1
    p.resetSimulation()
    p.setPhysicsEngineParameter(numSolverIterations=150)
    p.setTimeStep(self._timeStep)
    p.loadURDF(os.path.join(self._urdfRoot,"plane.urdf"),[0,0,-1])
    
    p.loadURDF(os.path.join(self._urdfRoot,"table/table.urdf"), 0.5000000,0.00000,-.820000,0.000000,0.000000,0.0,1.0)
    box_xpos = 0.85 + 0.15 * random.random() #TODO
    box_ypos = 0 + 0.15 * random.random() #TODO
    box_ang = 3.141593*random.random()
    box_orn = p.getQuaternionFromEuler([0,0,box_ang])

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

    #Load 2 cubes in the cup
    cup_pos, _ = p.getBasePositionAndOrientation(self.cupUid)
    self.cubeUids = []
    for i in range(2):
        self.cubeUids.append(p.loadURDF(os.path.join(self._urdfRoot,"cube_small.urdf"),\
                [cup_pos[0],cup_pos[1],(i+1)*0.1+cup_pos[2]]))
    tempJPosDiff = np.array(finalJPos) - np.array(resetInitPos)
    self._kuka.applyPosDiffAction(tempJPosDiff, self._renders)

    self.boxUid = p.loadURDF(os.path.join(os.environ['URDF_DATA'],"box.urdf"), [box_xpos,box_ypos,0], box_orn)
    p.resetJointState(self.boxUid, 1, 3.141596)

    self._envStepCounter = 0
    for i in range(100):
        p.stepSimulation()
    self._observation = self.getExtendedObservation()
    return np.array(self._observation)

  def getGoodInitState(self):
    goodJointPos=[ 0.250000, 0.250000, -0.011401, -1.589317, 0.005379, 0.400000, -1.570796]
    self.reset(finalJPos=goodJointPos)

    tempJPosDiff = [0, 0, 0, 0, 0, 0, -0.5+1.570796]
    self._kuka.applyPosDiffAction(tempJPosDiff, self._renders)
    goodJointPos[-1] = -0.5

    self._observation = self.getExtendedObservation()

    return np.array(self._observation), goodJointPos[0:7]

  def getMidInitState(self):
    midJointPos=[ 0.128209, 0.125000, -0.011401, -1.330057, 0.005379, 0.450000, -1.035398]
    self.reset(finalJPos=midJointPos)
    self._observation = self.getExtendedObservation()

    return np.array(self._observation)

  def getGoodMidInitState(self):
    goodMidJointPos=[ 0.189105, 0.187500, -0.011401, -1.459687, 0.005379, 0.425000, -1.570796]
    self.reset(finalJPos=goodMidJointPos)

    tempJPosDiff = [0, 0, 0, 0, 0, 0, -0.767699+1.570796]
    self._kuka.applyPosDiffAction(tempJPosDiff, self._renders)
    goodMidJointPos[-1] = -0.767699

    self._observation = self.getExtendedObservation()

    return np.array(self._observation)

  def setGoodInitState(self, ob, jointPoses, extra=None):
    tempJoint7 = jointPoses[-1]
    jointPoses[-1] = -1.570796
    self.reset(finalJPos=jointPoses)

    tempJPosDiff = [0, 0, 0, 0, 0, 0, tempJoint7+1.570796]
    self._kuka.applyPosDiffAction(tempJPosDiff, self._renders)
    jointPoses[-1] = tempJoint7

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
    boxJointOrn = p.getJointState(self.boxUid, 1)[0]
    contactPts = p.getContactPoints(self._kuka.kukaUid, self.cupUid)

    reward = 0.0

    if (len(contact1Pts) == 0 and len(contact2Pts) == 0 \
            and cube1Pos[2]>-0.0375 \
            and cube2Pos[2]>-0.0375 \
            and (boxJointOrn-3.141596) < 0.01 \
            and len(contactPts) > 0 \
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
