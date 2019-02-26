import os
import numpy as np
import pybullet as p
from . import kuka
import random
from kuka.kukaContiCamEnv import KukaContiCamEnv

class KukaContiCamOpenDoorEnv(KukaContiCamEnv):
  def __init__(self, renders=False):
    super(KukaContiCamOpenDoorEnv, self).__init__(renders=renders)
    '''
    self.viewMat = [-1.0, 5.355624210778842e-08, -6.160941268262832e-08, 0.0, -8.16332672570752e-08, -0.6560590267181396, \
            0.7547096014022827, 0.0, 0.0, 0.7547096014022827, 0.6560590267181396, 0.0, 0.5199999809265137, 0.11784231662750244, \
            -1.5674786567687988, 1.0]
    self.projMatrix = [0.69921875, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0, 0.0, \
            -0.02000020071864128, 0.0]
    '''
    self.viewMat = [1.0, 0.0, -0.0, 0.0, -0.0, 0.9998477101325989, -0.017452415078878403, 0.0, 0.0, 0.017452415078878403, \
            0.9998477101325989, 0.0, -0.7200000286102295, 0.20572884380817413, -1.6235408782958984, 1.0]
    self.projMatrix = [0.69921875, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0, 0.0, \
            -0.02000020071864128, 0.0]

  def reset(self, finalJPos=[0.006418, 0.413184, -0.011401, -1.589317, 0.005379, 1.137684, -0.006539]):
    self.terminated = 0
    self.gripper_closed = 0
    p.resetSimulation()
    p.setPhysicsEngineParameter(numSolverIterations=150)
    p.setTimeStep(self._timeStep)
    p.loadURDF(os.path.join(self._urdfRoot,"plane.urdf"),[0,0,-1])
    
    p.loadURDF(os.path.join(self._urdfRoot,"table/table.urdf"), 0.5000000,0.00000,-.820000,0.000000,0.000000,0.0,1.0)
    
    doorOrientation = p.getQuaternionFromEuler([0,0,1.570796])
    xpos = 0.9 + 0.1 * random.random() #TODO
    ypos = -0.25 + 0.1 * random.random() #TODO
    self.doorUid = p.loadURDF(os.path.join(os.environ['URDF_DATA'],"door.urdf"), [xpos, ypos, 0.0], doorOrientation)

    p.setGravity(0,0,-10)
    orn = p.getQuaternionFromEuler([0,0,0])
    jInitPos = finalJPos + [0.000048, -0.299912, 0.000000, -0.000043, 0.299960, 0.000000, -0.000200]
    self._kuka = kuka.Kuka(baseInitPos=[-0.1,0.0,0.07], jointInitPos = jInitPos, gripperInitOrn=[orn[0],orn[1],orn[2],orn[3]], \
            fingerAForce=60, fingerBForce=55, fingerTipForce=60, \
            urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
    self._envStepCounter = 0
    p.stepSimulation()
    self._observation = self.getExtendedObservation()
    return np.array(self._observation)

  def getGoodInitState(self):
    goodJointPos=[ 0.610865, 0.523599, -0.011401, -1.308997, 0.005379, 0.000000, -0.006539]
    self.reset(finalJPos=goodJointPos)
    self._observation = self.getExtendedObservation()

    return np.array(self._observation), goodJointPos[0:7]

  def getMidInitState(self):
    midJointPos=[ 0.308642, 0.468392, -0.011401, -1.449157, 0.005379, 0.568842, -0.006539]
    self.reset(finalJPos=midJointPos)
    self._observation = self.getExtendedObservation()

    return np.array(self._observation)

  def getGoodMidInitState(self):
    goodMidJointPos=[ 0.459754, 0.495996, -0.011401, -1.404077, 0.005379, 0.284421, -0.006539]
    self.reset(finalJPos=goodMidJointPos)
    self._observation = self.getExtendedObservation()

    return np.array(self._observation)

  def setGoodInitState(self, ob, jointPoses, extra=None): #extra --> door: [pos, orn]
    self.reset()
    self._kuka.setGoodInitStateEE(jointPoses, self._renders)
    #Set pos, orn, and joint angle for the door
    p.resetBasePositionAndOrientation(self.doorUid, extra[0], extra[1])

    p.stepSimulation()
    self._observation = self.getExtendedObservation()

  def getExtraInfo(self): #Current door info
    doorPos, doorOrn = p.getBasePositionAndOrientation(self.doorUid)

    return [doorPos, doorOrn]

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
