import os
import numpy as np
import pybullet as p
from . import kuka
import random
from kuka.kukaContiEnv import KukaContiEnv

class KukaContiScoopEnv(KukaContiEnv):
  def __init__(self, renders=False):
    super(KukaContiScoopEnv, self).__init__(renders=renders)

  def reset(self):
    self.terminated = 0
    p.resetSimulation()
    p.setPhysicsEngineParameter(numSolverIterations=150)
    p.setTimeStep(self._timeStep)
    p.loadURDF(os.path.join(self._urdfRoot,"plane.urdf"),[0,0,-1])
    
    p.loadURDF(os.path.join(self._urdfRoot,"table/table.urdf"), 0.5000000,0.00000,-.820000,0.000000,0.000000,0.0,1.0)
    p.loadURDF(os.path.join(self._urdfRoot,"tray/tray.urdf"), 0.640000,0.075000,-0.190000,0.000000,0.000000,1.000000,0.000000)
    
    p.setGravity(0,0,-10)
    ang = 3.1415925438*random.random()
    orn = p.getQuaternionFromEuler([0,0,ang])
    jInitPos = [0.006418, 1.097197, -0.011401, -1.589317, 0.005379, 0.523598, -0.006539, \
            0.000048, 0.0, 0.000000, -0.000043, 0.0, 0.000000, -0.000200]
    self._kuka = kuka.Kuka(baseInitPos=[-0.1,0.0,0.07], jointInitPos=jInitPos, gripperInitOrn=[orn[0],orn[1],orn[2],orn[3]], \
            urdfRootPath=self._urdfRoot, timeStep=self._timeStep)

    endEffectorPos = p.getLinkState(self._kuka.kukaUid,self._kuka.kukaEndEffectorIndex)[0]
    xpos = np.clip(endEffectorPos[0] + 0.05*random.choice([-1,1]) + 0.05*random.random(), 0.455, 0.69)
    ypos = np.clip(endEffectorPos[1] + 0.05*random.choice([-1,1]) + 0.05*random.random(), -0.125, 0.25)
    self.blockUid =p.loadURDF(os.path.join(self._urdfRoot,"cube_small.urdf"),xpos,ypos,-0.159,orn[0],orn[1],orn[2],orn[3])

    self._envStepCounter = 0
    p.stepSimulation()
    self._observation = self.getExtendedObservation()
    return np.array(self._observation)

  def _termination(self):
    blockPos,_=p.getBasePositionAndOrientation(self.blockUid)
 
    if (blockPos[2] >= -0.125 or self._envStepCounter > 10):
      self.terminated = 1
      self._observation = self.getExtendedObservation()
      return True

    return False
   
  def _reward(self):
    
    #rewards is height of target object
    blockPos,_=p.getBasePositionAndOrientation(self.blockUid)

    reward = 0.0

    if (blockPos[2] > -0.125 and self.terminated):
      reward = 1.0

    return reward
