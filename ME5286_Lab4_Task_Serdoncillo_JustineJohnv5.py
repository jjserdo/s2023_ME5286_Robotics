# This macro shows an example of running a program on the robot using the Python API (online programming)
# More information about the RoboDK API appears here:
# https://robodk.com/doc/en/RoboDK-API.html
##########################################
# Created by Justine John A. Serdoncillo #
# for ME 5286 Robotics: Lab 4            #
# Flashlight Assembly                    #
# started working on March 7, 2023       #
# University of Minnesota - Twin Cities  #
##########################################
# Version 5 Updates:
    # Easier access to changing joint speeds and accelerations
    # Twist count works again
    
import sys
sys.path.append("C:\RoboDK\Python")
# %%
from robolink import *    # API to communicate with RoboDK
from robodk import *      # robodk robotics toolbox
RDK = Robolink()
robot = RDK.ItemUserPick('Select a robot', ITEM_TYPE_ROBOT)
if not robot.Valid():
    raise Exception('No robot selected or available')
RUN_ON_ROBOT = False
if RDK.RunMode() != RUNMODE_SIMULATE:
    RUN_ON_ROBOT = False
if RUN_ON_ROBOT:
    # Connect to the robot using default IP
    success = robot.Connect() # Try to connect once
    status, status_msg = robot.ConnectedState()
    if status != ROBOTCOM_READY:
        # Stop if the connection did not succeed
        print(status_msg)
        raise Exception("Failed to connect: " + status_msg)
    # This will set to run the API programs on the robot and the simulator (online programming)
    RDK.setRunMode(RUNMODE_RUN_ROBOT)
joints_ref = robot.Joints()
target_ref = robot.Pose()
pos_ref = target_ref.Pos()
robot.setPoseFrame(robot.PoseFrame())
robot.setPoseTool(robot.PoseTool())
# %%
# JJ Code down below
import numpy as np

linspeed = [50,50,50,50] 
linaccel = [100,100,100,100]
joispeed = [15,15,15,15]
joiaccel = [60,60,60,60]
blends = [2,2,2,2]
##################################################################
gripper_force = [127,127,127,127] 
gripper_speed = [ 50, 50, 50, 50] 
pos = np.array([ [0, 0, 0, 0] , [0, 9, 6, 3] ]) * 25.4
##################################################################
# input values here
tray_og =  [-389.00 ,-494.00] # origin of the tray in x,y,z
chuck_og = [-322.00 ,  40.00] # origin of the chuck in x,y,z
#-388.22 ,-492.26
#-322.39 ,  39.89: old chuck_og and not centered
#-320.37 ,  39.9
#deg = [130,122,4] # ideal orientation
#deg = [-180,0,4] # from Lab 3 but hitting other components
#deg = [-176.79,-1.10,-89.55]
deg = [-180,0,-90]
tray_z  = np.array([39,50,43,38]) + 33 # Lab 3 has z=70 but these measurements s 
chuck_z = np.array([140,185,225,227]) + 40 # 2 and 3 not sure

twist = [False,True,False,True]
#twist = [False, False, False, False]
twist_num = [10,7]
#twist_num = [6,3]
#twist_num = [1,1]
twist_torque = [3,2]
#############################################
theta_1 = 0
theta_2 = np.pi
cB = np.cos(theta_1)
sB = np.sin(theta_1)
cC = np.cos(theta_2)
sC = np.sin(theta_2)
TB = tray_og.copy()
TC = chuck_og.copy()
RTAB = np.array([[cB,-sB,TB[0]],[sB,cB,TB[1]],[0,0,1]])
RTAC = np.array([[cC,-sC,TC[0]],[sC,cC,TC[1]],[0,0,1]])

tray   = np.zeros((4,6))
chuck  = np.zeros((4,6))
tray[:,3] = deg[0]
tray[:,4] = deg[1]
tray[:,5] = deg[2]
chuck[:,0] = TC[0]
chuck[:,1] = TC[1]
chuck[:,3] = deg[0]
chuck[:,4] = deg[1]
chuck[:,5] = deg[2]
for j in range(4):
  P1B = np.array([[pos[0,j]],[pos[1,j]],[1]])
  P1A = RTAB @ P1B
  tray[j,0] = P1A[0]
  tray[j,1] = P1A[1]
  tray[j,2] = tray_z[j]
  chuck[j,2] = chuck_z[j]
tray_up = tray.copy()
tray_up[:,2] += 30
chuck_up = chuck.copy()
chuck_up[:,2] += 110
  
twist_before = np.zeros((2,6))
twist_before[0] = [-27.14,-115.32,117.34,-92.04,-89.49,-206.91]
twist_before[1] = [-27.14,-115.32,117.34,-92.04,-89.49,-206.91]
twist_int   = twist_before.copy()
twist_after = twist_before.copy()

twist_int[:,5] += 65
twist_after[:,5] += 130
twist_count = 0

###################################################################
robot.setSpeed(50) 
robot.setAcceleration(100)
robot.setSpeedJoints(15) 
robot.setAccelerationJoints(60)
robot.setZoneData(2) 

home = [0, -90, 0, -90, 0, 0]
robot.MoveJ(home)
safe = [35.30,-90.35,87.92,-90.26,-90.32,-43.42]
robot.MoveJ(safe)

robot.RunCodeCustom('clamp()',INSTRUCTION_INSERT_CODE)
robot.RunCodeCustom('unclamp()',INSTRUCTION_INSERT_CODE)
for i in np.arange(0,4):
    robot.setSpeed(linspeed[i]) 
    robot.setAcceleration(linaccel[i]) 
    robot.setSpeedJoints(joispeed[i]) 
    robot.setAccelerationJoints(joiaccel[i]) 
    robot.setZoneData(blends[i]) 
    robot.RunCodeCustom('rq_set_force(%i)' %gripper_force[i], INSTRUCTION_CALL_PROGRAM)
    robot.RunCodeCustom('rq_set_speed(%i)' %gripper_speed[i], INSTRUCTION_CALL_PROGRAM)
    
    robot.MoveL(xyzrpw_2_pose(tray[i]))
    
    robot.RunCodeCustom('rq_close_and_wait()',INSTRUCTION_CALL_PROGRAM)
    
    robot.MoveL(xyzrpw_2_pose(tray_up[i]))
    robot.MoveL(xyzrpw_2_pose(chuck_up[i]))
    robot.MoveL(xyzrpw_2_pose(chuck[i]))
    if i != 3:  
        robot.RunCodeCustom('rq_open_and_wait()',INSTRUCTION_CALL_PROGRAM)
    if twist[i] == True:
        robot.MoveJ(twist_before[twist_count].tolist())
        for tnum in range(twist_num[twist_count]):
            robot.MoveJ(twist_before[twist_count].tolist())  
            robot.RunCodeCustom('rq_close_and_wait()',INSTRUCTION_CALL_PROGRAM) 
            robot.MoveJ(twist_int[twist_count].tolist())
            robot.MoveJ(twist_after[twist_count].tolist())
            robot.RunCodeCustom('rq_open_and_wait()',INSTRUCTION_CALL_PROGRAM) 
            robot.MoveJ(twist_before[twist_count].tolist())
        robot.RunCodeCustom('tighten_torque(%i,0,1.57,2,2,1,100,100,50)' %twist_torque[twist_count],INSTRUCTION_INSERT_CODE)
        twist_count += 1
    if i == 0:
        robot.RunCodeCustom('clamp()',INSTRUCTION_INSERT_CODE) 
    elif i == 3:
        robot.RunCodeCustom('unclamp()',INSTRUCTION_INSERT_CODE)
    robot.MoveL(xyzrpw_2_pose(chuck_up[i]))

robot.MoveJ(safe)
robot.MoveJ(home)
