#! /usr/bin/env python

import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import rospy
import platform #To check version
from std_msgs.msg import Int16
moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node('move_group_python_interface_tutorial', anonymous=True)

robot = moveit_commander.RobotCommander()
scene = moveit_commander.PlanningSceneInterface()
group = moveit_commander.MoveGroupCommander("arm")
#gripper_group = moveit_commander.MoveGroupCommander("gripper")
display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path', moveit_msgs.msg.DisplayTrajectory, queue_size=1)

#Position library
positions = [None]*10
positions[0] = [0.0,0.0,0.0,0.0] #BASE
positions[1] = [1.5,0.6,-0.6,00.0] #Bot left
positions[2] = [0.0,0.0,0.0,0.0] #Bot
positions[3] = [-1.5,0.6,-0.6,0.0] #Bot right
positions[4] = [1.0,0.75,-0.75,0.0] #Middle left
positions[5] = [0.0,0.75,-0.75,0.0] #Middle
positions[6] = [-1.0,0.75,-0.75,0.0] #Middle right
positions[7] = [0.5,1.5,0.0,-1.5] #Top left
positions[8] = [0.0,1.5,0.0,-1.5] #Top
positions[9] = [-0.5,1.5,0.0,-1.5] #Top right

if(len(sys.argv) == 1):
	print("NO COMMAND ENTERED")
else:
	user = sys.argv[1]
	if (0 <= int(user) < 10):
		print("Command: Move to Pos." +user)
		pos = positions[int(user)]
		group_variable_values = group.get_current_joint_values()
		group_variable_values[0] = pos[0]
		group_variable_values[1] = pos[1]
		group_variable_values[2] = pos[2]
		group_variable_values[3] = pos[3]
		group.set_joint_value_target(group_variable_values)
		group.set_planning_time(10); #Py2
		plan = group.plan()
		group.go(wait=True)
	elif (10 <= int(user) < 14):
		if (int(user) == 10):
			print("Command: Move Up")
			group_variable_values = group.get_current_joint_values()
			group_variable_values[1] = max(0, group_variable_values[1]-0.25)
			group.set_joint_value_target(group_variable_values)
			group.set_planning_time(10); #Py2
			plan = group.plan()
			group.go(wait=True)
		elif (int(user) == 11):
			print("Command: Move Down")
			group_variable_values = group.get_current_joint_values()
			group_variable_values[1] = min(1.5, group_variable_values[1]+0.25)
			group.set_joint_value_target(group_variable_values)
			group.set_planning_time(10); #Py2
			plan = group.plan()
			group.go(wait=True)
		elif (int(user) == 12):
			print("Command: Move Left")
			group_variable_values = group.get_current_joint_values()
			group_variable_values[0] = max(-1.5, group_variable_values[0]-0.25)
			group.set_joint_value_target(group_variable_values)
			group.set_planning_time(10); #Py2
			plan = group.plan()
			group.go(wait=True)
		elif (int(user) == 13):
			print("Command: Move Right")
			group_variable_values = group.get_current_joint_values()
			group_variable_values[0] = min(1.5, group_variable_values[0]+0.25)
			group.set_joint_value_target(group_variable_values)
			group.set_planning_time(10); #Py2
			plan = group.plan()
			group.go(wait=True)
		else:
			print("Unexpected value. Please try again.")
	else:
		print("Invalid position index. Please try again.")

print("Executed successfully.")
moveit_commander.roscpp_shutdown()
