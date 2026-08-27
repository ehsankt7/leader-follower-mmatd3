import math
import os
import random
import subprocess
import time
from os import path
import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from squaternion import Quaternion
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from .rewards import follower_reward, leader_reward

GOAL_REACHED_DIST = 0.4
COLLISION_DIST = 0.35
TIME_DELTA = 0.01
# Check if the random goal position is located on an obstacle and do not accept it if it is
def check_pos(x, y):
    goal_ok = True
    ##### STAGE 1 #################################################################
    # wall number 1:
    if  -1.75 < x < 1.75 and 4 < y:
        goal_ok = False
                        
    # wall number 2:
    if  x < -4 and 1.25 < y < 4.75:
        goal_ok = False   

    # wall number 4:
    if  x > 4 and -4.25 < y < -1.75:
        goal_ok = False 

    # X wall:
    if  -4.25 < x < -1.75 and -4.25 < y < -1.75:
        goal_ok = False 

    # cylinder number 1:
    if  0 < x < 4 and -0.75 < y < 2.75:
        goal_ok = False 

    # cylinder number 2:
    if  0.25 < x < 3.75 and -6.5 < y < -3.75:
        goal_ok = False 

    # room walls    
    if x > 6.25 or x < -6.25 or y > 6.25 or y < -6.25:
        goal_ok = False
    return goal_ok



class GazeboEnv:
    """Superclass for all Gazebo environments."""

    def __init__(self, launchfile, environment_dim):
        self.environment_dim = environment_dim
        self.i_decay = 0
        ## initialize the position for all of the agents and their corresponding goal points 
        self.pre_odom_x_r1 = 0
        self.pre_odom_y_r1 = 0
        self.pre_odom_x_r2 = 0
        self.pre_odom_y_r2 = 0
        
        self.odom_x_r1 = 0
        self.odom_y_r1 = 0
        self.odom_x_r2 = 0
        self.odom_y_r2 = 0

        self.goal_x = 1
        self.goal_y = 0.0


        ## change these values based on our own created gazebo world
        # self.upper = 15.5
        # self.lower = -15.5
        self.velodyne_data_r1 = np.ones(int(self.environment_dim/2)) * 16
        self.velodyne_data_r2 = np.ones(int(self.environment_dim/2)) * 16
        self.last_odom_r1 = None
        self.last_odom_r2 = None

        ## create model robot instances for all the agents
        self.set_self_state_r1 = ModelState()
        self.set_self_state_r1.model_name = "r1"
        self.set_self_state_r1.pose.position.x = 0.0
        self.set_self_state_r1.pose.position.y = 0.0
        self.set_self_state_r1.pose.position.z = 0.0
        self.set_self_state_r1.pose.orientation.x = 0.0
        self.set_self_state_r1.pose.orientation.y = 0.0
        self.set_self_state_r1.pose.orientation.z = 0.0
        self.set_self_state_r1.pose.orientation.w = 1.0
        
        
        ## create model robot instances for all the agents
        self.set_self_state_r2 = ModelState()
        self.set_self_state_r2.model_name = "r2"
        self.set_self_state_r2.pose.position.x = 1.0
        self.set_self_state_r2.pose.position.y = 1.0
        self.set_self_state_r2.pose.position.z = 0.0
        self.set_self_state_r2.pose.orientation.x = 0.0
        self.set_self_state_r2.pose.orientation.y = 0.0
        self.set_self_state_r2.pose.orientation.z = 0.0
        self.set_self_state_r2.pose.orientation.w = 1.0


        self.gaps_r1 = [[-np.pi / 2 - 0.03, -np.pi / 2 + np.pi / (self.environment_dim/2)]]
        for m in range((int(self.environment_dim/2)) - 1):
            self.gaps_r1.append(
                [self.gaps_r1[m][1], self.gaps_r1[m][1] + np.pi / (self.environment_dim/2)]
            )
        self.gaps_r1[-1][-1] += 0.03
        
        self.gaps_r2 = [[-np.pi / 2 - 0.03, -np.pi / 2 + np.pi / (self.environment_dim/2)]]
        for m in range((int(self.environment_dim/2)) - 1):
            self.gaps_r2.append(
                [self.gaps_r2[m][1], self.gaps_r2[m][1] + np.pi / (self.environment_dim/2)]
            )
        self.gaps_r2[-1][-1] += 0.03

        port = "11311"
        subprocess.Popen(["roscore", "-p", port])

        print("Roscore launched!")

        # # Launch the simulation with the given launchfile name
        rospy.init_node("gym", anonymous=True)
        if launchfile.startswith("/"):
            fullpath = launchfile
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", launchfile)
        if not path.exists(fullpath):
            raise IOError("File " + fullpath + " does not exist")

        subprocess.Popen(["roslaunch", "-p", port, fullpath])
        print("Gazebo launched!")
               

        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)
        
        self.vel_pub_r1 = rospy.Publisher("/r1/cmd_vel", Twist, queue_size=1)
        self.set_state = rospy.Publisher("gazebo/set_model_state", ModelState, queue_size=10)
        self.publisher_goal = rospy.Publisher("goal_point", MarkerArray, queue_size=3)
        self.publisher2_r1 = rospy.Publisher("linear_velocity_r1", MarkerArray, queue_size=1)
        self.publisher3_r1 = rospy.Publisher("angular_velocity_r1", MarkerArray, queue_size=1)
        self.velodyne_1r1 = rospy.Subscriber("/r1/velodyne_points", PointCloud2, self.velodyne_callback_r1, queue_size=1)
        self.odom_r1 = rospy.Subscriber("/r1/odom", Odometry, self.odom_callback_r1, queue_size=1)
        
        # Set up the ROS publishers and subscribers (robot 2)

        self.vel_pub_r2 = rospy.Publisher("r2/cmd_vel", Twist, queue_size=1)
        self.publisher2_r2 = rospy.Publisher("linear_velocity_r2", MarkerArray, queue_size=1)
        self.publisher3_r2 = rospy.Publisher("angular_velocity_r2", MarkerArray, queue_size=1)
        self.velodyne_r2 = rospy.Subscriber("/r2/velodyne_points", PointCloud2, self.velodyne_callback_r2, queue_size=1)
        self.odom_r2 = rospy.Subscriber("/r2/odom", Odometry, self.odom_callback_r2, queue_size=1)


    # Read velodyne pointcloud and turn it into distance data, then select the minimum value for each angle
    # range as state representation
    def velodyne_callback_r1(self, v):
        data = list(pc2.read_points(v, skip_nans=False, field_names=("x", "y", "z")))
        self.velodyne_data_r1 = np.ones(int(self.environment_dim/2)) * 16
        for i in range(len(data)):
            if data[i][2] > -0.2:
                dot = data[i][0] * 1 + data[i][1] * 0
                mag1 = math.sqrt(math.pow(data[i][0], 2) + math.pow(data[i][1], 2))
                mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
                beta = math.acos(dot / (mag1 * mag2)) * np.sign(data[i][1])
                dist = math.sqrt(data[i][0] ** 2 + data[i][1] ** 2 + data[i][2] ** 2)

                for j in range(len(self.gaps_r1)):
                    if self.gaps_r1[j][0] <= beta < self.gaps_r1[j][1]:
                        self.velodyne_data_r1[j] = min(self.velodyne_data_r1[j], dist)

                        break
                    
    def velodyne_callback_r2(self, v):
        data = list(pc2.read_points(v, skip_nans=False, field_names=("x", "y", "z")))
        self.velodyne_data_r2 = np.ones(int(self.environment_dim/2)) * 16
        for i in range(len(data)):
            if data[i][2] > -0.2:
                dot = data[i][0] * 1 + data[i][1] * 0
                mag1 = math.sqrt(math.pow(data[i][0], 2) + math.pow(data[i][1], 2))
                mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
                beta = math.acos(dot / (mag1 * mag2)) * np.sign(data[i][1])
                dist = math.sqrt(data[i][0] ** 2 + data[i][1] ** 2 + data[i][2] ** 2)

                for j in range(len(self.gaps_r2)):
                    if self.gaps_r2[j][0] <= beta < self.gaps_r2[j][1]:
                        self.velodyne_data_r2[j] = min(self.velodyne_data_r2[j], dist)
                        break


    ## need to store the position of all the agents                     
    def odom_callback_r1(self, od_data):
        self.last_odom_r1 = od_data
        
    def odom_callback_r2(self, od_data):
        self.last_odom_r2 = od_data

    # Perform an action and read a new state
    def step(self, action):
        targets = False
        targets_goal_r1 = False
        targets_goal_r2 = False
         
        action_r1 = action[0:2]
        action_r2 = action[2:]
        # Publish the robot action
        ## need to publish actions for all the agents 
        vel_cmd_r1 = Twist()
        vel_cmd_r1.linear.x = action_r1[0]
        vel_cmd_r1.angular.z = action_r1[1]
        self.vel_pub_r1.publish(vel_cmd_r1)

        
        vel_cmd_r2 = Twist()
        vel_cmd_r2.linear.x = action_r2[0]
        vel_cmd_r2.angular.z = action_r2[1]
        self.vel_pub_r2.publish(vel_cmd_r2)
        # self.publish_markers(action)

        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        # propagate state for TIME_DELTA seconds
        time.sleep(TIME_DELTA)

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            pass
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")

        # read velodyne laser state
        ## need to call this function and check collision for each of the agents 
        collision_r1, min_laser_r1 = self.observe_collision(self.velodyne_data_r1)
        collision_r2, min_laser_r2 = self.observe_collision(self.velodyne_data_r2)
        targets_collisions = bool(collision_r1 + collision_r2)
        
        v_state_r1 = []
        v_state_r1[:] = self.velodyne_data_r1[:]
        laser_state_r1 = [v_state_r1]
        
        v_state_r2 = []
        v_state_r2[:] = self.velodyne_data_r2[:]
        laser_state_r2 = [v_state_r2]

        # Calculate robot heading from odometry data
        ## get position and orientation of all the agents 
        self.pre_odom_x_r1 = self.odom_x_r1 
        self.pre_odom_x_r2 = self.odom_x_r2 
        self.pre_odom_y_r1 = self.odom_y_r1 
        self.pre_odom_y_r2 = self.odom_y_r2 
        
        
        self.odom_x_r1 = self.last_odom_r1.pose.pose.position.x
        self.odom_y_r1 = self.last_odom_r1.pose.pose.position.y
        quaternion = Quaternion(
            self.last_odom_r1.pose.pose.orientation.w,
            self.last_odom_r1.pose.pose.orientation.x,
            self.last_odom_r1.pose.pose.orientation.y,
            self.last_odom_r1.pose.pose.orientation.z,)
        
        euler = quaternion.to_euler(degrees=False)
        angle_r1 = round(euler[2], 4)
        
        self.odom_x_r2 = self.last_odom_r2.pose.pose.position.x
        self.odom_y_r2 = self.last_odom_r2.pose.pose.position.y
        quaternion = Quaternion(
            self.last_odom_r2.pose.pose.orientation.w,
            self.last_odom_r2.pose.pose.orientation.x,
            self.last_odom_r2.pose.pose.orientation.y,
            self.last_odom_r2.pose.pose.orientation.z,)
        
        euler = quaternion.to_euler(degrees=False)
        angle_r2 = round(euler[2], 4)

         
        # Calculate distance to the goal from the robot
        ## calculate the distance of each agent to its corresponding goal position 
        distance_r1 = np.linalg.norm([self.odom_x_r1 - self.goal_x, self.odom_y_r1 - self.goal_y])
        distance_r2 = np.linalg.norm([self.odom_x_r2 - self.odom_x_r1, self.odom_y_r2 - self.odom_y_r1])
        
        distance_r1_pre = np.linalg.norm([self.pre_odom_x_r1 - self.goal_x, self.pre_odom_y_r1 - self.goal_y])
        distance_r2_pre = np.linalg.norm([self.pre_odom_x_r2 - self.pre_odom_x_r1, self.pre_odom_y_r2 - self.pre_odom_y_r1])

        # Calculate the relative angle between the robots heading and heading toward the goal
        ## robot 1
        skew_x_r1 = self.goal_x - self.odom_x_r1
        skew_y_r1 = self.goal_y- self.odom_y_r1
        dot_r1 = skew_x_r1 * 1 + skew_y_r1 * 0
        mag1_r1 = math.sqrt(math.pow(skew_x_r1, 2) + math.pow(skew_y_r1, 2))
        mag2_r1 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta_r1 = math.acos(dot_r1 / (mag1_r1 * mag2_r1))
        if skew_y_r1 < 0:
            if skew_x_r1 < 0:
                beta_r1 = -beta_r1
            else:
                beta_r1 = 0 - beta_r1
        theta_r1 = beta_r1 - angle_r1
        if theta_r1 > np.pi:
            theta_r1 = np.pi - theta_r1
            theta_r1 = -np.pi - theta_r1
        if theta_r1 < -np.pi:
            theta_r1 = -np.pi - theta_r1
            theta_r1 = np.pi - theta_r1
            
        # Robot 2 
        skew_x_r2 = self.odom_x_r1 - self.odom_x_r2
        skew_y_r2 = self.odom_y_r1 - self.odom_y_r2
        dot_r2 = skew_x_r2 * 1 + skew_y_r2 * 0
        mag1_r2 = math.sqrt(math.pow(skew_x_r2, 2) + math.pow(skew_y_r2, 2))
        mag2_r2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta_r2 = math.acos(dot_r2 / (mag1_r2 * mag2_r2))
        if skew_y_r2 < 0:
            if skew_x_r2 < 0:
                beta_r2 = -beta_r2
            else:
                beta_r2 = 0 - beta_r2
        theta_r2 = beta_r2 - angle_r2
        if theta_r2 > np.pi:
            theta_r2 = np.pi - theta_r2
            theta_r2 = -np.pi - theta_r2
        if theta_r2 < -np.pi:
            theta_r2 = -np.pi - theta_r2
            theta_r2 = np.pi - theta_r2

        # Detect if the goal has been reached and give a large positive reward            
        if distance_r1 < GOAL_REACHED_DIST:
            targets_goal_r1 = True
            self.i_decay = self.i_decay + 1
        target_goal = bool(targets_goal_r1 + targets_goal_r2)
        
        ## store the state vector for all the agents 
        robot_state_r1 = [distance_r1, theta_r1, action_r1[0], action_r1[1]]
        robot_state_r2 = [distance_r2, theta_r2, action_r2[0], action_r2[1]]
        state_r1 = np.append(laser_state_r1, robot_state_r1)
        state_r2 = np.append(laser_state_r2, robot_state_r2)
        reward_r1 = self.get_reward(targets_goal_r1, collision_r1, action_r1, min_laser_r1,distance_r1,distance_r1_pre, 0, angle_r1, angle_r2)
        reward_r2 = self.get_reward(targets_goal_r1, collision_r2, action_r2, min_laser_r2,distance_r2,distance_r2_pre, 1, angle_r1, angle_r2)
        
        state = np.append(state_r1,state_r2)
        reward = np.append(reward_r1,reward_r2)
        target = np.append(targets_goal_r1,targets_goal_r2)
        collisions = np.append(collision_r1, collision_r2)
        targets = bool(targets_collisions + target_goal)
        return state, reward, targets, target, collisions

    def reset(self):

        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            self.reset_proxy()

        except rospy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")

        
        ## reset the starting position and angle of both agents 
        angle_r1 = np.random.uniform(-np.pi, np.pi)
        quaternion_r1 = Quaternion.from_euler(0.0, 0.0, angle_r1)
        object_state_r1 = self.set_self_state_r1

        x_r1 = 5
        y_r1 = 0
        position_ok_r1 = False
        while not position_ok_r1:
            x_r1 = np.random.uniform(-6.5, 6.5)
            y_r1 = np.random.uniform(-6.5, 6.5)
            position_ok_r1 = check_pos(x_r1, y_r1)
        object_state_r1.pose.position.x = x_r1
        object_state_r1.pose.position.y = y_r1
        # object_state.pose.position.z = 0.
        object_state_r1.pose.orientation.x = quaternion_r1.x
        object_state_r1.pose.orientation.y = quaternion_r1.y
        object_state_r1.pose.orientation.z = quaternion_r1.z
        object_state_r1.pose.orientation.w = quaternion_r1.w
        self.set_state.publish(object_state_r1)

        self.odom_x_r1 = object_state_r1.pose.position.x
        self.odom_y_r1 = object_state_r1.pose.position.y
        
        angle_r2 = np.random.uniform(angle_r1-np.pi/2, angle_r1+np.pi/2)
        quaternion_r2 = Quaternion.from_euler(0.0, 0.0, angle_r2)
        object_state_r2 = self.set_self_state_r2

        #$
        x_r2 = -5
        y_r2 = -1
        position_ok_r2 = False
        while not position_ok_r2:
            r_rand = np.random.uniform(1.0, 3.0)
            phi_rand =  np.random.uniform(0,2*np.pi)
            x_r2 = x_r1 + r_rand * math.cos(phi_rand)
            y_r2 = y_r1 + r_rand * math.sin(phi_rand)
            position_ok_r2 = check_pos(x_r2, y_r2)
        object_state_r2.pose.position.x = x_r2
        object_state_r2.pose.position.y = y_r2
        # object_state.pose.position.z = 0.
        object_state_r2.pose.orientation.x = quaternion_r2.x
        object_state_r2.pose.orientation.y = quaternion_r2.y
        object_state_r2.pose.orientation.z = quaternion_r2.z
        object_state_r2.pose.orientation.w = quaternion_r2.w
        self.set_state.publish(object_state_r2)

        self.odom_x_r2 = object_state_r2.pose.position.x
        self.odom_y_r2 = object_state_r2.pose.position.y

        # set a random goal in empty space in environment
        ## generate a random goal for each of the agents 
        self.change_goal()
        self.goal_flag()
        # randomly scatter boxes in the environment
        self.random_box()
        # self.publish_markers([0.0, 0.0, 0.0, 0.0])

        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        time.sleep(TIME_DELTA)

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")

        ## store the laser data for both of the agents 
        v_state_r1 = []
        v_state_r1[:] = self.velodyne_data_r1[:]
        laser_state_r1 = [v_state_r1]
        
        v_state_r2 = []
        v_state_r2[:] = self.velodyne_data_r2[:]
        laser_state_r2 = [v_state_r2]

        ## calculate the distance to goal and heading angle for each agent 
        distance_r1 = np.linalg.norm([self.odom_x_r1 - self.goal_x, self.odom_y_r1 - self.goal_y])
        distance_r2 = np.linalg.norm([self.odom_x_r2 - self.odom_x_r1, self.odom_y_r2 - self.odom_y_r1])

        skew_x_r1 = self.goal_x - self.odom_x_r1
        skew_y_r1 = self.goal_y - self.odom_y_r1

        dot_r1 = skew_x_r1 * 1 + skew_y_r1 * 0
        mag1_r1 = math.sqrt(math.pow(skew_x_r1, 2) + math.pow(skew_y_r1, 2))
        mag2_r1 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta_r1 = math.acos(dot_r1 / (mag1_r1 * mag2_r1))

        if skew_y_r1 < 0:
            if skew_x_r1 < 0:
                beta_r1 = -beta_r1
            else:
                beta_r1 = 0 - beta_r1
        theta_r1 = beta_r1 - angle_r1

        if theta_r1 > np.pi:
            theta_r1 = np.pi - theta_r1
            theta_r1 = -np.pi - theta_r1
        if theta_r1 < -np.pi:
            theta_r1 = -np.pi - theta_r1
            theta_r1 = np.pi - theta_r1
            
        skew_x_r2 = self.odom_x_r1 - self.odom_x_r2
        skew_y_r2 = self.odom_y_r1 - self.odom_y_r2

        dot_r2 = skew_x_r2 * 1 + skew_y_r2 * 0
        mag1_r2 = math.sqrt(math.pow(skew_x_r2, 2) + math.pow(skew_y_r2, 2))
        mag2_r2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta_r2 = math.acos(dot_r2 / (mag1_r2 * mag2_r2))

        if skew_y_r2 < 0:
            if skew_x_r2 < 0:
                beta_r2 = -beta_r2
            else:
                beta_r2 = 0 - beta_r2
        theta_r2 = beta_r2 - angle_r2

        if theta_r2 > np.pi:
            theta_r2 = np.pi - theta_r2
            theta_r2 = -np.pi - theta_r2
        if theta_r2 < -np.pi:
            theta_r2 = -np.pi - theta_r2
            theta_r2 = np.pi - theta_r2
        ## return the states for each agent 
        robot_state_r1 = [distance_r1, theta_r1, 0.0, 0.0]
        robot_state_r2 = [distance_r2, theta_r2, 0.0, 0.0]
        
        state_r1 = np.append(laser_state_r1, robot_state_r1)
        state_r2 = np.append(laser_state_r2, robot_state_r2)
        state = np.append(state_r1,state_r2)

        return state

    #$

    def change_goal(self):
        # Place a new goal and check if its location is not on one of the obstacles
        # if self.upper < 15.8:
        #     self.upper += 0.005
        # if self.lower > -15.8:
        #     self.lower -= 0.005

        goal_ok_r1 = False
        while not goal_ok_r1: 
            self.goal_x = random.uniform(-6.5, 6.5) 
            self.goal_y = random.uniform(-6.5, 6.5) 
            goal_ok_r1 = check_pos(self.goal_x, self.goal_y)
            ## need to check if the box is far enough from both of the agents 
            distance_goal_to_robot_r1 = np.linalg.norm([self.goal_x - self.odom_x_r1, self.goal_y - self.odom_y_r1])
            distance_goal_to_robot_r2 = np.linalg.norm([self.goal_x - self.odom_x_r2, self.goal_y - self.odom_y_r2])
            if distance_goal_to_robot_r1 < 1.5 or distance_goal_to_robot_r2 < 1.5 :
                goal_ok_r1 = False


    def random_box(self):
        # Randomly change the location of the boxes in the environment on each reset to randomize the training
        # environment
        for i in range(5):
            name = "cardboard_box_" + str(i)
            x = 0
            y = 0
            box_ok = False
            while not box_ok:
                x = np.random.uniform(-6.5, 6.5)
                y = np.random.uniform(-6.5, 6.5)
                yaw_angle = np.random.uniform(0.0, 1.57)
                box_ok = check_pos(x, y)
                ## need to check if the box is far enough from both of the agents 
                distance_to_robot_r1 = np.linalg.norm([x - self.odom_x_r1, y - self.odom_y_r1])
                distance_to_robot_r2 = np.linalg.norm([x - self.odom_x_r2, y - self.odom_y_r2])
                distance_to_goal = np.linalg.norm([x - self.goal_x, y - self.goal_y])
                if distance_to_robot_r1 < 2 or distance_to_robot_r2 < 2 or distance_to_goal < 2:
                    box_ok = False
            box_state = ModelState()
            box_state.model_name = name
            box_state.pose.position.x = x
            box_state.pose.position.y = y
            box_state.pose.position.z = 0.0
            box_state.pose.orientation.x = 0.0
            box_state.pose.orientation.y = 0.0
            box_state.pose.orientation.z = 1.0
            box_state.pose.orientation.w = yaw_angle
            self.set_state.publish(box_state)
            
            
    def goal_flag(self):
        # Randomly change the location of the boxes in the environment on each reset to randomize the training
        # environment
        name = "goal_flag_box"
        x = self.goal_x
        y = self.goal_y

        goal_flag_state = ModelState()
        goal_flag_state.model_name = name
        goal_flag_state.pose.position.x = x
        goal_flag_state.pose.position.y = y
        goal_flag_state.pose.position.z = 0.0
        goal_flag_state.pose.orientation.x = 0.0
        goal_flag_state.pose.orientation.y = 0.0
        goal_flag_state.pose.orientation.z = 1.0
        goal_flag_state.pose.orientation.w = 1.0
        self.set_state.publish(goal_flag_state)


    @staticmethod
    def observe_collision(laser_data):
        # Detect a collision from laser data
        min_laser = min(laser_data)
        if min_laser < COLLISION_DIST:
            return True, min_laser
        return False, min_laser

    def get_reward(self, target_goal_r1, collision, action, min_laser, distance_r, distance_r_pre, r_ind, angle_r1, angle_r2):
        """Compute the reward for the selected robot."""
        if r_ind == 0:
            return leader_reward(
                reached_goal=bool(target_goal_r1),
                collision=bool(collision),
                action=action,
                min_obstacle_distance_m=float(min_laser),
                distance_to_goal_m=float(distance_r),
                previous_distance_to_goal_m=float(distance_r_pre),
            )

        robots_distance = float(np.linalg.norm([
            self.odom_x_r1 - self.odom_x_r2,
            self.odom_y_r1 - self.odom_y_r2,
        ]))
        heading_difference_deg = abs(float(angle_r1 - angle_r2)) * (180.0 / np.pi)
        return follower_reward(
            collision=bool(collision),
            action=action,
            min_obstacle_distance_m=float(min_laser),
            distance_to_leader_m=robots_distance,
            heading_difference_deg=heading_difference_deg,
        )
