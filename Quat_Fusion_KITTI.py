#LIDAR GPS IMU FUSION
#uses 3 vel, 3 accel, 3 angular accel measurements from IMU
#X,Y,Theta from LIDAR
#X,Y from GPS

import numpy as np
import pykitti
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import itertools
import toml
from rotations import Quaternion, skew_symmetric


#Read the TOML config file
params = toml.load("fusion_config.toml")
basedir = params.get("basedir")
date = params.get("date")
drive = params.get("drive")


# Initialize and import KITTI dataset
dataset = pykitti.raw(basedir, date, drive)
oxts = dataset.oxts
time_stamp = dataset.timestamps
N = len(oxts) - 5      #Number of samples from dataset
states = 9             #Number of states in the state matrix


# Extract pre-generated LIDAR ICP and GPS data
pose_lidar = np.array(np.loadtxt('LIDAR_' + date + '_' + drive + ".txt"))
pose_gps = np.array(np.loadtxt('GPS_' + date + '_' + drive + ".txt"))


#Load Process noise parameters from config file
sigma_ax2 = params.get("sigma_ax2")
sigma_ay2 = params.get("sigma_ay2")
#Measurement noise parameters: Lidar
sigma_x2_lid = params.get("sigma_x2_lid")
sigma_y2_lid = params.get("sigma_y2_lid")
sigma_yaw_lid = params.get("sigma_theta_lid")
sigma_lid = params.get("sigma_lid")

#Measurement noise parameters: GPS
sigma_x2_gps = params.get("sigma_x2_gps")
sigma_y2_gps = params.get("sigma_y2_gps")
sigma_z2_gps = params.get("sigma_z2_gps")

# variances for imu
var_imu_f = 0.005
var_imu_w = 0.008


#initialization of P matrix
P = np.zeros((states,states)) 
np.fill_diagonal(P, 1)

#GPS Observation and Noise covariance matrices
HGps = np.asarray([[1,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0]])
RGps = np.asarray([[sigma_x2_gps,0,0],[0,sigma_y2_gps,0],[0,0,sigma_z2_gps]]) 

#Lidar Observation and Noise covariance matrices
HLidar = np.asarray([[1,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0]])
RLidar = np.asarray([[sigma_lid,0,0],[0,sigma_lid,0],[0,0,sigma_lid]]) 


# State matrix
pos = np.zeros([N, 3])  # position estimates
vel = np.zeros([N, 3])  # velocity estimates
quat = np.zeros([N, 4])  # orientation estimates as quaternions
quat[0] = Quaternion(euler = ([oxts[0].packet.roll,oxts[0].packet.pitch,oxts[0].packet.yaw])).to_numpy() #initial quaternion


# LIDAR Measurements
data_lidar=[]

#GPS Measurements
data_gps=[]

# IMU measurements for process model
acc_imu = []
omega_imu = []
rpy_imu = []

g = np.array([0,0,-9.81])        # acc due to gravity

#Number of states
Lidarstates = 3                 #x,y,theta
Gpsstates = 2                   #x,y
C_ns = []                       #rotation matrix


#Initialize the states
p_check = pos[0]
v_check = vel[0]
q_check = quat[0]

#Process covariance
def Q(dt):
    
    vfa = var_imu_f**2
    vfw = var_imu_w**2
    return  dt**2 * np.diag([vfa,vfa,vfa,vfw,vfw,vfw])
       
#generate state transition matrix    
def Fmatrix(dt):                
    
    F_k = np.eye(9)
    F_k[0:3,3:6] = dt * np.eye(3)
    F_k[3:6,6:9] = -skew_symmetric(np.dot(C_ns[0][0],acc_imu[i]).reshape(3,1)).reshape(3,3) * dt
    return F_k


#generate control matrix
def Bmatrix(dt):

    B_k = np.zeros((9,6))
    B_k[3:6,0:3] = np.eye(3)
    B_k[6:9,3:6] = np.eye(3)
    return B_k


# EKF Prediction    
def predict_with_imu(P,p_check,v_check,q_check):
    
        F = Fmatrix(dt)
        B = Bmatrix(dt) 
        
        p_check = pos[i] + dt * vel[i] + 0.5 * dt**2 * (C_ns@acc_imu[i] + g)      # implement the motion model , 
                                                                                  # refer rotations.py for more details about quaternion multiplications
        v_check = vel[i] + dt * (C_ns@acc_imu[i] + g)
        q_check = Quaternion(axis_angle=omega_imu[i] * dt).quat_mult_right(quat[i])

        P =  F @ P @ F.T + B @ Q(dt) @ B.T
        return  p_check, v_check, q_check, P
    
def correct(P,Z,sensor_states,H,R,p_check,v_check,q_check):
    
        K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)      # K = P * H.T * inv(H*P*H.T + R)
        
        # 3.3 Correct predicted state
        delta_x = np.dot(K,(Z - pos[i]))      
        delta_p = delta_x[0:3]      # x,y,z
        delta_v = delta_x[3:6]      # vx,vy,vz
        delta_rpy = delta_x[6:9]    # Roll, Pitch, Yaw

        p_check = p_check + delta_p 
        v_check = v_check + delta_v
        q_check = Quaternion(axis_angle=delta_rpy).quat_mult_left(q_check,out = 'np')
        P = np.matmul(np.eye(states)-np.matmul(K,H),P)
        return  p_check, v_check, q_check, P

for i in range(0,N):
    
    dt = (time_stamp[i].microsecond - time_stamp[i-1].microsecond)/10000.0    #1000000.0
    
    #copy measurements from lidar and gps datasets
    data_lidar.append([pose_lidar[i][0],pose_lidar[i][1],oxts[i].packet.alt])
    data_gps.append([pose_gps[i][0],pose_gps[i][1],pose_gps[i][2]])
    
    # IMU data
    acc_imu.append(np.array([oxts[i].packet.ax,oxts[i].packet.ay,oxts[i].packet.az]))
    omega_imu.append(np.array([oxts[i].packet.wx,oxts[i].packet.wy,oxts[i].packet.wz]))
    rpy_imu.append(np.array([oxts[i].packet.roll,oxts[i].packet.pitch,oxts[i].packet.yaw]))
    
    C_ns = Quaternion(*quat[i]).to_mat()                                                                        # generate rotation matrix from quat
    
    p_check, v_check, q_check, P = predict_with_imu(P,p_check,v_check,q_check)                                  # Predict using IMU in motion model
    
    if(oxts[i].packet.numsats > 3):   #if number of satellites < 3,  reject GPS measurement
        p_check, v_check, q_check, P = correct(P,data_gps[i],Gpsstates,HGps,RGps,p_check,v_check,q_check)       # correct using GPS
    p_check, v_check, q_check, P = correct(P,data_lidar[i],Lidarstates,HLidar,RLidar,p_check,v_check,q_check)   # correct using LIDAR
    
    # update state matrix
    pos[i] = p_check
    vel[i] = v_check
    quat[i] = q_check
    

# 3D plot
est_traj_fig = plt.figure(figsize = (12,8))
ax = est_traj_fig.add_subplot(111, projection='3d')
ax.plot(pos[1:,0], pos[1:,1], label='EKF', color = 'r')
ax.plot(pose_lidar[:,0],pose_lidar[:,1],label='Measured by Lidar', color = 'g')
ax.plot(pose_gps[1:,0], pose_gps[1:,1], label='GPS',color = 'y')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')
ax.set_title('Final Estimated Trajectory')
ax.legend()
ax.set_zlim(-1, 5)
plt.show()


# 2D plot
plt.figure(figsize = (12,8))
plt.plot(pos[1:,0],pos[1:,1],label = 'EKF',color = 'r',ls = '--')
plt.plot(pose_lidar[:,0],pose_lidar[:,1],label='Measured by Lidar', color = 'g')
plt.plot(pose_gps[:,0],pose_gps[:,1],label='Measured by GPS',color='b')
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.show()