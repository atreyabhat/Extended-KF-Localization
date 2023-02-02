#LIDAR GPS IMU FUSION
#uses ONLY [Vel_X,Omega_Z] measurements from IMU
#X,Y, Theta from LIDAR
#X,Y from GPS

import numpy as np
import pykitti
import matplotlib.pyplot as plt
import math
import pandas as pd
import itertools
import toml


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
states = 6             #Number of states in the state matrix


# Extract pre-generated LIDAR ICP and GPS data
pose_lidar = np.array(np.loadtxt('LIDAR_' + date + '_' + drive + ".txt")) 
pose_gps = np.array(np.loadtxt('GPS_' + date + '_' + drive + ".txt"))


#Load Process noise parameters from config file
sigma_ax2 = params.get("sigma_ax2")
sigma_ay2 = params.get("sigma_ay2")
#Measurement noise parameters: Lidar
sigma_x2_lid = params.get("sigma_x2_lid")
sigma_y2_lid = params.get("sigma_y2_lid")
sigma_theta_lid = params.get("sigma_theta_lid")
#Measurement noise parameters: GPS
sigma_x2_gps = params.get("sigma_x2_gps")
sigma_y2_gps = params.get("sigma_y2_gps")
#Measurement noise params : IMU
#sigma_theta_imu = params.get("sigma_theta_imu")
sigma_omega_imu = params.get("sigma_omega_imu")
sigma_velx_imu = params.get("sigma_vel_imu")


#initialization of P matrix
P = np.zeros((states,states)) 
np.fill_diagonal(P, 10)

X = np.asarray([[1],[1],[0.0],[1],[1],[1]])

#GPS Observation and Noise covariance matrices
HGps = np.asarray([[1,0,0,0,0,0],[0,1,0,0,0,0]])
RGps = np.asarray([[sigma_x2_gps,0],[0,sigma_y2_gps]]) 

#Lidar Observation and Noise covariance matrices
HLidar = np.asarray([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0]])
RLidar = np.asarray([[sigma_x2_lid,0,0],[0,sigma_y2_lid,0],[0,0,sigma_theta_lid]]) 


# STATE MATRIX
xpos=[]
ypos=[]
theta = []
vx=[]
vy = []
omega = []


# LIDAR Measurements
xpos_lidar=[]
ypos_lidar = []
theta_lidar = []


#GPS Measurements
xpos_gps=[]
ypos_gps = []


# IMU measurements for process model
theta_imu = []
omega_imu = []
velx_imu = []

#Number of states of the sensors
Lidarstates = 3 #x,y,theta
Gpsstates = 2   #x,y


#Process covariance
def Q(state):
    return  np.diag([0.1,0.15,0.20,0.25,0.20,0.15])


# Function to generate state transition matrix       
def Fmatrix(state):
    x,y,theta,vx,vy,w = tuple(np.concatenate(state))
    return np.asarray([[1,0,0,dt,0,0],[0,1,0,0,dt,0],[0,0,1,0,0,dt],
                       [0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])

#Function to generate control matrix
def Bmatrix(dt):
    return np.asarray([[dt,0],[0,0],[0,dt],[1,0],[0,0],[0,1]])


#Prediction step of EKF    
def predict_with_imu(X,P,Z):
        F = Fmatrix(X)                                              #load F matrix
        B = Bmatrix(dt)                                             #Load control matrix
        u = np.asarray([velx_imu[i],omega_imu[i]])                  #generate control input matrix 
        X = np.matmul(F,X) + np.matmul(B,u).reshape(6,1)  
        P = np.add(np.matmul(np.asarray(np.matmul(F,P)),np.transpose(F)),Q(dt))
        return X,P
    
# Correction step of EKF    
def correct(X,P,Z, sensor_states, H_sensor,R_sensor):
        
        S = H_sensor @ P @ H_sensor.T + R_sensor                                         #S = H * P * H.T + R
        K = P @ H_sensor.T @ np.linalg.inv(S)                                            #K = p * H.T * inv(S)
        y = np.reshape(np.asarray(Z[0:sensor_states]),(sensor_states,1))- H_sensor @ X   # y = Z - H*X
        X = X + K @ y                                                                    # X = X + K*y               
        P = np.matmul(np.identity(len(X)) - np.matmul(K,H_sensor),P)                     #P = P*(I - K*H)
        return X,P

for i in range(0,N):
    
    dt = (time_stamp[i].microsecond - time_stamp[i-1].microsecond)/10000.0
    
    #copy measurements from dataset
    xpos_lidar.append(pose_lidar[i][0])                             #extract the lidar data to np arrays
    ypos_lidar.append(pose_lidar[i][1])
    theta_lidar.append(pose_lidar[i][2])
    xpos_gps.append(pose_gps[i][0])                                 # extract GPS data to np arrays
    ypos_gps.append(pose_gps[i][1])
    omega_imu.append(oxts[i].packet.wz)                             # extract IMU data
    velx_imu.append(oxts[i].packet.vf)
    
    X,P = predict_with_imu(X,P,np.array([[velx_imu,omega_imu]]))    # Prediction step
    if(oxts[i].packet.numsats > 3):                                 #if number of satellites < 3, reject GPS measurement
        X,P = correct(X,P,pose_gps[i],Gpsstates,HGps,RGps)          # correct with GPS data
    X,P = correct(X,P,pose_lidar[i],Lidarstates,HLidar,RLidar)      # correct with lidar data
    
    xpos.append(X[0])                                               #update the state matrix with the EKF outputs
    ypos.append(X[1])
    theta.append(X[2])
    vx.append(X[3])
    vy.append(X[4])
    omega.append(X[5])


#plot 

plt.figure(figsize=(12,8))
plt.plot(xpos_lidar,ypos_lidar,label='Measured by Lidar', color = 'g')
plt.plot(xpos_gps,ypos_gps,label='Measured by GPS',color='b')
plt.plot(xpos,ypos,label='EKF - Lidar+GPS+IMU',color='r',ls='--')
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.legend()
plt.show()