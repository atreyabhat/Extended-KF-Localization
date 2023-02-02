#LIDAR ENCODER IMU FUSION
#uses 3 vel, 3 accel, 3 angular accel measurements from IMU
#X,Y,Theta from LIDAR
# V and W from encoder

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
states = 9

#path to dataset
path = "./data/iisc/IISC"

# Extract pre-generated LIDAR ICP data
pose_lidar = np.array(np.loadtxt('LIDAR_IISC_point.txt'))

timestamps_lid = np.load(path + "/lidar/frame_time.npy")            # Load timestamps
imu = pd.read_csv(path + '/imu.csv',delimiter=',')                  # load IMU data
wheel = pd.read_csv( path + '/wheel_control.csv',delimiter=',')     # Load wheel encoder data

imu_wheel = pd.merge_asof(left=imu , right=wheel, left_on='time', right_on='time')                # time sync the data from different sensors

pose_imu_enc = np.array(pd.merge_asof(left=pd.DataFrame(timestamps_lid, columns = ['time']) , right=imu_wheel, left_on='time', right_on='time'))

N = len(pose_imu_enc)                                               # Number of samples


#Load Process noise parameters from config file
sigma_ax2 = params.get("sigma_ax2")
sigma_ay2 = params.get("sigma_ay2")
#Measurement noise parameters: Lidar
sigma_x2_lid = params.get("sigma_x2_lid")
sigma_y2_lid = params.get("sigma_y2_lid")
sigma_yaw_lid = params.get("sigma_theta_lid")
sigma_lid = params.get("sigma_lid")

var_imu_f = params.get('var_imu_f')
var_imu_w = params.get('var_imu_w')


#initialization of P matrix
P = np.zeros((states,states)) 
np.fill_diagonal(P, 1)


#Lidar  Observation and Noise covariance matrices
HLidar = np.asarray([[1,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0]])
RLidar = np.asarray([[sigma_lid,0,0],[0,sigma_lid,0],[0,0,sigma_lid]]) 



# State matrix
pos = np.zeros([N, 3])  # position estimates
vel = np.zeros([N, 3])  # velocity estimates
quat = np.zeros([N, 4])  # orientation estimates as quaternions
quat[0] = Quaternion(euler = (pose_imu_enc[0][7:10])).to_numpy()

g = np.array([0,0,-9.81])   #acc due to gravity

#Number of states
Lidarstates = 3             #x,y,theta
C_ns = []                   # Rotation Matrix 

#init states
p_check = pos[0]
v_check = vel[0]
q_check = quat[0]

#Process covariance
def Q(dt):
    
    vfa = var_imu_f**2
    vfw = var_imu_w**2
    return  dt**2 * np.diag([vfa,vfa,vfa,vfw,vfw,vfw])
       

# state transition matrix
def Fmatrix(dt,sensor):                
    
    if(sensor == 'imu'):        # F matrix generator if IMU is chosen in motion model
        F_k = np.eye(9)
        F_k[0:3,3:6] = dt * np.eye(3)
        F_k[3:6,6:9] = -skew_symmetric(np.dot(C_ns,pose_imu_enc[i][1:4]).reshape(3,1)).reshape(3,3) * dt
        return F_k
    
    if(sensor == 'enc'):       # F matrix generator if wheel encoder is chosen in motion model
        F_k = np.eye(9)
        F_k[0,3] = dt
        F_k[1:3,4:6] = dt * np.eye(2)
        F_k[3:6,6:9] = -skew_symmetric(np.dot(C_ns,pose_imu_enc[i][1:4]).reshape(3,1)).reshape(3,3) * dt
        return F_k        

# control matrix
def Bmatrix(dt):

    B_k = np.zeros((9,6))
    B_k[3:6,0:3] = np.eye(3)
    B_k[6:9,3:6] = np.eye(3)
    return B_k    
    
def predict_with_imu(P,p_check,v_check,q_check):
    
    F = Fmatrix(dt, sensor = 'imu')
    B = Bmatrix(dt) 
        
    # implement the motion model
    p_check = pos[i] + dt * vel[i] + 0.5 * dt**2 * (C_ns @ pose_imu_enc[i][1:4] + g)
    v_check = vel[i] + dt * (C_ns @ pose_imu_enc[i][1:4] + g)
    q_check = Quaternion(axis_angle = pose_imu_enc[i][1:4] * dt).quat_mult_right(quat[i])

    P =  F @ P @ F.T + B @ Q(dt) @ B.T
    return  p_check, v_check, q_check, P
    
    
def predict_with_enc(P,p_check,v_check,q_check):
    
    F = Fmatrix(dt,sensor = 'enc')
    B = Bmatrix(dt)

    # load from dataset
    vel_enc = np.array([pose_imu_enc[i][11],0,0]).T
    omega_enc = np.array([0,0,pose_imu_enc[i][12]]).T
        
    # implement the motion model
    p_check = pos[i] + dt * vel_enc #+ 0.5 * dt**2 * (C_ns @ pose_imu_enc[i][1:4] + g)
    v_check = vel_enc #+ dt * (C_ns @ pose_imu_enc[i][1:4] + g)
    q_check = Quaternion(axis_angle = omega_enc * dt).quat_mult_left(quat[i])

    P =  F @ P @ F.T + B @ Q(dt) @ B.T
    return  p_check, v_check, q_check, P
    
    
def correct(P,Z,sensor_states,H,R,p_check,v_check,q_check):
    
    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
        
    #Correct predicted state
    delta_x = np.dot(K,(Z - pos[i]))      
    delta_p = delta_x[0:3]      # x,y,z
    delta_v = delta_x[3:6]      # vx,vy,vz
    delta_rpy = delta_x[6:9]    # Roll, Pitch, Yaw

    p_check = p_check + delta_p
    v_check = v_check + delta_v
    q_check = Quaternion(axis_angle=delta_rpy).quat_mult_left(q_check,out = 'np')
    P = np.matmul(np.eye(states)-np.matmul(K,H),P)
    return  p_check, v_check, q_check, P


# MAIN ##################################################################################

position = []
velocity = []
quaternion = []

for i in range(1,N-1):
    
    dt = (pose_imu_enc[i+1,0] - pose_imu_enc[i,0])
    C_ns = Quaternion(*quat[i]).to_mat()
    
    p_check, v_check, q_check, P = predict_with_enc(P,p_check,v_check,q_check)   #EKF predict with enc and imu as control inputs
    p_check, v_check, q_check, P = predict_with_imu(P,p_check,v_check,q_check)
    
    if(i%5==0):                                                                  # predict for every 5 samples

        p_check, v_check, q_check, P = correct(P,pose_lidar[i],Lidarstates,HLidar,RLidar,p_check,v_check,q_check)  # EKF predict with lidar data

        #update states
        pos[i] = p_check
        vel[i] = v_check
        quat[i] = q_check
        position.append(p_check)
        velocity.append(v_check)
        quaternion.append(q_check)
    
position = np.array(position)

# 2D plot
plt.figure(figsize = (12,8))
plt.plot(position[1:,0],position[1:,1],label = 'EKF',color = 'r',linewidth=3)
plt.plot(pose_lidar[:,0],pose_lidar[:,1],label='Measured by Lidar', color = 'b')
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.show()