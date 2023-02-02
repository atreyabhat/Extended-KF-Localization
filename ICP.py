#ICP on LIDAR kitti datatset in 2D
LIDAR_PTS = 110000
DIMS = 2

def to_vector(pose_hc):
    return [pose_hc[0][2], pose_hc[1][2], np.arccos(pose_hc[0][0])]

def to_hc(pose):
    x, y, theta = pose
    return np.array([[np.cos(theta), -np.sin(theta), x],[np.sin(theta), np.cos(theta), y],[0, 0, 1]])

N = len(time_stamp)
pose = to_hc([0, 0, np.pi/2])
all_poses = [[0, 0, np.pi/2]]
prev_frame =  dataset.get_velo(0)[:LIDAR_PTS,:DIMS]
for i in range(N):
    curr_frame = dataset.get_velo(i+1)[:LIDAR_PTS,:DIMS]
    if(i%5==0):
        print("ICP on " + str(i) + "th frame, out of " + str(N) + "frames")
    T, _, _ = icp(curr_frame, prev_frame, tolerance=0.01)
    pose = T @ pose
    all_poses.append(to_vector(pose))
    prev_frame = curr_frame

all_poses = np.array(all_poses)
plt.figure()
plt.plot(all_poses[:,0], all_poses[:,1])
plt.show()
