#GPS
import math
oxts = dataset.oxts

a = 6378137.0
b = 6356752.3142
e = math.sqrt(1-(b**2/a**2))

def gps_raw_to_xyz(lat,long,alt):
    #N = a/math.sqrt(1-e**2 * np.sin(lat))
    
    return scale*a*(np.pi*long/180) , scale*a*np.log(np.tan(np.pi*(90+lat)/360)) , alt #(N*(1-e**2) + alt)*np.sin(lat)

x_gps = []
y_gps = []
z_gps = []
p = []

lat_0 =  oxts[0][0][0]
long_0 = oxts[0][0][1]
alt_0 = oxts[0][0][2]
    
phi = oxts[0][0][3]
theta = oxts[0][0][4]
psi = oxts[0][0][5]
    
scale = np.cos(lat_0*np.pi/180)
x_0 =  s*a*(np.pi*long_0/180)
y_0 = s*a*np.log(np.tan(np.pi*(90+lat_0)/360))
z_0 = alt_0
#z_0 = (N*(1-e**2) + alt_0)*np.sin(lat_0)   

Tr_0_inv = np.linalg.inv(np.asarray([[np.cos(theta)*np.cos(psi),np.cos(psi)*np.sin(phi)*np.sin(theta) - np.cos(phi)*np.sin(psi),np.cos(phi)*np.cos(psi)*np.sin(theta) + np.sin(theta)*np.sin(psi),x_0],
                   [np.cos(theta)*np.sin(psi),np.cos(psi)*np.cos(phi) + np.sin(phi)*np.sin(psi)*np.sin(theta),-np.sin(phi)*np.cos(psi) + np.sin(theta)*np.sin(psi)*np.cos(phi),y_0],
                   [-np.sin(theta),np.cos(theta)*np.sin(phi),np.cos(phi)*np.sin(theta),z_0],
                   [0,0,0,1]]))
    
for i in range(0,len(oxts)):  
    x,y,z = gps_raw_to_xyz(oxts[i][0][0],oxts[i][0][1],oxts[i][0][2])
    
    p_g = np.asarray([[x],[y],[z],[1]])
    p = np.dot(Tr_0_inv, p_g)
    x_gps.append((p[0]))
    y_gps.append((p[1]))
    #z_gps.append(p[2])

    
plt.figure(figsize = (12,8))
plt.plot(x_gps,y_gps,color = 'r')
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.show()
