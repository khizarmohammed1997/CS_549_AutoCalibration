import numpy as np
import glob
from utils import*
import sys
import os
import cv2
import copy
import math
def main():
    # Calculating the homographies of all the given images
    homographies_list,corners,V_list=homographies(os.path.join(os.getcwd(),"Calibration_Imgs"))
    A=intrinsinc_matrix(V_list)
    print(A)

    

def intrinsinc_matrix(V_list):
    ur,sigma,vr=np.linalg.svd(V_list)
    b=vr[:,-1]
    w = b[0]*b[2]*b[5] - (b[1]**2)*b[5] - b[0]*(b[4]**2) + 2*b[1]*b[3]*b[4] - b[2]*(b[3]**2)
    d = b[0]*b[2] - b[1]**2
    alpha = math.sqrt(w/(d*b[0]))
    beta = math.sqrt((w/(d**2))*b[0])
    gamma = math.sqrt(w/((d**2)*b[0])) * b[1] * -1
    u0 = (b[1]*b[4] - b[2]*b[3]) / d
    v0 = (b[1]*b[3] - b[0]*b[4]) / d
    return np.array([[alpha, gamma, u0],[0, beta, v0],[0, 0, 1]])

def display(image,name="test"):
    cv2.imshow("name",image)
    cv2.waitKey(0)

def v(H, i, j):
	return np.array([H[0, i]*H[0, j],H[0, i]*H[1, j]+H[1, i]*H[0, j],H[1, i]*H[1, j],H[2, i]*H[0, j]+H[0, i]*H[2, j],H[2, i]*H[1, j]+H[1, i] * H[2, j],H[2, i] * H[2, j]]).reshape(1,6)

def world_coordinates(max_x,max_y):
	box_size = 21.5
	corner_locations= []
	for i in range(1,max_y+1):
		for j in range(1,max_x+1):
			corner_locations.append([box_size*j,box_size*i,0,1])
	
	return np.array(World_xy)



def homographies(path=None):
    
    world_corner_locations= np.array([[21.5, 21.5],[21.5*9,21.5],[21.5*9, 21.5*6],[21.5,21.5*6]], dtype='float32')
    dir_list = os.listdir(path)
    corners_list=[]
    Homographies = np.zeros((1,3,3))
    V_list=np.zeros([1,6])

    # Initializaing the corners list
    for an_image in dir_list:
        img=cv2.imread(os.path.join(os.getcwd(),"Calibration_Imgs",an_image))
        img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        val, corners = cv2.findChessboardCorners(img,(9,6),None)
        corners_list.append(corners)
        
        # Visualizing corners
        img_ = cv2.drawChessboardCorners(copy.deepcopy(img),(9,6),corners,val)
        
        #comparing corners with the selected world corners
        camera_corner_locations=np.array([[corners[0][0]],[corners[8][0]],[corners[53][0]],[corners[45][0]]],dtype='float32')
        H,_ = cv2.findHomography(world_corner_locations,camera_corner_locations)
        H=np.array(H)
        Homographies = np.concatenate((Homographies,H.reshape(1,3,3)),axis=0)


        #calculating V for Vb=0
        v01=v(H,0,1)
        v00=v(H,0,1)
        v11=v(H,1,1)
        V_list=np.concatenate([V_list,v01,(v00-v11)],axis=0)
    return Homographies[1:,:,:], np.array(corners_list),V_list[1:,:]

if __name__=='__main__':
    main()