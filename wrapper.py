import numpy as np
import glob
from utils import*
import sys
import os
import cv2
import copy
import math
from scipy.optimize import least_squares

def lamda_cal(inverse_K,Homography):
	lamda=1/np.linalg.norm(np.matmul(inverse_K,Homography[:,0]))
	return lamda

def extrinsic(intrinsinc_matrix,homography_list):
	extrinsinc_matrix_list = []
	inverse_K = np.linalg.inv(intrinsinc_matrix)
	n=homography_list.shape[0]

	for i in range(n):
		a_homography=homography_list[i]
		lamda= lamda_cal(inverse_K,a_homography)
		r1 = lamda*np.dot(inverse_K,a_homography[:,0])
		r2 = lamda*np.dot(inverse_K,a_homography[:,1])
		t = lamda*np.dot(inverse_K,a_homography[:,2])
		r3 = np.cross(r1,r2)
		R = np.asarray([r1, r2, r3]).T


		extrinsic=np.hstack((R,t.reshape(3,1)))
		extrinsinc_matrix_list.append(extrinsic)

	return extrinsinc_matrix_list

def optimize(parameters,corner_points,extrinsinc):
	World_locations = world_coordinates(9,6)
	updated_params = least_squares(fun=optimization_function,x0 = parameters,method="lm",args=[corner_points,extrinsinc,World_locations])
	[alpha, beta, u0, v0 , gamma, k1, k2] = updated_params.x
	A_optimized=np.array([[alpha, gamma, u0],[0, beta, v0],[0, 0, 1]])
	return  A_optimized,k1, k2
	

def intrinsinc_matrix(V_list):
	ur,sigma,vr=np.linalg.svd(V_list)
	b=vr[:][-1]
	w = b[0]*b[2]*b[5] - (b[1]**2)*b[5] - b[0]*(b[4]**2) + 2*b[1]*b[3]*b[4] - b[2]*(b[3]**2)
	d = b[0]*b[2] - b[1]**2
	# print(w/(d*b[0]))
	alpha = math.sqrt(w/(d*b[0]))
	beta = math.sqrt((w/(d**2))*b[0])
	gamma = math.sqrt(w/((d**2)*b[0])) * b[1] * -1
	u0 = (b[1]*b[4] - b[2]*b[3]) / d
	v0 = (b[1]*b[3] - b[0]*b[4]) / d
	return np.array([[alpha, gamma, u0],[0, beta, v0],[0, 0, 1]])

def display(image,name="test"):
	image=cv2.resize(image,(960,540))
	cv2.imshow("name",image)
	cv2.waitKey(0)

def v(H, i, j):
	return np.array([H[0, i]*H[0, j],H[0, i]*H[1, j]+H[1, i]*H[0, j],H[1, i]*H[1, j],H[2, i]*H[0, j]+H[0, i]*H[2, j],H[2, i]*H[1, j]+H[1, i] * H[2, j],H[2, i] * H[2, j]])

def world_coordinates(max_x,max_y):
	sqaure_side = 21.5
	world_corner_locations= []
	for i in range(1,max_y+1):
		for j in range(1,max_x+1):
			world_corner_locations.append([sqaure_side*j,sqaure_side*i,0,1])
	return np.array(world_corner_locations)
def optimization_function(parameters,corners,extrinsic_matrix,World_locations,reprojection=False):
	#initilaize the distortion parameters
	k1 = parameters[5]
	k2 = parameters[6]

	#initilize the principal points
	u0 = parameters[2]
	v0 = parameters[3]
	
	#Initialize the A matrix
	A = np.zeros([3,3])
	A[0,0] = parameters[0]
	A[1,1] = parameters[1]
	A[0,2] = parameters[2]
	A[1,2] = parameters[3]
	A[0,1] = parameters[4]
	A[2,2] = 1

	#initializing error
	e = []
	predicted_corners=[]
	for image_corners,an_extrinsinc in zip(corners,extrinsic_matrix):
		predicted_corners_image=[]
		for a_camera_corner,a_world_coordinate in zip(image_corners,World_locations):
	
			#transforming to camera coordinates
			camera_coordinates = np.matmul(an_extrinsinc,a_world_coordinate)
			camera_coordinates=camera_coordinates/camera_coordinates[-1]

			#transforming to image coordinates
			H = np.matmul(A,camera_coordinates)
			H = H/H[-1]

			#getting coordinatge poistions
			xc,yc = camera_coordinates[0],camera_coordinates[1]
			u,v = H[0],H[1]
			
			#caclcualting uhat, vhat
			r = xc**2 + yc**2
			u_hat = u + (u-u0)*(k1*r + k2*(r**2))
			v_hat = v + (v-v0)*(k1*r + k2*(r**2))
			
			#saving error
			if reprojection==True:
				current_error=math.sqrt((a_camera_corner[0]-u_hat)**2+(a_camera_corner[1]-v_hat)**2)
				e.append(current_error)
				predicted_corners_image.append([u_hat,v_hat])

			else:
				e.append(a_camera_corner[0] - u_hat)
				e.append(a_camera_corner[1] - v_hat)
		if reprojection:		
			predicted_corners.append(predicted_corners_image)
	
	if reprojection:
		return np.float64(e).flatten(),predicted_corners
	else:
		return np.float64(e).flatten()

def visualization(corners,predicted_corners,path):
	dir_list = os.listdir(path)
	for i in range(len(dir_list)):
		an_image=dir_list[i]
		img=cv2.imread(os.path.join(os.getcwd(),"Calibration_Imgs",an_image))
		for a_corner,a_predicted_corner in zip(corners[i],predicted_corners[i]):
			j=1
			cv2.circle(img,(int(a_corner[0]),int(a_corner[1])),5,(0,0,255),15)
			cv2.circle(img,(int(a_predicted_corner[0]),int(a_predicted_corner[1])),5,(255,0,0),10)
		display(img)

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
		corners=corners.reshape(-1,2)
		corners_list.append(corners)
		
		#comparing corners with the selected world corners
		camera_corner_locations=np.concatenate((corners[0],corners[8],corners[53],corners[45]),dtype='float32')
		H,_ = cv2.findHomography(world_corner_locations,camera_corner_locations.reshape(-1,2))
		H=np.array(H)
		Homographies = np.concatenate((Homographies,H.reshape(1,3,3)),axis=0)

		#calculating V for Vb=0
		v01=v(H,0,1)
		v00=v(H,0,0)
		v11=v(H,1,1)
		v_current=np.concatenate((v01,v00-v11)).reshape(2,6)
		V_list=np.concatenate((V_list,v_current),axis=0)
	return Homographies[1:,:,:], np.array(corners_list),V_list[1:,:]
def main():
	
	#Calculating the intrinsinc and extrinsinc
	homographies_list,corners,V_list=homographies(os.path.join(os.getcwd(),"Calibration_Imgs"))
	A=intrinsinc_matrix(V_list)
	E=extrinsic(A,homographies_list)

	#optimization
	parameters=np.array([A[0,0],A[1,1],A[0,2],A[1,2],A[0,1],0,0])
	A_optimimzed,k1,k2 = optimize(parameters,corners,E)
	
	#calculating reprojection error with optimized A and distortion coefficents
	World_locations = world_coordinates(9,6)
	parameters=np.array([A_optimimzed[0,0],A_optimimzed[1,1],A_optimimzed[0,2],A_optimimzed[1,2],A_optimimzed[0,1],k1,k2])
	error,predicted_corners=optimization_function(parameters,corners,E,World_locations,reprojection=True)
	print(f"A={A}")
	print(f"A_optimized={A_optimimzed}")
	print(f"k1={k1} k2={k2}")
	print(f"Mean errro after optimization = {np.mean(error)}")
	visualization(corners,predicted_corners,os.path.join(os.getcwd(),"Calibration_Imgs"))

	#calculating reprojection error with unoptimized A
	World_locations = world_coordinates(9,6)
	parameters=np.array([A[0,0],A[1,1],A[0,2],A[1,2],A[0,1],0,0])
	error_old,predicted_corners_old=optimization_function(parameters,corners,E,World_locations,reprojection=True)
	print(f"Mean errro before optimization = {np.mean(error_old)}")
	visualization(corners,predicted_corners_old,os.path.join(os.getcwd(),"Calibration_Imgs"))
	
if __name__=='__main__':
	main()