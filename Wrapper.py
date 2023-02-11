import cv2

import os
import numpy as np
import copy
from statistics import mean
import scipy.optimize as optimize

def readImages(ImagePath):

    
    if os.path.exists(ImagePath): 
        image_list = os.listdir(ImagePath)
        image_list.sort()
    else:
        raise Exception ("Directory doesn't exist")
    images_path = []
    for i in range(len(image_list)):
        image_path = os.path.join(ImagePath,image_list[i])
        images_path.append(image_path)

    images = [cv2.imread(i) for i in images_path]
    
    return images
       

    

def imagePoint(img,index , X_limit, Y_limit, world_corners): #hi

    
    img_gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(img_gray , (X_limit,Y_limit), None)
    corners = corners.reshape(corners.shape[0],corners.shape[2])    
    corners2 = cv2.cornerSubPix(img_gray,corners, (11,11), (-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))   
    H,_ = cv2.findHomography(world_corners,corners,cv2.RANSAC,5.0)    
    if(ret):
        cv2.drawChessboardCorners(img, (X_limit,Y_limit), corners2, ret)
        cv2.imwrite(os.path.join( 'Misc/Results/','Corners_'+str(index+1)+'.jpg'), img)
        # cv2.imshow('img', img)                
        # cv2.waitKey(0)

    return corners, H
    


def imagePointH(images, X_limit, Y_limit, world_corners):
    corners_images =[]   
    homography      = []
    for i,img in enumerate(images):
            corners, H = imagePoint(img,i, X_limit, Y_limit, world_corners)
            homography.append(H)
            corners_images.append(corners)
    return corners_images, homography
     


def worldPoints(X_limit, Y_limit, CheckerBoard_Width): #Hi

   
    world_corners = np.zeros((X_limit * Y_limit, 3), np.float32)
    world_corners[:,:2] = np.mgrid[0:X_limit,0:Y_limit].T.reshape(-1,2)

    world_corners = world_corners * CheckerBoard_Width

    return world_corners




def computerVij(H,i,j):

    Vij = np.array([ H[0,i]*H[0,j], 
                       H[0,i]*H[1,j] + H[1,i]*H[0,j],
                       H[1,i]*H[1,j],
                       H[2,i]*H[0,j] + H[0,i]*H[2,j],
                       H[2,i]*H[1,j] + H[1,i]*H[2,j],
                       H[2,i]*H[2,j]])
    
    return Vij

def computeIntrinsics(allH):
    V = np.zeros(6).reshape(1,6)
    for H in allH:
        # H= H.T
        # print("H . shape",H.shape)
        # V.append(computerVij(H,0,1))
        # V.append(computerVij(H,0,0)-computerVij(H,1,1))
        v12     = computerVij(H,0,1).reshape(1,6)
        vdif    = computerVij(H,0,0).reshape(1,6)-computerVij(H,1,1).reshape(1,6)
        v_vec   = np.vstack((v12,vdif))
        V       = np.vstack((V,v_vec))
    V = V[1:]
    # V = np.array(V)
    # V= V.T
    print(V.shape)
    #Vb = 0

    #In order to compute b from B, we do SVD of V and obtain 
    U,S,VT = np.linalg.svd(V)
    b=VT[np.argmin(S)]
    B11 = b[0]
    B12 = b[1]
    B22 = b[2]
    B13 = b[3]
    B23 = b[4]
    B33 = b[5]
    # print("B11 :",B11)
    v0      = (B12*B13 - B11*B23)/(B11*B22 - B12**2)
    lamda   = B33 - (B13**2 + v0*(B12*B13 - B11*B23))/B11
    alpha   = np.sqrt(lamda/B11)
    beta    = np.sqrt(lamda*B11 /(B11*B22 - B12**2))
    gamma   = -B12*(alpha**2)*beta/lamda
    u0      = gamma*v0/beta -B13*(alpha**2)/lamda

    K = np.array([[alpha, gamma, u0],
                    [    0,  beta, v0],
                    [    0,     0,  1]],dtype = float)
    print("K:",K)            
    return K


def computeEntrinsics(K, H):
    R = []
    for h in H:
        h1 = h[:,0]
        h2 = h[:,1]
        h3 = h[:,2]
        lmbda = mean([1/np.linalg.norm(np.matmul(np.linalg.inv(K), h1),2),1/np.linalg.norm(np.matmul(np.linalg.inv(K), h2),2)])
        # print("Lambda : ",lmbda)
        r1 = np.matmul(lmbda * np.linalg.inv(K), h1)
        r2 = np.matmul(lmbda * np.linalg.inv(K), h2)
        r3 = np.cross(r1, r2)
        t = np.matmul(lmbda * np.linalg.inv(K), h3)
        RT = np.vstack((r1, r2, r3, t)).T
        R.append(RT)

    return R



def rms_error_reprojection(K, Distortion, R, images_corners, world_corners):
    alpha = K[0,0]
    gamma = K[0,1]
    beta = K[1,1]
    u0 = K[0,2]
    v0 = K[1,2]
    k1 = Distortion[0]
    k2 = Distortion[1]
    

    # print("Distortion Value :",k1,k2)
    error_all_images = []
    reprojected_corners_all  =[]
    err_list    = []
    for i in range(len(images_corners)):
        img_corners = images_corners[i]
        Rt = R[i]
        
        L = np.dot(K, Rt)
        
        
        reprojected_corners_img = []
        for j in range(len(img_corners)):
            #World Corners
            world_point = world_corners[j]
            model_points = np.array([[world_point[0]],[world_point[1]],[0],[1]])
            #Image Corners
            corners = img_corners[j]
            corners = np.array([[corners[0]],  [corners[1]],[1]], dtype = float)
            
            #world to camera
            transformed_coords = np.matmul(Rt , model_points)
            x =  transformed_coords[0] / transformed_coords[2]
            y = transformed_coords[1] / transformed_coords[2]
            r = np.sqrt(x**2 + y**2)
            #world to image pixel 
            pixel_coords = np.matmul(L , model_points)
            u =  pixel_coords[0] / pixel_coords[2]
            v = pixel_coords[1] / pixel_coords[2]            
            corners_hat_intial =np.array([[u],[v],[1]], dtype = float)


           
            # world to image pixel along with distortion model
            u_hat = u+(u - u0)*(k1*r**2+k2*r**4)
            v_hat = v+(v - v0)*(k1*r**2+k2*r**4)
            corners_hat = np.array([[u_hat],[v_hat],[1]], dtype = float)
            reprojected_corners_img.append((corners_hat[0],corners_hat[1]))
            if(k1==0 and k2 ==0):
                error = np.linalg.norm((corners - corners_hat_intial), 2)
            else :
                error = np.linalg.norm((corners - corners_hat), 2)
            
            err_list.append(error)

        reprojected_corners_all.append(reprojected_corners_img)
        error_all_images.append(mean(err_list))
    return np.array(error_all_images), np.array(reprojected_corners_all)



def loss_func(param, R, images_corners, world_corners):
    
    alpha, gamma, beta, u0, v0, k1, k2 = param
    K = np.array([[alpha, gamma, u0],
                  [0, beta, v0],
                  [0, 0, 1]])
    k_distortion = np.array([[k1],[k2]])
    
    error_all_images = []    
    err_list    = []
    for i in range(len(images_corners)):
        img_corners = images_corners[i]
        Rt = R[i]        
        L = np.dot(K, Rt)   
        for j in range(len(img_corners)):
            #World Corners
            world_point = world_corners[j]
            model_points = np.array([[world_point[0]],[world_point[1]],[0],[1]])
            #Image Corners
            corners = img_corners[j]
            corners = np.array([[corners[0]],  [corners[1]],[1]], dtype = float)
            
            #world to camera
            transformed_coords = np.matmul(Rt , model_points)
            x =  transformed_coords[0] / transformed_coords[2]
            y = transformed_coords[1] / transformed_coords[2]
            r = np.sqrt(x**2 + y**2)
            #world to image pixel 
            pixel_coords = np.matmul(L , model_points)
            u =  pixel_coords[0] / pixel_coords[2]
            v = pixel_coords[1] / pixel_coords[2]            
            corners_hat_intial =np.array([[u],[v],[1]], dtype = float)


           
            # world to image pixel along with distortion model
            u_hat = u+(u - u0)*(k1*r**2+k2*r**4)
            v_hat = v+(v - v0)*(k1*r**2+k2*r**4)
            corners_hat = np.array([[u_hat],[v_hat],[1]], dtype = float)
            
            if(k1==0 and k2 ==0):
                error = np.linalg.norm((corners - corners_hat_intial), 2)
            else :
                error = np.linalg.norm((corners - corners_hat), 2)
            
            err_list.append(error)

        
        error_all_images.append(mean(err_list))
    return error_all_images

def plot_reproj_images(images, K_new,K_distortion_new,reprojected_corners_all_modified):

    K_distortion_new = np.array([K_distortion_new[0], K_distortion_new[1], 0, 0, 0], dtype = float)
    for i in range(len(images)):
        img1 = images[i]
        img1 =copy.deepcopy(img1)
        img1 = cv2.undistort(img1, K_new, K_distortion_new)
        corners = reprojected_corners_all_modified[i]
        for j in range(len(corners)):
            cv2.circle(img1, (int(corners[j][0]),int(corners[j][1])), 7, (0,0,255), -1)
        cv2.imwrite("Misc/Results/" + "Output_" + str(i) + ".png", img1)
        

def main():

    #Read Images from a  Calibration_Imgs folder
    ImagePath = "Misc/Calibration_Imgs"
    images = readImages(ImagePath)
    original_images = images
    # The images in Calibration Imgs are of 10*7 hence we are taking the corners in 9*6 
    X_limit = 9
    Y_limit = 6
    CheckerBoard_Width = 21.5
    
    wP =  worldPoints(X_limit, Y_limit, CheckerBoard_Width)
    corners_images, allH =imagePointH(images, X_limit, Y_limit, wP)
   
    K = computeIntrinsics(allH)
    RT = computeEntrinsics(K,allH)
   
    Distortion = np.array([0.0,0.0])
    error_all_images,_ = rms_error_reprojection(K, Distortion, RT, corners_images, wP)
    print("Initial Error Estimate : ",np.mean(error_all_images,dtype = float))
    alpha = K[0,0]
    gamma = K[0,1]
    beta = K[1,1]
    u0 = K[0,2]
    v0 = K[1,2]
    k1 = Distortion[0]
    k2 = Distortion[1]
    param = np.array([alpha, gamma, beta, u0, v0, k1, k2])
    resultant_parameters = optimize.least_squares(loss_func, x0=param, method="lm", args=[RT, corners_images, wP])
    
    
    alpha_new, gamma_new, beta_new, u0_new, v0_new, k1_new, k2_new = resultant_parameters.x
    K_new= np.array([[alpha_new, gamma_new, u0_new],
                  [0, beta_new, v0_new],
                  [0, 0, 1]])
    K_distortion_new = np.array([[k1_new],[k2_new]])
    print("Updated K matrix")
    print(K_new)
    print("Updated Distortion matrix")
    print(K_distortion_new)
    error_all_images_modified,reprojected_corners_all_modified = rms_error_reprojection(K_new, K_distortion_new, RT, corners_images, wP)
    print("Final Error Estimate : ",np.mean(error_all_images_modified,dtype = float))
    plot_reproj_images(original_images, K_new,K_distortion_new,reprojected_corners_all_modified)

if __name__ == "__main__":
    main()
    # print("running")

    