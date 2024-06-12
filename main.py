from utils.file_utils import *
from utils.aruco_info import *
from utils.camera_info import Camera_info
import numpy as np
import argparse
import cv2
import os, collections

class Calibration:
    def __init__(self, args):
        self.calibrate_folder = os.path.join("camera_data", args["image"])
        self.save_prefix = os.path.join("output", args["image"])
        self.camera = collections.defaultdict(Camera_info)
        self.ARUCO_DICT = get_aruco_dict()
        self.threshold = args["threshold"]
        self.aruco_corners = self.get_aruco_corners(args["length"])
        self.args = args
        
    def stereo_calibrate(self):
        path_prefix = os.path.join(self.calibrate_folder, "images")
        for camera_name in os.listdir(path_prefix):
            Id = camera_name.split('.')[0]

            # set camera information
            self.camera[Id].set_img_name(Id)
            self.camera[Id].set_img_path(os.path.join(path_prefix, camera_name))
            self.camera[Id].set_camera_matrix(read_camera_matrix(os.path.join(f"{self.calibrate_folder}/matrix", f"{Id}.pickle")))
            self.camera[Id].set_distortion(read_distortion(os.path.join(f"{self.calibrate_folder}/matrix", f"{Id}.pickle")))
            self.camera[Id].set_aruco_type(self.args["type"])
            
            vector_rotation, vector_translation, extrinsic = self.aruco_detetion(self.camera[Id])
            if vector_rotation is not None:
                self.camera[Id].set_extrinsic(extrinsic)
                self.camera[Id].set_rotation_matrix(cv2.Rodrigues(vector_rotation.reshape(1,3))[0])
                self.camera[Id].set_translation_vector(vector_translation)
            
                # save camera information
                write_pkl_file(self.camera[Id], f"{self.save_prefix}/{Id}.pkl")
                write_json_file(self.camera[Id], f"{self.save_prefix}/{Id}.json")
        
    def aruco_detetion(self, camera : Camera_info):
        # print(f"loading image: {camera.img_path}...")
        ori_img = cv2.imread(camera.img_path)

        gray_image = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
        ret, binary_image = cv2.threshold(gray_image, self.threshold, 255, cv2.THRESH_BINARY)
        cv2.imshow("bin_img", binary_image)
        cv2.waitKey(0)
        
        arucoDict = cv2.aruco.Dictionary_get(self.ARUCO_DICT[camera.aruco_type])
        arucoParams = cv2.aruco.DetectorParameters_create()
        
        (corners, ids, rejected) = cv2.aruco.detectMarkers(binary_image, arucoDict, parameters=arucoParams)

        # Verify that at least one ArUco marker was detected
        if len(corners) > 0:
            # Flatten the ArUco IDs list
            ids = ids.flatten()

            # Loop over the detected ArUco corners
            for (markerCorner, markerID) in zip(corners, ids):
                corner = self.set_marker_info(markerCorner)
                for key, val in corner.items():
                    print(f"{key}: {val}")
                # cv2.namedWindow("Image Viewer")
                # cv2.setMouseCallback("Image Viewer", self.mouse_event_handler)
                # cv2.imshow("Image Viewer", ori_img)
                # cv2.waitKey(0)

                # corner['topLeft'] = np.array([249, 271])
                # corner['topRight'] = np.array([420, 272])
                # corner['bottomRight'] = np.array([450, 315])
                # corner['bottomLeft'] = np.array([230, 311])
                
                self.draw_aruco_bbox(ori_img, corner)
                self.draw_aruco_id(ori_img, markerID, corner)
                self.save_aruco_image(camera, markerID, ori_img)
                all_corners = markerCorner.reshape((4, 2))
                # all_corners[0] = corner['topLeft']
                # all_corners[1] = corner['topRight']
                # all_corners[2] = corner['bottomRight']
                # all_corners[3] = corner['bottomLeft']
                
            # SolvePnP
            camera.distortion = np.zeros_like(camera.distortion)
            success, vector_rotation, vector_translation = cv2.solvePnP(
                                                                self.aruco_corners, 
                                                                all_corners, 
                                                                camera.camera_matrix, 
                                                                camera.distortion)
            if not success:
                print("Error: Can't generate extrinsic parameters")
                
            # Rvec, tvec to extrinsic matrix
            extrinsic = np.eye(4,4)
            R = cv2.Rodrigues(vector_rotation.reshape(1,3))
            extrinsic[0:3,0:3] = R[0]
            extrinsic[0:3,3] = vector_translation.reshape(1,3)
            
            self.check(ori_img, extrinsic, camera.camera_matrix)
            
            return vector_rotation, vector_translation, extrinsic
        else:
            print("No Aruco marker detected")
            return None, None, None
                
    def set_marker_info(self, markerCorner):
        (topLeft, topRight, bottomRight, bottomLeft) = markerCorner.reshape((4, 2))
        return {
            'topLeft': topLeft,
            'topRight': topRight,
            'bottomRight': bottomRight,
            'bottomLeft': bottomLeft
        }

    def draw_aruco_bbox(self, ori_image, corner):
        
        cv2.line(ori_image, corner['topLeft'].astype(int), corner['topRight'].astype(int), (0, 255, 0), 2)
        cv2.line(ori_image, corner['topRight'].astype(int), corner['bottomRight'].astype(int), (0, 255, 0), 2)
        cv2.line(ori_image, corner['bottomRight'].astype(int), corner['bottomLeft'].astype(int), (0, 255, 0), 2)
        cv2.line(ori_image, corner['bottomLeft'].astype(int), corner['topLeft'].astype(int), (0, 255, 0), 2)
         
    def compute_marker_center(self, corner):
        cX = int((corner['topLeft'][0] + corner['bottomRight'][0]) / 2.0)
        cY = int((corner['topLeft'][1] + corner['bottomRight'][1]) / 2.0)
        return cX, cY
    
    def draw_aruco_id(self, ori_image, markerID, corner):
        (cX, cY) = self.compute_marker_center(corner)
        # cv2.circle(ori_image, (cX, cY), 4, (0, 0, 255), -1)
        # cv2.putText(ori_image, f'({cX}, {cY})', (cX, cY - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # cv2.putText(ori_image, str(markerID), (int(corner['topLeft'][0]), int(corner['topLeft'][1]) - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # DEBUG
        # cX, cY = (int(corner['topRight'][0]), int(corner['topRight'][1]))
        # cv2.circle(ori_image, (cX, cY), 4, (0, 0, 255), -1)
        for i, corner in enumerate([corner['topLeft'], corner['topRight'], corner['bottomRight'], corner['bottomLeft']]):
            cX, cY = (int(corner[0]), int(corner[1]))
            cv2.circle(ori_image, (cX, cY), 4, (0, 0, 255), -1)
            cv2.putText(ori_image, f'({cX}, {cY})', (int(corner[0]), int(corner[1]) + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    def save_aruco_image(self, camera : Camera_info, markerID, ori_image):
        save_path = f"{self.save_prefix}/images/"
        check_path(save_path)
        save_path = os.path.join(save_path, f"{camera.img_name}_{markerID}.png")
        cv2.imwrite(save_path, ori_image)
    
    def get_aruco_corners(self, side_length, offset=[0, 0, 0]):
        corners = np.zeros((4, 3))
        half = side_length / 2

        # Define the corners based on the side_length and offset
        # left-top
        corners[0] = [offset[0] - half, offset[1], offset[2] + half]
        # right-top
        corners[1] = [offset[0] + half, offset[1], offset[2] + half]
        # right-bottom
        corners[2] = [offset[0] + half, offset[1], offset[2] - half]
        # left-bottom
        corners[3] = [offset[0] - half, offset[1], offset[2] - half]

        return corners     
    
    def mouse_event_handler(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            # Print the (x, y) coordinates where the mouse is moving
            print(f"Mouse Position: (X: {x}, Y: {y})")
            
    def check(self, img, extrinsic, camera_matrix):
        z_3d_path_points_list = [[0, 0, 0], [0, 0, 0.5], [0, 0, 1], [0, 0, 1.5], [0, 0, 2]]
        y_3d_path_points_list = [[0, 0, 0], [0, 0.5, 0], [0, 1, 0], [0, 1.5, 0], [0, 2, 0]]
        x_3d_path_points_list = [[0, 0, 0], [0.5, 0, 0], [1, 0, 0], [1.5, 0, 0], [2, 0, 0]]
        _3d_points_list = [[-0.5, 0, -0.5], [0.5, 0, -0.5], [0.5, 0, 0.5], [-0.5, 0, 0.5]]
            
        for _3d_points in _3d_points_list:
            _2d_points = camera_matrix @ extrinsic[:3, :] @ np.append(_3d_points, 1)
            _2d_points = _2d_points / _2d_points[2]
            cv2.circle(img, (int(_2d_points[0]), int(_2d_points[1])), 3, (0, 255, 0), -1)
            # print(_2d_points[:2])

        for x, y, z in zip(x_3d_path_points_list, y_3d_path_points_list, z_3d_path_points_list):
            _2d_points = camera_matrix @ extrinsic[:3, :] @ np.append(x, 1)
            _2d_points = _2d_points / _2d_points[2]
            cv2.circle(img, (int(_2d_points[0]), int(_2d_points[1])), 3, (0, 0, 255), -1)
            # print(_2d_points[:2])

            _2d_points = camera_matrix @ extrinsic[:3, :] @ np.append(y, 1)
            _2d_points = _2d_points / _2d_points[2]
            cv2.circle(img, (int(_2d_points[0]), int(_2d_points[1])), 3, (0, 255, 100), -1)
            # print(_2d_points[:2])

            _2d_points = camera_matrix @ extrinsic[:3, :] @ np.append(z, 1)
            _2d_points = _2d_points / _2d_points[2]
            cv2.circle(img, (int(_2d_points[0]), int(_2d_points[1])), 3, (255, 0, 0), -1)
            # print(_2d_points[:2])
            
        cv2.imshow(f"img", img)
        cv2.waitKey(0)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", type=str, required=True,
	                help="path to input image containing ArUCo tag")
    ap.add_argument("-t", "--type", type=str, required=True,
	                help="type of ArUCo tag to detect")
    ap.add_argument("-l", "--length", type=int, required=True,
	                help="ArUCo tag side length")
    ap.add_argument("-thres", "--threshold", type=int, default=210,
                    help="the black and white threshold of binary image")
    ap.add_argument("-vis", "--visualize", action="store_true", 
                    help="Display binary threshold result")
    
    args = vars(ap.parse_args())

    calibration = Calibration(args)
    calibration.stereo_calibrate()
    # python main.py -i unity -t DICT_5X5_100 -l 1 -thres 130
