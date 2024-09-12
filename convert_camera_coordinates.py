
import numpy as np
import cv2
import pickle

    
class convert_camera_coordinates:
    def __init__(self, camera_matrix, extrinsic_matrix, rotation_matrix, camera_transfrom):
        self.camera_matrix = camera_matrix
        self.extrinsic_matrix = extrinsic_matrix
        self.rotation_matrix = rotation_matrix
        self.camera_transfrom = camera_transfrom

    def unity_to_reality_coordinates(self, coords):
        """
        Converts Unity coordinates to reality coordinates.

        Parameters:
        coords (np.ndarray): A NumPy array with shape (n, 3), where n is the number of coordinates.
                            Each row should contain (x, y, z) coordinates.

        Returns:
        np.ndarray: A NumPy array with shape (n, 3) containing the converted reality coordinates (x, -y, -z).
        """            
        if coords.shape[1] == 4:        
            converted_coords = np.empty_like(coords)
            converted_coords[:, 0] = coords[:, 0]  
            converted_coords[:, 1] = -coords[:, 1]  
            converted_coords[:, 2] = coords[:, 2]  
            converted_coords[:, 3] = coords[:, 3]  
            return converted_coords

        if coords.shape[1] == 3:
            converted_coords = np.empty_like(coords)
            converted_coords[:, 0] = coords[:, 0]  
            converted_coords[:, 1] = -coords[:, 1]  
            converted_coords[:, 2] = coords[:, 2]

        return converted_coords

    def world_point_to_screen_point(self, world_points, camera_matrix, extrinsic_matrix):
        camera_matrix = np.hstack((camera_matrix, np.zeros((camera_matrix.shape[0], 1))))
        world_points = self.unity_to_reality_coordinates(world_points)
        image_points = camera_matrix @ extrinsic_matrix @ world_points.T
        print("depth",image_points[2])
        image_points = image_points / image_points[2]
        return image_points[:2]
    
    def screen_point_to_world_point(self, scream_points, depth):
        return np.linalg.inv(self.rotation_matrix) @ (np.linalg.inv(self.camera_matrix) @ scream_points.T * depth - self.camera_transfrom.T)
        

    
    def check(self):
        _3d_path_points_list = [[0,0,0], [0,1,0], [0,2,0], [0,3,0], [0,4,0], [0,5,0]]
        for _3d_points in _3d_path_points_list:
            _2d_points = self.camera_matrix @ self.extrinsic_matrix[:3, :] @ np.append(_3d_points, 1)
            _2d_points = _2d_points / _2d_points[2]
            print(_2d_points)

    def check_position(self, img_path, points):
        ori_img = cv2.imread(img_path)
        _2d_points = self.camera_matrix @ self.extrinsic_matrix[:3, :] @ np.append(points, 1)
        _2d_points = _2d_points / _2d_points[2]
        cv2.circle(ori_img, (int(_2d_points[0]), int(_2d_points[1])), 3, (0, 255, 0), -1)                                
        cv2.imshow(f"img", ori_img)
        cv2.waitKey(0)
    
    def check_line(self, img_path, start, end):
        ori_img = cv2.imread(img_path)
        start_2d_points = self.camera_matrix @ self.extrinsic_matrix[:3, :] @ np.append(start, 1)
        start_2d_points = start_2d_points / start_2d_points[2]
        end_2d_points = self.camera_matrix @ self.extrinsic_matrix[:3, :] @ np.append(end, 1)
        end_2d_points = end_2d_points / end_2d_points[2]
        cv2.line(ori_img, (int(start_2d_points[0]), int(start_2d_points[1])), (int(end_2d_points[0]), int(end_2d_points[1])), (0, 255, 0), 2)                                
        cv2.imshow(f"img", ori_img)
        cv2.waitKey(0)


if __name__ == "__main__":
    with open("output/unity/camera_1.pkl", "rb") as infile:
        camera = pickle.load(infile)
        camera_matrix = camera["camera_matrix"]
        extrinsic_matrix = camera["extrinsic_matrix"]
        rotation_matrix = camera["rotation_matrix"]

    camera_transfrom = np.array( [[0, 4, 9]] )
    convert_camera = convert_camera_coordinates(camera_matrix, extrinsic_matrix, rotation_matrix, camera_transfrom)
    scream_points = np.array(
        [[387.45, 326.39, 1]]
    )
    depth = 8.57
    # world_points = np.array(
    #     [[1, 5, 1, 1]]
    # )
    
    print(convert_camera.screen_point_to_world_point(scream_points, depth))
    # print(convert_camera.check())
    print(convert_camera.check_line('/home/chenfu/workspaces/CameraCalibration/camera_data/unity/images/camera_1.png', [6, 0, 0], [-6, 0, 0]))
    print(convert_camera.check_line('/home/chenfu/workspaces/CameraCalibration/camera_data/unity/images/camera_1.png', [6, -1, 0], [-6, -1, 0]))
    print(convert_camera.check_line('/home/chenfu/workspaces/CameraCalibration/camera_data/unity/images/camera_1.png', [6, -2, 0], [-6, -2, 0]))
    print(convert_camera.check_position('/home/chenfu/workspaces/CameraCalibration/camera_data/unity/images/camera_1.png', [-1.53, -2.34, 0.40]))
    print(convert_camera.check_position('/home/chenfu/workspaces/CameraCalibration/camera_data/unity/images/camera_1.png', [-1.47, -2.16, 0.48]))
    
 