import numpy as np


class Camera_info:
    def __init__(self):
        self.img_name = ""
        self.img_path = ""
        self.camera_matrix = np.zeros((3, 3))
        self.extrinsic = np.eye(4, 4)
        self.distortion = np.zeros(5)
        self.rotation_matrix = np.zeros((3, 3))
        self.translation_vector = np.zeros((3, 1))
        self.aruco_type = "DICT_ARUCO_ORIGINAL"
        
    def set_img_name(self, name):
        self.img_name = name

    def set_img_path(self, path):
        self.img_path = path

    def set_camera_matrix(self, mat):
        self.camera_matrix = mat
        
    def set_extrinsic(self, extrinsic):
        self.extrinsic = extrinsic

    def set_distortion(self, dist):
        self.distortion = dist

    def set_rotation_matrix(self, R):
        self.rotation_matrix = R

    def set_translation_vector(self, V):
        self.translation_vector = V
        
    def set_aruco_type(self, type):
        self.aruco_type = type