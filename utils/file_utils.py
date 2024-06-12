from utils.camera_info import Camera_info
import json, pickle, os
import numpy as np

def check_path(path):
    if not os.path.exists(path):
        os.system(f'mkdir "{path}"')
        
def numpy_array_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, list):
        return [numpy_array_to_list(item) for item in obj]
    if isinstance(obj, dict):
        return {key: numpy_array_to_list(value) for key, value in obj.items()}
    return obj
        
def read_camera_matrix(file_path):
    with open(file_path, "rb") as pkl_file:
        data = pickle.load(pkl_file)
    
    return data["camera_matrix"]

def read_distortion(file_path):
    with open(file_path, "rb") as pkl_file:
        data = pickle.load(pkl_file)
    
    return data["distortion"]
    
def write_pkl_file(Cameras : Camera_info, save_path):
    dict_info = {
        "name" : Cameras.img_name,
        "img_path" : Cameras.img_path,
        "camera_matrix" : Cameras.camera_matrix,
        "extrinsic_matrix" : Cameras.extrinsic,
        "distortion" : Cameras.distortion,
        "rotation_matrix" : Cameras.rotation_matrix,
        "translation_vector" : Cameras.translation_vector,
        "aruco_type" : Cameras.aruco_type
    }
    with open(save_path, "wb") as pkl_file:
        pickle.dump(dict_info, pkl_file)
    
def write_json_file(Cameras : Camera_info, save_path):
    dict_info = {
        "name" : Cameras.img_name,
        "img_path" : Cameras.img_path,
        "camera_matrix" : Cameras.camera_matrix.tolist(),
        "extrinsic_matrix" : Cameras.extrinsic.tolist(),
        "distortion" : Cameras.distortion.tolist(),
        "rotation_matrix" : Cameras.rotation_matrix.tolist(),
        "translation_vector" : Cameras.translation_vector.tolist(),
        "aruco_type" : Cameras.aruco_type
    }
    with open(save_path, "w") as json_file:
        json.dump(dict_info, json_file, indent=4)