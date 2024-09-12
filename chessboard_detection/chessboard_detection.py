import numpy as np
import cv2
import os, json, pickle
import argparse
import collections

class Chessboard_detection:
    def __init__(self, row, col, board_size):
        '''
        row: Number of rows in the checkerboard
        col: Number of columns in the checkerboard
        board_size: Length of the checkerboard square in meters
        '''
        self.row = row
        self.col = col
        self.board_size = board_size
        
    def images_list(self, images_folder):
        '''
        images_folder: Folder containing the videos

        Returns a list of images

        If the images are in a video format, the function will extract frames from the video
        '''
        images_prefix = f"camera_calibration_data/{images_folder}"
        images = []

        for images_name in os.listdir(images_prefix):
            image_path = os.path.join(images_prefix, images_name)
            if images_name.endswith(".mp4"):
                frames = self.extract_frames_from_video(image_path)
            else:
                img = cv2.imread(image_path, 1)
                frames = [cv2.imread(image_path, 1)]

            images.extend(frames)

        return images

    def extract_frames_from_video(self, video_path):
        '''
        video_path: Path to the video

        Returns a list of frames
        '''
        frames = []
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        # Take one frame every 10 frames if there are more than 10 frames
        if len(frames) >= 10:
            frames = [frames[i] for i in range(0, len(frames), len(frames) // 10)]

        return frames
    
    def numpy_array_to_list(self, obj):
        '''
        obj: Numpy array

        Returns a list
        '''
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, list):
            return [self.numpy_array_to_list(item) for item in obj]
        if isinstance(obj, dict):
            return {key: self.numpy_array_to_list(value) for key, value in obj.items()}
        return obj
        
    def calibrate(self, images_folder):
        '''
        images_folder: Folder containing the camera matrix and distortion coefficients

        This function will calibrate the camera and save the camera matrix and distortion coefficients
        '''
        save_path = f"output/{images_folder}"

        images = self.images_list(images_folder)
        width, height = images[0].shape[1], images[0].shape[0]
        
        #criteria used by checkerboard pattern detector.
        #Change this if the code can't find the checkerboard
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        #coordinates of squares in the checkerboard world space
        objp = np.zeros((self.row*self.col, 3), np.float32)
        objp[:,:2] = np.mgrid[0:self.row, 0:self.col].T.reshape(-1,2)
        objp = self.board_size * objp

        #Pixel coordinates of checkerboards
        imgpoints = [] # 2d points in image plane.
        objpoints = [] # 3d point in real world space
        img_p = collections.defaultdict(list)
        obj_p = collections.defaultdict(list)
        
        for idx, frame in enumerate(images):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
            #find the checkerboard
            ret, corners = cv2.findChessboardCorners(gray, (self.row, self.col), None)
            
            if ret == True:
                #Convolution size used to improve corner detection. Don't make this too large.
                conv_size = (11, 11)
    
                #opencv can attempt to improve the checkerboard coordinates
                corners = cv2.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)
                oreintation = "LR" if corners[0][0][0] < corners[-1][0][0] else "RL"
                img_p[oreintation].append(corners)
                obj_p[oreintation].append(objp)
                
                cv2.drawChessboardCorners(frame, (self.row,self.col), corners, ret)
                cv2.imshow('img', frame)
                k = cv2.waitKey(0)
    
        
        if (len(img_p["LR"]) > len(img_p["RL"])):
            imgpoints = img_p["LR"]
            objpoints = obj_p["LR"]
            print("LR")
        else:
            imgpoints = img_p["RL"]
            objpoints = obj_p["RL"]
            print("RL")

        # Calibrate the camera.
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
        
        info_dict = {
            "width" : width,
            "height" : height,
            "camera_matrix" : mtx,
            "distortion" : dist
        }
        with open(f"{save_path}.pickle", "wb") as outfile:
            pickle.dump(info_dict, outfile)
        
        with open(f"{save_path}.json", "w") as outfile:
            json.dump(self.numpy_array_to_list(info_dict), outfile, indent=4)
        
        
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-l", "--length", type=float, required=True,
	                help="Length of the checkerboard square in meters")
    ap.add_argument("-r", "--row", type=int, required=True,
                    help="Number of rows in the checkerboard")
    ap.add_argument("-c", "--col", type=int, required=True,
                    help="Number of columns in the checkerboard")
    ap.add_argument("-f", "--folder", type=str, required=True,
                    help="Folder containing the images/videos")
    args = vars(ap.parse_args())
    
    Calibrate = Chessboard_detection(args["row"], args["col"], args["length"])
    Calibrate.calibrate(args["folder"])
    
    # python chessboard_detection.py -l 1.25 -r 3 -c 5 -f camera_1