import cv2
import os
import argparse

class jpgtomp4:
    def __init__(self, input_folder:str, output_video:str, frame_rate:int):
        '''
        input_folder: str
            Folder containing jpg images
        output_video: str
            Output video file name
        frame_rate: int
            Frame rate of the output video

        Example:
        jpgtomp4("input_folder", "output_video", frame_rate).jpg_to_mp4()
        '''
        self.input_folder = input_folder
        self.output_video = output_video
        self.frame_rate = frame_rate
    
    def jpg_to_mp4(self):
        image_files = [os.path.join(self.input_folder, f)
                        for f in os.listdir(self.input_folder) if f.endswith('.jpg')]
        image_files.sort()

        img = cv2.imread(image_files[0])
        height, width, _ = img.shape

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(self.output_video, fourcc, self.frame_rate,
                                       (width, height))

        for image_file in image_files:
            img = cv2.imread(image_file)
            video_writer.write(img)

        video_writer.release()

if __name__ == "__main__":
    # jpgtomp4("input_folder", "output_video", frame_rate).jpg_to_mp4()
    # jpgtomp4("camera_1", "chessboard_1.mp4", 30).jpg_to_mp4()

    # add argument parser
    parser = argparse.ArgumentParser(description='Convert jpg images to mp4 video')

    parser.add_argument('--input_folder', type=str, required=True,
                        help='Folder containing jpg images')
    parser.add_argument('--output_video', type=str, required=True,
                        help='Output video file name')
    parser.add_argument('--frame_rate', type=int, required=True,
                        help='Frame rate of the output video')

    args = parser.parse_args()

    jpgtomp4(args.input_folder, args.output_video, args.frame_rate).jpg_to_mp4()