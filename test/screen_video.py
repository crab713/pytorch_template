import imageio
import random
import cv2

filename = 'test.mp4'

# vid = imageio.get_reader(filename,  'ffmpeg')
# num_frames=0
# for num,_ in enumerate(vid):
#     num_frames = num
# print(num_frames)


def screen_video(video_info : dict):
        """对单个视频的取帧规则,
        Inputs:
            video_info: 待裁剪的视频信息: {name, total_frame, labels}
        
        Outputs:
            output: 处理后的视频数组,shape=[frame, height, width, channel]
        """
        # TODO
        frame = 8
        capture = cv2.VideoCapture(video_info['name'])
        start_index = random.randint(0, video_info['total_frame'] - frame*18)

        frames = []
        for i in range(frame):
            index = start_index + i * 15
            capture.set(propId=cv2.CAP_PROP_POS_FRAMES, value=index) # 跳到指定帧
            hasframe, image1 = capture.read()
            frames.append(image1)
        
        return frames

frames = screen_video({'name':'test.mp4', 'total_frame':1646, 'labels':True})
print(len(frames))
print(frames[0].shape)