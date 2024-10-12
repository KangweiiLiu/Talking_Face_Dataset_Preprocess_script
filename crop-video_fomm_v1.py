import face_alignment
import skimage.io
import numpy
from argparse import ArgumentParser
from skimage import img_as_ubyte
from skimage.transform import resize
from tqdm import tqdm
import os
import imageio
import numpy as np
import warnings
import cv2
warnings.filterwarnings("ignore")

def extract_bbox(frame, fa):
    if max(frame.shape[0], frame.shape[1]) > 640:
        scale_factor =  max(frame.shape[0], frame.shape[1]) / 640.0
        frame = resize(frame, (int(frame.shape[0] / scale_factor), int(frame.shape[1] / scale_factor)))
        frame = img_as_ubyte(frame)
    else:
        scale_factor = 1
    frame = frame[..., :3]
    bboxes = fa.face_detector.detect_from_image(frame[..., ::-1])
    if len(bboxes) == 0:
        return []
    return np.array(bboxes)[:, :-1] * scale_factor

def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def join(tube_bbox, bbox):
    xA = min(tube_bbox[0], bbox[0])
    yA = min(tube_bbox[1], bbox[1])
    xB = max(tube_bbox[2], bbox[2])
    yB = max(tube_bbox[3], bbox[3])
    return (xA, yA, xB, yB)

def compute_bbox(frame_count, start, end, fps, tube_bbox, frame_shape, inp, image_shape, increase_area=0.1, index=0):
    left, top, right, bot = tube_bbox
    width = right - left
    height = bot - top

    width_increase = max(increase_area, ((1 + 2 * increase_area) * height - width) / (2 * width))
    height_increase = max(increase_area, ((1 + 2 * increase_area) * width - height) / (2 * height))

    left = int(left - width_increase * width)
    top = int(top - height_increase * height)
    right = int(right + width_increase * width)
    bot = int(bot + height_increase * height)

    top, bot, left, right = max(0, top), min(bot, frame_shape[0]), max(0, left), min(right, frame_shape[1])
    h, w = bot - top, right - left

    start = start / fps
    end = end / fps
    time = end - start

    scale = f'{image_shape[0]}:{image_shape[1]}'
    
    input_name = os.path.splitext(os.path.basename(inp))[0]
    output_dir = os.path.dirname(inp)
    
    output_name = f'{input_name}_crop_{index}_{frame_count}.mp4'
    output_path = os.path.join(output_dir, output_name)

    return f'ffmpeg -i {inp} -ss {start} -t {time} -filter:v "crop={w}:{h}:{left}:{top}, scale={scale}" -r 25 {output_path}'

def compute_bbox_trajectories(frame_count,trajectories, fps, frame_shape, args):
    commands = []
    for i, (bbox, tube_bbox, start, end) in enumerate(trajectories):
        end = end-args.frame_interval
        if (end - start) > args.min_frames:
            command = compute_bbox(frame_count,start, end, fps, tube_bbox, frame_shape, inp=args.inp, image_shape=args.image_shape, increase_area=args.increase, index=i)
            commands.append(command)
    return commands

def process_video(args):
    device =  'cuda'
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device=device)
    video = cv2.VideoCapture(args.inp)

    trajectories = []
    previous_frame = None
    fps = video.get(cv2.CAP_PROP_FPS)
    commands = []
    frame_count = 0
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # try:
    with tqdm(total=total_frames) as pbar:
        while True:
            ret, frame = video.read()
            if not ret:
                break

            frame_count += 1
            pbar.update(1)

            if frame_count % args.frame_interval == 0 or frame_count == 1:
                frame_shape = frame.shape
                frame_resized = cv2.resize(frame, (480, int(480 * frame.shape[0] / frame.shape[1])))
                bboxes = extract_bbox(frame_resized, fa)
                
                scale_factor = frame.shape[1] / frame_resized.shape[1]
                bboxes = np.array(bboxes)
                bboxes = bboxes.astype(float) * scale_factor  # 将bboxes转换为浮点数类型

                not_valid_trajectories = []
                valid_trajectories = []

                for trajectory in trajectories:
                    tube_bbox = trajectory[0]
                    intersection = 0
                    for bbox in bboxes:
                        intersection = max(intersection, bb_intersection_over_union(tube_bbox, bbox))
                    if intersection > args.iou_with_initial:
                        valid_trajectories.append(trajectory)
                    else:
                        not_valid_trajectories.append(trajectory)
                
                commands += compute_bbox_trajectories(frame_count,not_valid_trajectories, fps, frame_shape, args)
                # print(frame_count,commands)
                trajectories = valid_trajectories

                for bbox in bboxes:
                    intersection = 0
                    current_trajectory = None
                    for trajectory in trajectories:
                        tube_bbox = trajectory[0]
                        current_intersection = bb_intersection_over_union(tube_bbox, bbox)
                        if intersection < current_intersection and current_intersection > args.iou_with_initial:
                            intersection = bb_intersection_over_union(tube_bbox, bbox)
                            current_trajectory = trajectory

                    if current_trajectory is None:
                        trajectories.append([bbox, bbox, frame_count, frame_count])
                    else:
                        current_trajectory[3] = frame_count
                        current_trajectory[1] = join(current_trajectory[1], bbox)

            elif trajectories:
                for trajectory in trajectories:
                    if frame_count - trajectory[3] <= args.frame_interval:
                        trajectory[3] = frame_count

    # except Exception as e:
    #     print(f"发生错误: {e}")

    # finally:
    #     video.release()

    commands += compute_bbox_trajectories(total_frames,trajectories, fps, frame_shape, args)
    return commands

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--image_shape", default=(256, 256), type=lambda x: tuple(map(int, x.split(','))),
                        help="图像形状")
    parser.add_argument("--increase", default=0.1, type=float, help='增加边界框的比例')
    parser.add_argument("--iou_with_initial", type=float, default=0.25, help="与初始边界框的最小允许IOU")
    parser.add_argument("--inp", required=True, help='输入图像或视频')
    parser.add_argument("--min_frames", type=int, default=100,  help='最小帧数')
    parser.add_argument("--cpu", dest="cpu", action="store_true", help="使用CPU模式")
    parser.add_argument("--frame_interval", type=int, default=10, help="处理帧的间隔")
    parser.add_argument("--output", default="output.txt", help="输出文件路径")
    args = parser.parse_args()

    commands = process_video(args)
    with open(args.output, 'a+') as f:
        for command in commands:
            print(command)
            f.write(command + '\n')
    