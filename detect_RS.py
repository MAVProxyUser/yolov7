import argparse
import time
from time import sleep
from pathlib import Path
import os
import cv2
import torch
from numpy import random
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, \
    scale_coords, strip_optimizer, set_logging, increment_path
from utils.torch_utils import select_device, load_classifier, TracedModel
import pyrealsense2 as rs
import numpy as np
import HiwonderServoController as servo

#Setup the HiWonder Servo
servo.setConfig('/dev/ttyUSB0', 1)
g = [1,5]
tilt, pan = g

#Setup GPIO
os.system('echo 20 > /sys/class/gpio/export')
os.system('echo out > /sys/class/gpio/gpio20/direction')


#Query and Display current servo position
boot_pos = servo.multServoPosRead(g)
print("Boot Up Servo Position: Tilt/Pan")
print(boot_pos[1],boot_pos[5])

#Print out current DC IN voltage
print(servo.getBatteryVoltage())

#Set home position of HiWonder servos
def xy_home():
    servo.moveServo(tilt, 550, 500)
    servo.moveServo(pan, 530, 500)

#Test Pan Sweep
def test_pan():
    for i in range(300, 700):
         servo.moveServo(pan, i, 500)

#Test Tilt
def test_tilt():
    for i in range(400, 500):
         servo.moveServo(tilt, i, 500)

xy_home()
test_tilt()
xy_home()

def fire():
    print("start firing")
    os.system('echo 1 > /sys/class/gpio/gpio20/value')
    sleep(.5)
    os.system('echo 0 > /sys/class/gpio/gpio20/value')
    print("stop firing")


def move_servos(error_x, error_y):
    position = servo.multServoPosRead(g)
    pan_pos = position[5]
    tilt_pos = position[1]
    
    print("Here's the incoming correction coordinates:")
    print (error_x, error_y)
    print("Here's where the servos started:")
    print (pan_pos, tilt_pos)

    #Bring in the error of hit from center frame and scale to servo world
    pan_pos += error_x / 3 if abs(error_x) > 100 else pan_pos == pan_pos
    tilt_pos += error_y / 3 if abs(error_y) > 100 else tilt_pos == tilt_pos

    #Error correct for moving quickly/out of frame
    if abs(error_x) > 200:
        (pan_pos == pan_pos)
    if abs(error_y) > 200:
        (tilt_pos == tilt_pos)

    #Add in the good old anti-nutshot
    antinutshot = 20
    tilt_pos += antinutshot

    #Are we close?
    if abs(error_x) < 100:
        if abs(error_y) < 100:
            fire()

    #Show where we're gonna go
    print("Here's where the servos are going:")
    print(int(pan_pos), int(tilt_pos))

    #Move the servos
    servo.moveServo(pan, int(pan_pos), 400)
    servo.moveServo(tilt, int(tilt_pos), 400)
        
def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    # Configure depth and color streams

    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)

    pipeline = rs.pipeline()
    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: " , depth_scale)

    align_to = rs.stream.color
    align = rs.align(align_to)

    frames_counter = 0
    start_time = time.time()

    while(True):

        t0 = time.time()
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)


        ###comment this out to REMOVE depth
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not aligned_depth_frame or not color_frame:
            continue

        img = np.asanyarray(color_frame.get_data())

        ###comment this out to REMOVE depth
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.08), cv2.COLORMAP_JET)
        depth = frames.get_depth_frame()
        if not depth: continue

        # Letterbox
        im0 = img.copy()
        img = img[np.newaxis, :, :, :]        

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Process detections
        nohuman = 0
        for i, det in enumerate(pred):  # detections per image

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if names[int(cls)] == "person":
                        #                  (0,0)                         (640,0)
                        #                        -----------------------Y
                        #                       |                       |
                        #                       |                       |   ^
                        #                       |                       |   |
                        #                       |       (320,240)       |   (y)
                        #                       |        center         |   |
                        #                       |                       |   ∀
                        #                       |                       |
                        #                        -----------------------
                        #                (0,480)                         (640,480)
                        #                              <---(x)--->
                        w = float(xyxy[2])
                        h = float(xyxy[3])

                        hit_x = float( (xyxy[0] + w)/2 )
                        hit_y = float( (xyxy[1] + h)/2 )

                        center_x = 320
                        center_y = 240
                        
                        error_x = center_x - hit_x
                        error_y = center_y - hit_y 

                        #c = int(cls)  # integer class
                        #[label = f'({x},{y}) - {names[c]}'

                        # Draw green circle around center pixel.
                        #cv2.circle(im0, (int(hit_x), int(hit_y)), int(15), (0,255,0), 5)
                        
                        # Draw bounding box around target on regular camera view, label as class such as "person".
                        #plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)
                        
                        #xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        #d1, d2 = int((int(xyxy[0])+int(xyxy[2]))/2), int((int(xyxy[1])+int(xyxy[3]))/2)
                        #target_depth = depth.get_distance(int(d1),int(d2))

                        # Draw bounding box around target on depth camera view, label with depth of class in question.
                        #depthlabel = str(round((target_depth* 39.3701 ),2))+"in "+str(round((target_depth* 100 ),2))+" cm"
                        #plot_one_box(xyxy, depth_colormap, label=label, color=colors[int(cls)], line_thickness=2)

                        print("I see you! At coords:")
                        print("X: " + str(hit_x))
                        print("Y: " + str(hit_y))
                        print("My Error Correction is:")
                        print("X: " + str(error_x))
                        print("Y: " + str(error_y))
                        #print("Depth: " + depthlabel)

                        move_servos(error_x, error_y)

                    else:
                        nohuman=nohuman+1
                        print("no human count: " + str(nohuman))
                        print("saw a: " + names[int(cls)])
                        if nohuman > 100:
                            nohuman=0
                            xy_home()
                    frames_counter += 1
                    elapsed_time = time.time() - start_time
                    if elapsed_time > 1:
                        fps = frames_counter / elapsed_time
                        print('FPS: ',(fps))
                        frames_counter = 0
                        start_time = time.time()
            # Stream results
#            cv2.namedWindow("Recognition result", cv2.WINDOW_KEEPRATIO)
#            cv2.resizeWindow("Recognition result", 640,480)
#            cv2.imshow("Recognition result", im0)
#            cv2.namedWindow("Recognition result depth", cv2.WINDOW_KEEPRATIO)
#            cv2.resizeWindow("Recognition result depth", 640,480)
#            cv2.imshow("Recognition result depth",depth_colormap)
#            cv2.moveWindow("Recognition result depth", 0, 480)
           
#            if cv2.waitKey(1) & 0xFF == ord('q'):
#                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7-tiny.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
