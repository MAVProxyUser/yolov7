import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

import pyrealsense2 as rs
import numpy as np

from dynio import *
import threading
import Jetson.GPIO as GPIO

def fire():
    print("start firing")
    output_pin = 16  # BOARD pin number
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(output_pin, GPIO.OUT)

    GPIO.output(output_pin, GPIO.HIGH)
    time.sleep(.3)
    GPIO.output(output_pin, GPIO.LOW)
    print("stop firing")

def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, depth, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.depth, not opt.no_trace

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
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    if opt.depth == "yes":
        print("using depth camera")
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    pipeline = rs.pipeline()
    profile = pipeline.start(config)

    sensors = profile.get_device().query_sensors()

# Results in dark indoor imagery
#    for sensor in sensors:
#        if sensor.supports(rs.option.auto_exposure_priority):
#            #print('Start setting AE priority.')
#            aep = sensor.get_option(rs.option.auto_exposure_priority)
#            print(str(sensor),' supports AEP.')
#            print('Original AEP = %d' %aep)
#            aep = sensor.set_option(rs.option.auto_exposure_priority, 0)
#            aep = sensor.get_option(rs.option.auto_exposure_priority)
#            print('New AEP = %d' %aep)
#            ep = sensor.set_option(rs.option.exposure, 78)

    align_to = rs.stream.color
    align = rs.align(align_to)

    nohuman = 0
    shootit = 0
    frames_counter = 0
    start_time = time.time()

    while(True):

        frames = pipeline.wait_for_frames()

        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        if opt.depth == "yes":
            depth_frame = aligned_frames.get_depth_frame()
            if not depth_frame:
                continue

        if not color_frame:
            continue

        img = np.asanyarray(color_frame.get_data())
        if opt.depth == "yes":
            depth_image = np.asanyarray(depth_frame.get_data())
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
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if names[int(cls)] == "person":
                        # https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/
                        # Yolo coordinates array - normalized [×_center, _center, width, height]
                        # In yolo, a bounding box is represented by four values [x_center, y_center, width, height].
                        # x_center and y_center are the normalized coordinates of the center of the bounding box.
                        # To make coordinates normalized, we take pixel values of x and y, which marks the center
                        # of the bounding box on the x- and y-axis. Then we divide the value of x by the width of
                        # the image and value of y by the height of the image. width and height represent the width
                        # and the height of the bounding box. They are normalized as well.
                        #
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
                        #
                        # These coordinates are seemingly NOT stored in the above mentioned format, rather in
                        # [x_topleft, y_topleft, width, height] format. These are COCO style detection coordinates

                        x = float(xyxy[0])
                        y = float(xyxy[1])
                        w = float(xyxy[2])
                        h = float(xyxy[3])

                        hit_x = float( (xyxy[0] + w)/2 )
                        hit_y = float( (xyxy[1] + h)/2 )

                        hit_y = hit_y - 20

                        c = int(cls)  # integer class
                        label = f'({x},{y}) - {names[c]}'

                        # Draw green circle around center pixel.
                        cv2.circle(im0, (int(hit_x), int(hit_y)), int(15), (0,255,0), 5)

                        # Draw bounding box around target on regular camera view, label as class such as "person".
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)

                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        d1, d2 = int((int(xyxy[0])+int(xyxy[2]))/2), int((int(xyxy[1])+int(xyxy[3]))/2)
                        if opt.depth == "yes":
                            target_depth = depth.get_distance(int(d1),int(d2))

                            # Draw bounding box around target on depth camera view, label with depth of class in question.
                            depthlabel = str(round((target_depth* 39.3701 ),2))+"in "+str(round((target_depth* 100 ),2))+" cm"
                            plot_one_box(xyxy, depth_colormap, label=label, color=colors[int(cls)], line_thickness=2)

                        if opt.verbose == "yes":
                            print("I see you! At coords:")
                            print("X: " + str(hit_x))
                            print("Y: " + str(hit_y))
                            if opt.depth == "yes":
                                print("Depth: " + depthlabel)

                        if opt.servo == "yes":
                             positiony = mx_28_y.get_position()
                             positionx = mx_28_x.get_position()
                             angley = mx_28_y.get_angle()
                             anglex = mx_28_x.get_angle()

                        if opt.verbose == "yes":
                             print("Servo position info:")
                             print("Servo X: " + str(positionx))
                             print("Servo Y: " + str(positiony))
                             print("Servo Angle x: " + str(anglex))
                             print("Servo Angle y: " + str(angley))

                        if hit_x >= 480:
                             if opt.servo == "yes":
                                 mx_28_x.set_angle(anglex-10)
                             print("Clockwise")

                        elif hit_x <= 160:
                             if opt.servo == "yes":
                                 mx_28_x.set_angle(anglex+10)
                             print("Counter clockwise")

                        elif hit_y > 240:
                             if opt.servo == "yes":
                                 mx_28_y.set_angle(angley-10)
                             print("Down")

                        elif hit_y < 120:
                             if opt.servo == "yes":
                                 mx_28_y.set_angle(angley+10)
                             print("Up")
                        else:
                             print("Locked")
                             shootit = shootit +1
                             if shootit >= 5:
                                 print("shoot it!")
                                 thread = threading.Thread(target=fire, daemon=True)
                                 thread.start()
                                 thread.join()
                                 shootit = 0
                    else:
                        nohuman=nohuman+1
                        #print("no human count: " + str(nohuman))
                        #print("saw a: " + names[int(cls)])
                        if nohuman >= 75:
                            if opt.servo == "yes":
                                mx_28_y.set_angle(180)
                                mx_28_x.set_angle(180)
                            nohuman=0

            # Print time (inference + NMS)
            #print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            if opt.verbose == "yes":
                frames_counter += 1
                elapsed_time = time.time() - start_time
                if elapsed_time > 1:
                    fps = frames_counter / elapsed_time
                    print('FPS: ',(fps))
                    frames_counter = 0
                    start_time = time.time()

            # Stream results
            cv2.namedWindow("Recognition result", cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow("Recognition result", 640,480)
            cv2.imshow("Recognition result", im0)
            if opt.depth == "yes":
                cv2.namedWindow("Recognition result depth", cv2.WINDOW_KEEPRATIO)
                cv2.resizeWindow("Recognition result depth", 640,480)
                cv2.imshow("Recognition result depth",depth_colormap)
                cv2.moveWindow("Recognition result depth", 0, 480)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7-tiny.pt', help='model.pt path(s)')
    parser.add_argument('--servo', type=str, default='no', help='source')  # "no" to disable
    parser.add_argument('--depth', type=str, default='yes', help='source')  # "no" to disable
    parser.add_argument('--verbose', type=str, default='no', help='source')  # "no" to disable
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

    if opt.servo == "yes":
        print("centering dynamixel")
        dxl_io = dxl.DynamixelIO('/dev/ttyUSB0', baud_rate=57600)
        mx_28_y = dxl_io.new_mx28(1, 1)  # MX-64 protocol 1 with ID 2
        mx_28_x = dxl_io.new_mx28(2, 1)  # MX-64 protocol 1 with ID 2

        mx_28_y.torque_enable()
        mx_28_x.torque_enable()

        mx_28_y.set_angle(180)
        mx_28_x.set_angle(180)



    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
    GPIO.cleanup()
