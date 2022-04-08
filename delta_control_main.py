#!/usr/bin/env python3

from queue import Queue
from threading import Thread, Event

import cv2
import depthai as dai
import numpy as np
import argparse
import time
import blobconverter

from delta_work_runner import process_work_queue, PICKUP_WORK, TOGGLE_PICKUP_ENABLED, GOTO_REF_PICKUP_WORK, GOTO_REF_READY_WORK, GOTO_REF_DROP_WORK, CLOSE_HAND_WORK, OPEN_HAND_WORK

# indices for bbox
BX1 = 1
BY1 = 0
BX2 = 3
BY2 = 2

# indices for crop cordinates
CX1 = 0
CY1 = 1
CX2 = 2
CY2 = 3

NN_WIDTH = 192
NN_HEIGHT = 192
NN_RATIO = NN_WIDTH/NN_HEIGHT

PREVIEW_WIDTH = 1080
PREVIEW_HEIGHT = 1080
PREVIEW_RATIO = PREVIEW_WIDTH/PREVIEW_HEIGHT

CROP_BOX_X1 = 0.35
CROP_BOX_Y1 = 0.2
CROP_BOX_X2 = 0.85
CROP_BOX_Y2 = CROP_BOX_Y1 + \
    (PREVIEW_RATIO*(CROP_BOX_X2 - CROP_BOX_X1)/NN_RATIO)
CROP_BOX = [CROP_BOX_X1, CROP_BOX_Y1, CROP_BOX_X2, CROP_BOX_Y2]

MAX_AREA = 0.5
MIN_AREA = 0.05

#BGR
COLOR_BLUE = (255, 0, 0)
COLOR_GRAY = (200, 200, 200)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_CYAN = (255, 255, 0)

DETECT_PICKUP_START_Y = 0.2
DETECT_PICKUP_END_Y = 0.5
CROP_BOX_ARR = np.array([CROP_BOX_X1, CROP_BOX_Y1, CROP_BOX_X2, CROP_BOX_Y2])

start_time = time.time()
print('START'.format(start_time))
# time.sleep(5)
# elasped_time = time.time() - start_time
# print('ELAPS {0:.3}'.format(elasped_time))


'''
Mobile object localizer demo running on device on RGB camera.
Run as:
python3 -m pip install -r requirements.txt
python3 main.py

Link to the original model:
https://tfhub.dev/google/lite-model/object_detection/mobile_object_localizer_v1/1/default/1

Blob taken from:
https://github.com/PINTO0309/PINTO_model_zoo/tree/main/151_object_detection_mobile_object_localizer
'''

# --------------- Arguments ---------------
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--threshold', type=float,
                    help="Coonfidence threshold", default=0.2)

args = parser.parse_args()
THRESHOLD = args.threshold
NN_PATH = blobconverter.from_zoo(
    name="mobile_object_localizer_192x192", zoo_type="depthai")

# nn data (bounding box locations) are in <0..1> range - they need to be normalized with frame width/height
# def frameNorm(frame, bbox):
#     normVals = np.full(len(bbox), frame.shape[0])
#     normVals[::2] = frame.shape[1]
#     return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)


# --------------- Methods ---------------
def plot_box_xyxy(frame, box, color, text):
    plot_box_yxyx(frame, np.array(
        (box[1], box[0], box[3], box[2])), color, text)


def plot_box_yxyx(frame, box, color, text):
    y1 = (frame.shape[0] * box[0]).astype(int)
    y2 = (frame.shape[0] * box[2]).astype(int)
    x1 = (frame.shape[1] * box[1]).astype(int)
    x2 = (frame.shape[1] * box[3]).astype(int)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.rectangle(frame, (x1, y1), (x1 + len(text)*11, y1 + 15), color, -1)
    cv2.putText(frame, text, (x1 + 10, y1 + 10),
                cv2.FONT_HERSHEY_TRIPLEX, 0.4, COLOR_WHITE)


def is_box_valid(box):
    area = (box[BX2]-box[BX1]) * (box[BY2]-box[BY1])
    is_too_small = area < MIN_AREA
    is_too_large = area > MAX_AREA
    return is_too_small, is_too_large, area


def is_ready_to_pickup(box):
    return box[BY1] < DETECT_PICKUP_END_Y and box[BY1] > DETECT_PICKUP_START_Y


def draw_threshold(frame,threshold_y,text):
        start = to_uncrop_cordinates((0, threshold_y))
        end = to_uncrop_cordinates((1, threshold_y))
        frame_width = int(frame.shape[0])
        frame_height = int(frame.shape[1])
        x1 = int(frame_width*start[0])
        y1 = int(frame_height*start[1])
        x2 = int(frame_width*end[0])
        y2 = int(frame_height*end[1])        
        color = COLOR_CYAN
        cv2.rectangle(frame, (x1,y1),(x2,y2), color, 2)
        cv2.rectangle(frame, (x1, y1), (x1 + len(text)*11, y1 + 15), color, -1)
        cv2.putText(frame, text, (x1+10, y1 +10), cv2.FONT_HERSHEY_TRIPLEX, 0.4, COLOR_BLACK)


def draw_fps(frame,fps):        
        # show fps and predicted count
        color_black, color_white = (0, 0, 0), (255, 255, 255)
        label_fps = "Fps: {:.2f}".format(fps)
        (w1, h1), _ = cv2.getTextSize(label_fps, cv2.FONT_HERSHEY_TRIPLEX, 0.4, 1)
        cv2.rectangle(
            frame, (0, frame.shape[0] - h1 - 6), (w1 + 2, frame.shape[0]), color_white, -1)
        cv2.putText(frame, label_fps, (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX,
                    0.4, color_black)

def plot_boxes(frame, boxes, colors, scores, to_uncrop=False):
    for i in range(boxes.shape[0]):
        box = boxes[i]
        # Discard very large matches (like a hand, or the whole area)
        is_too_small, is_too_large, area = is_box_valid(box)
        if (is_too_large and area > 0.8):
            # print('Skipping box as too large/small. Area:{} Frame:{}'.format(area,frame_count) )
            break

        color = COLOR_GRAY if not is_ready_to_pickup(box) else (
            COLOR_RED if (is_too_small or is_too_large) else COLOR_GREEN)
        # color = (int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2]))

        # color =
        if (to_uncrop):
            box = to_uncrop_cordinates_box(box)
        score = f"{scores[i]:.2f}"
        plot_box_yxyx(frame, box, color, score)


work_que = Queue(maxsize=10)
live_detections = []

def process_work(frame_count):
    #remove any detection that has its work completed
    def is_set(item):
        _,_,_,_,_,completed_evt = item #generalise this!
        return completed_evt.is_set()

    completed_work = next(filter(is_set, live_detections), None)
    if (completed_work != None):
        _, work_item_x, work_item_width, work_timestamp, _, _ = completed_work
        live_detections.remove(completed_work)
        # print('D << Removing completed detection. x:{0:>4.3} width:{1:>4.3} timestamp:{2} time:{3:>4.3}'.format(
            # work_item_x, work_item_width, work_timestamp, work_timestamp-start_time))
        # print('D    Detection Len {0}'.format(len(live_detections)))

    if (work_que.empty() and len(live_detections)) > 0:
        # TODO - get the oldest item, rather than the first
        next_work = live_detections[0]
        _, work_item_x, work_item_width, work_timestamp, is_in_progress, _ = next_work
        if (not is_in_progress.is_set()):
            # print('Q >> Adding work to queue. x:{0:>4.3} width:{1:>4.3} timestamp:{2} time:{3:>4.3}'.format(
                # work_item_x, work_item_width, work_timestamp, work_timestamp-start_time))
            work_que.put(next_work, block=False)

def process_detection(boxes, frame_count):
    # if we have a new box, add it to the work queue once it passes the crossing line
    for i in range(boxes.shape[0]):
        box = boxes[i]
        is_too_small, is_too_large, _ = is_box_valid(box)
        if (not is_too_small and not is_too_large and is_ready_to_pickup(box)):
            item_x = box[BX1]
            item_width = box[BX2]-box[BX1]
            timestamp = time.time()

            # print('Found something to pickup. Frame:{0} x:{1:>4.3} width:{2:>4.3}'.format(frame_count,itemX,itemWidth))
            
            # print('>> Adding work to queue. Frame:{0} x:{1:>4.3} width:{2:>4.3} timestamp:{3}'.format(
                # frame_count, item_x, item_width, timestamp))
            def approx_equal(a,b,max_diff_percent):
                return (abs(a-b)/b) < max_diff_percent

            def matches_live_detection(work):
                _, prev_item_x, prev_item_width, prev_timestamp, _, _ = work  # generalise this!
                # see if box starts within 5% of prev
                if (approx_equal(item_x, prev_item_x, 0.1)):
                    # print('D   :x matched previous, so skipping')
                    return True
                # see if box center is within 5% of prev
                center = 0.0 + item_x + item_width/2
                prev_center = 0.0 + prev_item_x + prev_item_width/2
                if (approx_equal(center, prev_center, 0.1)):
                    print('D   :center matched previous, so skipping')
                    return True

                # does not match, so must be new
                return False

            if (next(filter(matches_live_detection, live_detections), None) == None):
                # new item, so add it to the live_work
                in_progress_evt = Event()
                completed_evt = Event()
                work = (PICKUP_WORK, item_x, item_width, timestamp, in_progress_evt, completed_evt)
                live_detections.append(work)
                print('D >> Adding detection to live_detections. frame:{0} x:{1:>4.3} width:{2:>4.3} timestamp:{3}'.format(frame_count, item_x, item_width, timestamp))
                print('D    Detections Len {0}'.format(len(live_detections)))


# maps normalized cordinates to the pre-cropped cordinates
# bbox uses yxyx vs uncropped which is xyxy
# the return value is yxyx
def to_uncrop_cordinates(point):
    return (
        CROP_BOX[CX1] + (point[0] * (CROP_BOX[CX2] - CROP_BOX[CX1])),
        CROP_BOX[CY1] + (point[1] * (CROP_BOX[CY2] - CROP_BOX[CY1]))
    )


def to_uncrop_cordinates_box(box):
    remapped_bbox = np.array([0.0, 0, 0, 0])
    remapped_bbox[BX1] = CROP_BOX[CX1] + \
        (box[BX1] * (CROP_BOX[CX2] - CROP_BOX[CX1]))
    remapped_bbox[BY1] = CROP_BOX[CY1] + \
        (box[BY1] * (CROP_BOX[CY2] - CROP_BOX[CY1]))
    remapped_bbox[BX2] = CROP_BOX[CX1] + \
        (box[BX2] * (CROP_BOX[CX2] - CROP_BOX[CX1]))
    remapped_bbox[BY2] = CROP_BOX[CY1] + \
        (box[BY2] * (CROP_BOX[CY2] - CROP_BOX[CY1]))
    return remapped_bbox


print('Starting work queue processer thread')
worker_abort = Event()
deltaArmThread = Thread(target=process_work_queue,
                        args=(work_que, worker_abort))
deltaArmThread.start()

# def plot_boxes(frame, boxes, colors, scores):
#     color_black = (0, 0, 0)
#     for i in range(boxes.shape[0]):
#         box = boxes[i]
#         y1 = (frame.shape[0] * box[0]).astype(int)
#         y2 = (frame.shape[0] * box[2]).astype(int)
#         x1 = (frame.shape[1] * box[1]).astype(int)
#         x2 = (frame.shape[1] * box[3]).astype(int)
#         color = (int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2]))

#         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#         cv2.rectangle(frame, (x1, y1), (x1 + 50, y1 + 15), color, -1)
#         cv2.putText(frame, f"{scores[i]:.2f}", (x1 + 10, y1 + 10), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color_black)

# --------------- Pipeline ---------------
pipeline = dai.Pipeline()

pipeline.setOpenVINOVersion(version=dai.OpenVINO.VERSION_2021_4)

# Define a neural network that will make predictions based on the source frames
detection_nn = pipeline.create(dai.node.NeuralNetwork)
detection_nn.setBlobPath(NN_PATH)
detection_nn.setNumPoolFrames(4)
detection_nn.input.setBlocking(False)
detection_nn.setNumInferenceThreads(2)

# Color camera
cam = pipeline.create(dai.node.ColorCamera)
cam.setPreviewKeepAspectRatio(True)
cam.setPreviewSize(PREVIEW_WIDTH, PREVIEW_HEIGHT)
cam.setInterleaved(False)
cam.setFps(30)
cam.initialControl.setAntiBandingMode(dai.CameraControl.AntiBandingMode.AUTO)
cam.initialControl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.AUTO)
# cam.initialControl.setAutoFocusTrigger()
cam.initialControl.setAutoExposureEnable()

# Create manip
manip = pipeline.create(dai.node.ImageManip)
manip.initialConfig.setCropRect(*CROP_BOX)
manip.initialConfig.setResize(NN_WIDTH, NN_HEIGHT)
manip.initialConfig.setFrameType(dai.ImgFrame.Type.RGB888p)
manip.initialConfig.setKeepAspectRatio(True)

# Link preview to manip and manip to nn
cam.preview.link(manip.inputImage)

manip.out.link(detection_nn.input)

# Create inputs
xin_control = pipeline.create(dai.node.XLinkIn)
xin_control.setStreamName('control')
xin_control.out.link(cam.inputControl)

# Create outputs
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("cam")
xout_rgb.input.setBlocking(False)
cam.preview.link(xout_rgb.input)

xout_nn = pipeline.create(dai.node.XLinkOut)
xout_nn.setStreamName("nn")
xout_nn.input.setBlocking(False)
detection_nn.out.link(xout_nn.input)

xout_manip = pipeline.create(dai.node.XLinkOut)
xout_manip.setStreamName("manip")
xout_manip.input.setBlocking(False)
manip.out.link(xout_manip.input)

# Pipeline defined, now the device is assigned and pipeline is started
with dai.Device(pipeline) as device:
    controlQueue = device.getInputQueue('control')

    np.random.seed(0)
    colors_full = np.random.randint(255, size=(100, 3), dtype=int)

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    q_cam = device.getOutputQueue(name="cam", maxSize=4, blocking=False)
    q_manip = device.getOutputQueue(name="manip", maxSize=4, blocking=False)
    q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    start_time = time.time()
    frame_count = 0
    counter = 0
    fps = 0
    layer_info_printed = False

    while True:
        in_cam = q_cam.get()
        in_nn = q_nn.get()
        in_manip = q_manip.get()

        frame = in_cam.getCvFrame()
        frame_manip = in_manip.getCvFrame()

        # remap the colors here
        frame_manip = cv2.cvtColor(frame_manip, cv2.COLOR_RGB2BGR)

        # get outputs
        detection_boxes = np.array(
            in_nn.getLayerFp16("ExpandDims")).reshape((100, 4))
        detection_scores = np.array(
            in_nn.getLayerFp16("ExpandDims_2")).reshape((100,))

        # keep boxes bigger than threshold
        mask = detection_scores >= THRESHOLD
        boxes = detection_boxes[mask]
        colors = colors_full[mask]
        scores = detection_scores[mask]

        process_detection(boxes, frame_count)
        process_work(frame_count)

        # drawing time
        draw_threshold(frame, DETECT_PICKUP_START_Y, 'end')
        draw_threshold(frame, DETECT_PICKUP_END_Y, 'start')
        plot_box_xyxy(frame, CROP_BOX_ARR, COLOR_BLUE, 'Detect Zone')
        draw_fps(frame, fps)

        # draw boxes
        plot_boxes(frame, boxes, colors, scores, True)
        plot_boxes(frame_manip, boxes, colors, scores, False)

        # show frame
        cv2.imshow("Full Preview", frame)
        cv2.imshow("Detect NN", frame_manip)

        frame_count += 1
        counter += 1
        if (time.time() - start_time) > 1:
            fps = counter / (time.time() - start_time)

            counter = 0
            start_time = time.time()

        key = cv2.waitKey(1)
        if key == ord('r'):
            # Press 'r' to request arm to goto reference point
            work = (GOTO_REF_READY_WORK, 0, 0, 0, Event(), Event())
            work_que.put(work, block=False)

        if key == ord('o'):            
            work = (OPEN_HAND_WORK, 0, 0, 0, Event(), Event())
            work_que.put(work, block=False)

        if key == ord('c'):            
            work = (CLOSE_HAND_WORK, 0, 0, 0, Event(), Event())
            work_que.put(work, block=False)

        if key == ord('p'):
            # Press 'r' to request arm to goto reference point
            work = (GOTO_REF_PICKUP_WORK, 0, 0, 0, Event(), Event())
            work_que.put(work, block=False)

        if key == ord('d'):        
            work = (GOTO_REF_DROP_WORK, 0, 0, 0, Event(), Event())
            work_que.put(work, block=False)

        if key == ord('e'):            
            work = (TOGGLE_PICKUP_ENABLED, 0, 0, 0, Event(), Event())
            work_que.put(work, block=False)

        if key == ord('a'):
            # Press 'a' to send autofocus
            camControl = dai.CameraControl()
            camControl.setAutoFocusTrigger()
            controlQueue.send(camControl)

        if key == ord('z'):
            # Press 'z' to live_detection 
            live_detections.clear()
            print('Clearing all live detections')

        if key == ord('q'):
            break


worker_abort.set()
deltaArmThread.join()