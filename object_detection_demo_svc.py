#!/usr/bin/env python3
"""

Original bits of this code:
 Copyright (C) 2018-2023 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

"""
Example usage:
python3 object_detection_demo_svc.py -i person-bicycle-car-detection.mp4 -u http://localhost:8080/pipelines/bench-mobilenet-pipeline --output out.mp4
"""
import asyncio
import logging as log
import pickle
import sys
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path
from time import perf_counter

sys.path.append('common/python')
sys.path.append('common/python/openvino/model_zoo')

import wallaroo_edge as wallaroo
import httpx
import cv2
import monitors
import numpy as np
import pyarrow as pa
from PIL import Image
from images_capture import open_images_capture
from helpers import resolution, log_latency_per_stage
from visualizers import ColorPalette

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)



ModelConfigs = {
    "mobilenet": { "tensor": "data", "width" : 640, "height": 480, "classes" : "out.2519", "confidences" : "out.2518" },
    "frcnn": { "tensor": "data", "width" : 640, "height": 480, "classes" : "out.3070", "confidences" : "out.3069" },
    "yolov8": { "tensor": "images", "width" : 640, "height": 640, "classes" : None, "confidences" : None, "combined" : "out.output0" },
}

MODEL_CONFIG = {}
CLASS_LABELS = []


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument('-i', '--input', required=True,
                      help='Required. An input to process. The input must be a single image, '
                           'a folder of images, video file or camera id.')
    args.add_argument('-u', '--url', required=True,
                      help='Required. URL of the inference pipeline to connect to.')
    args.add_argument('--loop', default=False, action='store_true',
                         help='Optional. Enable reading the input in a loop.')
    args.add_argument('-o', '--output', required=False,
                         help='Optional. Name of the output file(s) to save.')
    args.add_argument('-c', "--model-config", help="Selects the model being used. One of 'mobilenet' or 'frcnn'.",)

    # args.add_argument('-d', '--device', default='CPU', type=str,
    #                   help='Optional. Specify the target device to infer on; CPU, GPU, HDDL or MYRIAD is '
    #                        'acceptable. The demo will look for a suitable plugin for device specified. '
    #                        'Default value is CPU.')



    return parser


async def frame_processors(worker_id, conn, queue, write_queue):
    """ Pulls images from the queue, sends the frame to Wallaroo for inference, and then
        processes the results. Create multiple workers to keep multiple submissions in the
        inference queue at once to improve throughput."""
    print(f"Worker {worker_id} starting")
    params = wallaroo.get_dataset_params(dataset=["out", "time", "metadata"])
    while True:
        # print(f"Worker {worker_id} waiting")
        idx, frame = await queue.get()
        if frame is None:
            print(f"Worker {worker_id}: done")
            queue.task_done()
            break

        print(f"Worker {worker_id}: got frame {idx} {frame.shape}")

        orig_frame = frame
        h, w, c = frame.shape
        if c == 3:
            # Convert from H*W*C to C*H*W
            frame = np.array([frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]], dtype=np.float32) / 256.0
        elif h == 3:
            if frame.dtype == np.int32:
                # Already organized as needed but needs to be float
                frame = np.array(frame, dtype=np.float32) / 256.0
        else:
            assert False, "Expected 3*H*W or H*W*C organiztion, got {arr.shape}}"
        frame = frame.reshape((1, c*h*w))

        frame = pa.FixedShapeTensorArray.from_numpy_ndarray(frame)
        table = pa.Table.from_pydict({MODEL_CONFIG["tensor"]: frame})

        # Run the inference. Data adds metadata for performance, drops "in" so we don't get the image back
        results = await wallaroo.async_infer(conn, table, dataset_params=params, timeout=30)
        # print(f"Results {worker_id}: {idx} {results}")
        await write_queue.put((idx, orig_frame, results))

        # Notify the queue that the "work item" has been processed.
        queue.task_done()


async def frame_writer(dest, queue, video_writer):
    """ Pulls completed frames + object detections from the queue and writes them to the video file. Frames
        are kepted ordered by frame id so that they can be written in order."""
    next_frame_id = 0
    out_of_order_queue = {}
    while True:
        if next_frame_id in out_of_order_queue:
            results = out_of_order_queue[next_frame_id]
            del out_of_order_queue[next_frame_id]
            next_frame_id += 1
        else:
            # print("Writer waiting...")
            results = await queue.get()
            if not results:
                print("Writer got finish signal")
                queue.task_done()
                break
        idx, frame, results = results
        # print(f"Writer: received frame {idx}")

        if idx > next_frame_id:
            out_of_order_queue[idx] = (idx, frame, results)
            continue

        # Render the detections on the original frame. Different models have different output styles.
        if MODEL_CONFIG["classes"]:
            objects, frame = render_resnet(results, frame)
        elif MODEL_CONFIG["combined"]:
            objects, frame = render_yolo(results, frame)

        # Writing image to file
        print(f"Writer: writing frame {idx}, {objects} boxes")
        if video_writer.isOpened():
            video_writer.write(frame)

        # if objects > 0:
        #     cv2.imwrite(f"img-{idx}.png", frame)
        queue.task_done()
        next_frame_id += 1
    print("frame_writer done")

def render_yolo(results, frame):
    """ Renders the detected objects into the frame based on results from a Yolo-style model

        Returns the number of object detected"""

    combined = results[MODEL_CONFIG["combined"]][0]
    arr = np.array(combined.as_py()).reshape((84, 8400))
    objects = 0
    for i in range(8400):
        row = arr[:, i]
        idx = np.argmax(row[4:])+4
        maxconf = row[idx]
        if maxconf > 0.5:
            # YoLo format: centroid X, centroid Y, width, height, class 1, class 2, ...
            cenx = int(row[0])
            ceny = int(row[1])
            width = int(row[2]/2)
            height = int(row[3]/2)
            cv2.rectangle(frame, (cenx - width, ceny - height), (cenx + width, ceny + height), (255, 0, 0), 2)
            cv2.putText(frame, f"{idx} {maxconf:.2f}",
                            (cenx, ceny - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 0), 1)

            objects += 1
    return objects, frame

def render_resnet(results, frame):
    """ Renders results into the frame using output from a ResNet, MobileNet, or similar model

        Returns the number of objects detected. """
    confidences = results[MODEL_CONFIG["confidences"]][0]
    classes = results[MODEL_CONFIG["classes"]][0]
    bboxes = results["out.output"][0]

    objects = 0
    for bidx in range(len(classes)):
        if confidences[bidx].as_py() < 0.5:
            continue
        objects += 1

            # Draw the bounding box around the detected object.
            # TODO: No labels array yet so just object index
        class_id = int(classes[bidx].as_py())
        label = CLASS_LABELS[class_id] if CLASS_LABELS and len(CLASS_LABELS) >= class_id else f"#{class_id}"
        xmin = int(bboxes[bidx].as_py())
        ymin = int(bboxes[bidx+1].as_py())
        xmax = int(bboxes[bidx+2].as_py())
        ymax = int(bboxes[bidx+3].as_py())
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        cv2.putText(frame, f"{label} {confidences[bidx].as_py():.2f}",
                        (xmin, ymin - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 0), 1)

    return objects, frame



async def main():
    args = build_argparser().parse_args()
    cap = open_images_capture(args.input, args.loop)

    global MODEL_CONFIG
    MODEL_CONFIG = ModelConfigs.get(args.model_config or "mobilenet")
    assert MODEL_CONFIG, f"Unknown model config '{args.model_config}', must be one of: {', '.join(ModelConfigs.keys())}"

    # Create the class labels
    try:
        global CLASS_LABELS
        CLASS_LABELS = pickle.loads(open("coco_classes.pickle", "rb").read())
    except:
        print("Unable to read class labels, skipping")

    next_frame_id = 0
    next_frame_id_to_show = 0

    presenter = None
    output_transform = None
    video_writer = cv2.VideoWriter()
    if args.output:
        video_writer.open(args.output, cv2.VideoWriter_fourcc(*'MJPG'), cap.fps(), (MODEL_CONFIG["width"], MODEL_CONFIG["height"]))

    # Max number of frames to keep in memory. We want enough that we can keep the inference server busy
    WORKER_COUNT = 2
    conn = wallaroo.connect(args.url)
    work_queue = asyncio.Queue(WORKER_COUNT*6)
    write_queue = asyncio.Queue()
    workers = [asyncio.create_task(frame_processors(i, conn, work_queue, write_queue)) for i in range(WORKER_COUNT)]
    workers.append(asyncio.create_task(frame_writer(args.output, write_queue, video_writer)))

    start_time = perf_counter()
    resize_shape = (MODEL_CONFIG["width"], MODEL_CONFIG["height"])
    while True:
        # Get new image/frame
        frame = cap.read()
        if frame is None:
            if next_frame_id == 0:
                raise ValueError("Can't read an image from the input")
            break

        img = Image.fromarray(frame)
        img = img.resize(resize_shape)
        # img.save(f"test-{next_frame_id}.jpg")
        frame = np.array(img)
        print(f"That's cap: {next_frame_id} {frame.shape}")
        await work_queue.put((next_frame_id, frame))

        next_frame_id += 1
        # if next_frame_id > 50:
        #     break

    # Signal each worker and the writer to finish up
    for i in range(WORKER_COUNT):
        print(f"Signalling worker {i} to finish")
        await work_queue.put((-1, None))
    print(f"Joining work queue")
    await work_queue.join()

    # Once all of the work queues are done we can signal the write queue to finish
    print(f"Joining write queue")
    await write_queue.put(None)
    await write_queue.join()

    print(f"Joining workers")
    await asyncio.gather(*workers, return_exceptions=True)

    end_time = perf_counter()
    print(f"Processed {next_frame_id} frames in {end_time - start_time:.2f} seconds")
    return 0


if __name__ == '__main__':
    asyncio.run(main())
    sys.exit(0)
