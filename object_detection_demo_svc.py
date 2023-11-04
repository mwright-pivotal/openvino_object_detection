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

import asyncio
import logging as log
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


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    # args.add_argument('-at', '--architecture_type', help='Required. Specify model\' architecture type.',
    #                   type=str, required=True, choices=available_model_wrappers)
    # args.add_argument('--adapter', help='Optional. Specify the model adapter. Default is openvino.',
    #                   default='openvino', type=str, choices=('openvino', 'ovms'))
    args.add_argument('-i', '--input', required=True,
                      help='Required. An input to process. The input must be a single image, '
                           'a folder of images, video file or camera id.')
    # args.add_argument('-d', '--device', default='CPU', type=str,
    #                   help='Optional. Specify the target device to infer on; CPU, GPU, HDDL or MYRIAD is '
    #                        'acceptable. The demo will look for a suitable plugin for device specified. '
    #                        'Default value is CPU.')

    common_model_args = parser.add_argument_group('Common model options')
    common_model_args.add_argument('--labels', help='Optional. Labels mapping file.', default=None, type=str)
    common_model_args.add_argument('-t', '--prob_threshold', default=0.5, type=float,
                                   help='Optional. Probability threshold for detections filtering.')
    common_model_args.add_argument('--anchors', default=None, type=float, nargs='+',
                                   help='Optional. A space separated list of anchors. '
                                        'By default used default anchors for model. Only for YOLOV4 architecture type.')
    common_model_args.add_argument('--masks', default=None, type=int, nargs='+',
                                   help='Optional. A space separated list of mask for anchors. '
                                        'By default used default masks for model. Only for YOLOV4 architecture type.')
    common_model_args.add_argument('--layout', type=str, default=None,
                                   help='Optional. Model inputs layouts. '
                                        'Ex. NCHW or input0:NCHW,input1:NC in case of more than one input.')
    common_model_args.add_argument('--num_classes', default=None, type=int,
                                   help='Optional. Number of detected classes. Only for NanoDet, NanoDetPlus '
                                        'architecture types.')

    infer_args = parser.add_argument_group('Inference options')
    infer_args.add_argument('-nireq', '--num_infer_requests', help='Optional. Number of infer requests',
                            default=0, type=int)
    infer_args.add_argument('-nstreams', '--num_streams',
                            help='Optional. Number of streams to use for inference on the CPU or/and GPU in throughput '
                                 'mode (for HETERO and MULTI device cases use format '
                                 '<device1>:<nstreams1>,<device2>:<nstreams2> or just <nstreams>).',
                            default='', type=str)
    infer_args.add_argument('-nthreads', '--num_threads', default=None, type=int,
                            help='Optional. Number of threads to use for inference on CPU (including HETERO cases).')

    io_args = parser.add_argument_group('Input/output options')
    io_args.add_argument('--loop', default=False, action='store_true',
                         help='Optional. Enable reading the input in a loop.')
    io_args.add_argument('-o', '--output', required=False,
                         help='Optional. Name of the output file(s) to save.')
    io_args.add_argument('-limit', '--output_limit', required=False, default=1000, type=int,
                         help='Optional. Number of frames to store in output. '
                              'If 0 is set, all frames are stored.')
    io_args.add_argument('--no_show', help="Optional. Don't show output.", action='store_true')
    io_args.add_argument('--output_resolution', default=None, type=resolution,
                         help='Optional. Specify the maximum output window resolution '
                              'in (width x height) format. Example: 1280x720. '
                              'Input frame size used by default.')
    io_args.add_argument('-u', '--utilization_monitors', default='', type=str,
                         help='Optional. List of monitors to show initially.')

    input_transform_args = parser.add_argument_group('Input transform options')
    input_transform_args.add_argument('--reverse_input_channels', default=False, action='store_true',
                                      help='Optional. Switch the input channels order from '
                                           'BGR to RGB.')
    input_transform_args.add_argument('--mean_values', default=None, type=float, nargs=3,
                                      help='Optional. Normalize input by subtracting the mean '
                                           'values per channel. Example: 255.0 255.0 255.0')
    input_transform_args.add_argument('--scale_values', default=None, type=float, nargs=3,
                                      help='Optional. Divide input by scale values per channel. '
                                           'Division is applied after mean values subtraction. '
                                           'Example: 255.0 255.0 255.0')

    debug_args = parser.add_argument_group('Debug options')
    debug_args.add_argument('-r', '--raw_output_message', help='Optional. Output inference results raw values showing.',
                            default=False, action='store_true')
    return parser


def draw_detections(frame, detections, palette, labels, output_transform):
    frame = output_transform.resize(frame)
    for detection in detections:
        class_id = int(detection.id)
        color = palette[class_id]
        det_label = labels[class_id] if labels and len(labels) >= class_id else '#{}'.format(class_id)
        xmin, ymin, xmax, ymax = detection.get_coords()
        xmin, ymin, xmax, ymax = output_transform.scale([xmin, ymin, xmax, ymax])
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(frame, '{} {:.1%}'.format(det_label, detection.score),
                    (xmin, ymin - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)
        if isinstance(detection, DetectionWithLandmarks):
            for landmark in detection.landmarks:
                landmark = output_transform.scale(landmark)
                cv2.circle(frame, (int(landmark[0]), int(landmark[1])), 2, (0, 255, 255), 2)
    return frame


def print_raw_results(detections, labels, frame_id):
    log.debug(' ------------------- Frame # {} ------------------ '.format(frame_id))
    log.debug(' Class ID | Confidence | XMIN | YMIN | XMAX | YMAX ')
    for detection in detections:
        xmin, ymin, xmax, ymax = detection.get_coords()
        class_id = int(detection.id)
        det_label = labels[class_id] if labels and len(labels) >= class_id else '#{}'.format(class_id)
        log.debug('{:^9} | {:10f} | {:4} | {:4} | {:4} | {:4} '
                  .format(det_label, detection.score, xmin, ymin, xmax, ymax))


async def frame_processors(worker_id, conn, queue, write_queue):
    """ Pulls images from the queue, sends the frame to Wallaroo for inference, and then
        processes the results. Create multiple workers to keep multiple submissions in the
        inference queue at once to improve throughput."""
    print(f"Worker {worker_id} starting")
    params = wallaroo.get_dataset_params(dataset=["out", "time", "metadata"])
    while True:
        print(f"Worker {worker_id} waiting")
        idx, frame = await queue.get()
        print(f"Worker {worker_id} got input: {idx} {frame.shape}")
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
        table = pa.Table.from_pydict({"data": frame})

        # Run the inference. Data adds metadata for performance, drops "in" so we don't get the image back
        results = await wallaroo.async_infer(conn, table, dataset_params=params, timeout=30)
        # print(f"Results {worker_id}: {idx} {results}")
        await write_queue.put((idx, orig_frame, results))

        # Notify the queue that the "work item" has been processed.
        queue.task_done()
        print(f"{worker_id}: finished")


async def frame_writer(dest, queue):
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
            results = await queue.get()
            if not results:
                queue.task_done()
                break
        idx, frame, results = results
        print(f"Writer: received frame {idx}")

        if idx > next_frame_id:
            out_of_order_queue[idx] = (idx, frame, results)
            continue

        # Render the detections on the original frame
        #    For MobileNet model:
        #       out.2518: confidences
        #       out.2519: detection classes
        #       out.output: bounding boxes
        print(f"Writer: rendering frame {idx} {len(results['out.output'][0])} {len(results['out.2518'][0])}")
        confidences = results["out.2518"][0]
        classes = results["out.2519"][0]
        bboxes = results["out.output"][0]
        
        objects = 0
        for bidx in range(len(classes)):
            if confidences[bidx].as_py() < 0.5:
                continue
            objects += 1
            class_id = int(classes[bidx].as_py())
            xmin = int(bboxes[bidx].as_py())
            ymin = int(bboxes[bidx+1].as_py())
            xmax = int(bboxes[bidx+2].as_py())
            ymax = int(bboxes[bidx+3].as_py())
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            cv2.putText(frame, f"{class_id} {confidences[bidx].as_py():.2f}",
                        (xmin, ymin - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 0), 1)
        # Writing image to file
        print(f"Writer: writing frame {idx}, {objects} boxes")
        if objects > 0:
            cv2.imwrite(f"img-{idx}.png", frame)
        queue.task_done()
        next_frame_id += 1
        


async def main():
    args = build_argparser().parse_args()
    cap = open_images_capture(args.input, args.loop)
    configuration = {
        # 'resize_type': args.resize_type,
        # 'mean_values': args.mean_values,
        # 'scale_values': args.scale_values,
        # 'reverse_input_channels': args.reverse_input_channels,
        'path_to_labels': args.labels,
        'confidence_threshold': args.prob_threshold,
        # 'input_size': args.input_size, # The CTPN specific
        'num_classes': args.num_classes, # The NanoDet and NanoDetPlus specific
    }

    
    next_frame_id = 0
    next_frame_id_to_show = 0

    presenter = None
    output_transform = None
    video_writer = cv2.VideoWriter()

    # Max number of frames to keep in memory. We want enough that we can keep the inference server busy
    WORKER_COUNT = 3
    conn = wallaroo.connect("http://localhost:8081/pipelines/bench-mobilenet-pipeline")
    work_queue = asyncio.Queue(WORKER_COUNT*6)
    write_queue = asyncio.Queue()
    workers = [asyncio.create_task(frame_processors(i, conn, work_queue, write_queue)) for i in range(WORKER_COUNT)]
    workers.append(asyncio.create_task(frame_writer(args.output, write_queue)))

    while True:
        # if detector_pipeline.callback_exceptions:
        #     raise detector_pipeline.callback_exceptions[0]
        # # Process all completed requests
        # results = detector_pipeline.get_result(next_frame_id_to_show)
        # if results:
        #     objects, frame_meta = results
        #     frame = frame_meta['frame']
        #     start_time = frame_meta['start_time']

        #     if len(objects) and args.raw_output_message:
        #         print_raw_results(objects, model.labels, next_frame_id_to_show)

        #     presenter.drawGraphs(frame)
        #     rendering_start_time = perf_counter()
        #     frame = draw_detections(frame, objects, palette, model.labels, output_transform)
        #     render_metrics.update(rendering_start_time)
        #     metrics.update(start_time, frame)

        #     if video_writer.isOpened() and (args.output_limit <= 0 or next_frame_id_to_show <= args.output_limit-1):
        #         video_writer.write(frame)
        #     next_frame_id_to_show += 1

        #     if not args.no_show:
        #         cv2.imshow('Detection Results', frame)
        #         key = cv2.waitKey(1)

        #         ESC_KEY = 27
        #         # Quit.
        #         if key in {ord('q'), ord('Q'), ESC_KEY}:
        #             break
        #         presenter.handleKey(key)
        #     continue

        # Get new image/frame
        start_time = perf_counter()
        frame = cap.read()
        if frame is None:
            if next_frame_id == 0:
                raise ValueError("Can't read an image from the input")
            for _ in range(WORKER_COUNT):
                await work_queue.put((-1, None))
            break

        img = Image.fromarray(frame)
        img = img.resize((640, 480))
        # img.save(f"test-{next_frame_id}.jpg")
        frame = np.array(img)
        print(f"That's cap: {next_frame_id} {frame.shape}")
        await work_queue.put((next_frame_id, frame))

        next_frame_id += 1

    write_queue.put(None)

    await work_queue.join()
    await write_queue.join()
    await asyncio.gather(workers, return_exceptions=True)



    detector_pipeline.await_all()
    if detector_pipeline.callback_exceptions:
        raise detector_pipeline.callback_exceptions[0]
    # Process completed requests
    for next_frame_id_to_show in range(next_frame_id_to_show, next_frame_id):
        results = detector_pipeline.get_result(next_frame_id_to_show)
        objects, frame_meta = results
        frame = frame_meta['frame']
        start_time = frame_meta['start_time']

        if len(objects) and args.raw_output_message:
            print_raw_results(objects, model.labels, next_frame_id_to_show)

        presenter.drawGraphs(frame)
        rendering_start_time = perf_counter()
        frame = draw_detections(frame, objects, palette, model.labels, output_transform)
        render_metrics.update(rendering_start_time)
        metrics.update(start_time, frame)

        if video_writer.isOpened() and (args.output_limit <= 0 or next_frame_id_to_show <= args.output_limit-1):
            video_writer.write(frame)

        if not args.no_show:
            cv2.imshow('Detection Results', frame)
            key = cv2.waitKey(1)

            ESC_KEY = 27
            # Quit.
            if key in {ord('q'), ord('Q'), ESC_KEY}:
                break
            presenter.handleKey(key)

    metrics.log_total()
    log_latency_per_stage(cap.reader_metrics.get_latency(),
                          detector_pipeline.preprocess_metrics.get_latency(),
                          detector_pipeline.inference_metrics.get_latency(),
                          detector_pipeline.postprocess_metrics.get_latency(),
                          render_metrics.get_latency())
    for rep in presenter.reportMeans():
        log.info(rep)


if __name__ == '__main__':
    asyncio.run(main())
    sys.exit(0)
