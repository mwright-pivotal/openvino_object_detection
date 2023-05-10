This is a modfied version of one of Intel's sample apps for using their Openvino framework to do computer vision/object detection AI with help of GPU.  Inferenceing results are written to RabbitMQ streams for store and forward to datacenter/cloud

Orignal README here: https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/object_detection_demo/python/README.md

This extends the use of Intel's Openvino demo to the ability of using Cloud Native Buildpacks to generate the container that has all the necessary media processing libraries, Intel's Openvino Framework, and ability to leverage GPU hardware.

Prior to building the app, you need a custom builder as explained in this repo: https://github.com/mwright-pivotal/openvino-buildpack
Then building the app is as simple as:

```pack build object-detection  --builder openvino-builder:bionic```
