This is a modfied version of one of Intel's sample apps for using their Openvino framework to do computer vision/object detection AI with help of GPU.

Orignal README here: https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/object_detection_demo/python/README.md

This extends the use of Intel's Openvino demo to the ability of using Cloud Native Buildpacks to generate the container that has all the necessary media processing libraries, Intel's Openvino Framework, and ability to leverage GPU hardware.

Prior to building the app, you need a custom builder as explained in this repo: https://github.com/mwright-pivotal/openvino-buildpack
Then building the app is as simple as:

```pack build object-detection  --builder openvino-builder:jammy```

to run locally:
1. create a docker network named "inferencing"
2. startup rabbitmq
```docker run -p 5672:5672 -p 15672:15672 --name tanzu-messaging --network inferencing -e RABBITMQ_DEFAULT_USER=user -d rabbitmq:3.10.22-management```

3. enable streams plugin
```docker exec -it tanzu-messaging rabbitmq-plugins enable rabbitmq_stream```

4. run container
```docker run -e VIDEO_INPUT=starwars-sample.mp4 -e ACCELERATION_DEVICE=GPU -e INFERENCING_MODEL=custom_models/saved_model.xml -e LABELS_FILE=starwars_labels.txt -e AMQP_HOSTNAME=tanzu-messaging --network inferencing --device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render*) object-detection```
