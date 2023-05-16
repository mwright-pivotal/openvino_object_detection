to run locally:
1. create a docker network named "inferencing"
2. startup rabbitmq
```docker run -p 5672:5672 -p 15672:15672 --name tanzu-messaging --network inferencing -e RABBITMQ_DEFAULT_USER=user -d rabbitmq:3.10.22-management```

3. enable streams plugin
```docker exec -it tanzu-messaging rabbitmq-plugins enable rabbitmq_stream```

4. run container
```docker run -e VIDEO_INPUT=starwars-sample.mp4 -e ACCELERATION_DEVICE=GPU -e INFERENCING_MODEL=custom_models/saved_model.xml -e LABELS_FILE=starwars_labels.txt -e AMQP_HOSTNAME=tanzu-messaging --network inferencing --device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render*) object-detection```
