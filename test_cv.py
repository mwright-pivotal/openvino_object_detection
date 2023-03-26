import logging as log
import sys
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path
from time import perf_counter

import cv2

sys.path.append('common/python')
sys.path.append('common/python/openvino/model_zoo')

from model_api.models import DetectionModel, DetectionWithLandmarks, RESIZE_TYPES, OutputTransform
from model_api.performance_metrics import PerformanceMetrics
from model_api.pipelines import get_user_config, AsyncPipeline
from model_api.adapters import create_core, OpenvinoAdapter, OVMSAdapter

# content of test_sample.py
def load_model(model):
    detector_pipeline = AsyncPipeline(model)
    return detector_pipeline.is_ready()

def test_answer():
    plugin_config = get_user_config('CPU', '', '')
    model_adapter = OpenvinoAdapter(create_core(), 'public/ssd300/FP16/ssd300.xml', device='CPU', plugin_config=plugin_config,
        max_num_requests=0, model_parameters = {'input_layouts': 'None'})
    
    configuration = {
        'resize_type': None,
        'mean_values': None,
        'scale_values': None,
        'reverse_input_channels': False,
        'path_to_labels': './voc_20cl_bkgr.txt',
        'confidence_threshold': 0.5,
        'input_size': 600, # The CTPN specific
        'num_classes': None, # The NanoDet and NanoDetPlus specific
    }

    model = DetectionModel.create_model('SSD', model_adapter, configuration)
    model.log_layers_info()
    assert load_model(model)
