# -*- coding: utf-8 -*-
import log
import time
import torch
import base64
import numpy as np
from PIL import Image
from io import BytesIO
import torch.nn.functional as F
from collections import OrderedDict
from torch.autograd import Variable
from networks.Resnet18Unet import ResNet18Unet, ResNet34Unet
import torchvision.transforms as transforms
from metric.metrics_manager import MetricsManager
from model_service.pytorch_model_service import PTServingBaseService
Image.MAX_IMAGE_PIXELS = 1000000000000000
logger = log.getLogger(__name__)


class ImageClassificationService(PTServingBaseService):
    def __init__(self, model_name, model_path):
        self.model_name = model_name
        self.model_path = model_path

        self.model = ResNet34Unet(pretrained=False)
        self.use_cuda = False
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print('Using GPU for inference')
            self.use_cuda = True
            checkpoint = torch.load(self.model_path)
            self.model = self.model.to(device)
            self.model.load_state_dict(checkpoint)
        else:
            print('Using CPU for inference')
            checkpoint = torch.load(self.model_path, map_location='cpu')
            self.model = self.model.to(device)
            self.model.load_state_dict(checkpoint)

        self.model.eval()

    def _preprocess(self, data):
        preprocessed_data = {}
        for k, v in data.items():
            for file_name, file_content in v.items():
                img = Image.open(file_content)
                img = np.array(img)
                preprocessed_data[k] = img
        return preprocessed_data

    def _inference(self, data):
        target_l = 256
        mean = [0.486, 0.459, 0.408]
        std = [0.229, 0.224, 0.225]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        img = data["input_img"]
        img_h, img_w, img_c = img.shape
        h_num = (img_h // target_l + 1) if img_h % target_l else img_h // target_l
        w_num = (img_w // target_l + 1) if img_w % target_l else img_w // target_l
        pad_shape = (h_num * target_l, w_num * target_l, 3)
        pad_img = np.zeros(pad_shape)
        pad_img[0: img_h, 0: img_w, :] = img
        if pad_img.max() > 1:
            pad_img = pad_img / 255.0 * 3.2 -1.6
        pad_img = pad_img.transpose(2, 0, 1)
        
        label = np.zeros((pad_shape[0], pad_shape[1]))
        for i in range(h_num):
            for j in range(w_num):
                y1 = i * target_l
                y2 = min((i + 1) * target_l, pad_shape[0])
                x1 = j * target_l
                x2 = min((j + 1) * target_l, pad_shape[1])
                img = pad_img[:, y1: y2, x1: x2]
                img = img[np.newaxis, :, :, :].astype(np.float32)
                img = torch.from_numpy(img)
                img = Variable(img.to(device))
                with torch.no_grad():
                    out_l = self.model(img)
                    #out_l = torch.sigmoid(out_l)
                out_l = out_l.cpu().data.numpy()
                out_l[out_l >= 0.5] = 1
                out_l[out_l < 0.5] = 0
                label[y1: y2, x1: x2] = out_l.astype(np.int8)
        label = label[0: img_h, 0: img_w]
        _label = label.astype(np.int8).tolist()
        _len, __len = len(_label), len(_label[0])
        o_stack = []
        for _ in _label:
            out_s = {"s": [], "e": []}
            j = 0
            while j < __len:
                if _[j] == 0:
                    out_s["s"].append(str(j))
                    while j < __len and _[j] == 0:
                        j += 1
                    out_s["e"].append(str(j))
                j += 1
            o_stack.append(out_s)
        result = {"result": o_stack}
        return result

    def _postprocess(self, data):
        return data

    def inference(self, data):
        pre_start_time = time.time()
        data = self._preprocess(data)
        infer_start_time = time.time()
        # Update preprocess latency metric
        pre_time_in_ms = (infer_start_time - pre_start_time) * 1000
        logger.info('preprocess time: ' + str(pre_time_in_ms) + 'ms')
        if self.model_name + '_LatencyPreprocess' in MetricsManager.metrics:
            MetricsManager.metrics[self.model_name + '_LatencyPreprocess'].update(pre_time_in_ms)
        data = self._inference(data)
        infer_end_time = time.time()
        infer_in_ms = (infer_end_time - infer_start_time) * 1000
        logger.info('infer time: ' + str(infer_in_ms) + 'ms')
        data = self._postprocess(data)
        # Update inference latency metric
        post_time_in_ms = (time.time() - infer_end_time) * 1000
        logger.info('postprocess time: ' + str(post_time_in_ms) + 'ms')
        if self.model_name + '_LatencyInference' in MetricsManager.metrics:
            MetricsManager.metrics[self.model_name + '_LatencyInference'].update(post_time_in_ms)
        # Update overall latency metric
        if self.model_name + '_LatencyOverall' in MetricsManager.metrics:
            MetricsManager.metrics[self.model_name + '_LatencyOverall'].update(pre_time_in_ms + post_time_in_ms)
        logger.info('latency: ' + str(pre_time_in_ms + infer_in_ms + post_time_in_ms) + 'ms')
        data['latency_time'] = pre_time_in_ms + infer_in_ms + post_time_in_ms
        return data

