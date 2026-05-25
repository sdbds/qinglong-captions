from __future__ import annotations

import concurrent.futures

import cv2
import numpy as np
from PIL import Image

from module.wdtagger import constants
from utils.console_util import print_exception
from utils.wdtagger_siglip2 import Siglip2InferenceContext, process_siglip2_batch


def preprocess_image(image, is_cl_tagger=False):
    image = np.array(image)
    image = image[:, :, ::-1]

    h, w = image.shape[:2]
    size = max(h, w)
    pad_y, pad_x = size - h, size - w
    pad_t, pad_l = pad_y // 2, pad_x // 2
    pad_b, pad_r = pad_y - pad_t, pad_x - pad_l

    use_gpu = cv2.cuda.getCudaEnabledDeviceCount() > 0

    if use_gpu and size > 1024:
        gpu_image = cv2.cuda_GpuMat()
        gpu_image.upload(image)
        gpu_image = cv2.cuda.copyMakeBorder(
            gpu_image,
            pad_t,
            pad_b,
            pad_l,
            pad_r,
            cv2.BORDER_CONSTANT,
            value=[255, 255, 255],
        )

        if size > constants.IMAGE_SIZE:
            gpu_image = cv2.cuda.resize(gpu_image, (constants.IMAGE_SIZE, constants.IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        else:
            image = Image.fromarray(image)
            image = image.resize((constants.IMAGE_SIZE, constants.IMAGE_SIZE), Image.LANCZOS)
            image = np.array(image)

        image = gpu_image.download()
    else:
        image = cv2.copyMakeBorder(
            image,
            pad_t,
            pad_b,
            pad_l,
            pad_r,
            cv2.BORDER_CONSTANT,
            value=[255, 255, 255],
        )

        if size > constants.IMAGE_SIZE:
            image = cv2.resize(image, (constants.IMAGE_SIZE, constants.IMAGE_SIZE), cv2.INTER_AREA)
        else:
            image = Image.fromarray(image)
            image = image.resize((constants.IMAGE_SIZE, constants.IMAGE_SIZE), Image.LANCZOS)
            image = np.array(image)

    if is_cl_tagger:
        image = image.transpose(2, 0, 1)
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
        image = (image - mean) / std
    else:
        image = image.astype(np.float32)

    return image


def load_and_preprocess_batch(uris, is_cl_tagger=False):
    def load_single_image(uri):
        try:
            return preprocess_image(Image.open(uri).convert("RGB"), is_cl_tagger)
        except Exception as e:
            print_exception(constants.console, e, prefix=f"Error processing {uri}")
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        batch_images = list(executor.map(load_single_image, uris))

    valid_images = [(i, img) for i, img in enumerate(batch_images) if img is not None]
    return [img for _, img in valid_images]


def load_siglip2_rgb_batch(uris):
    def load_single_image(uri):
        try:
            with Image.open(uri) as image:
                return str(uri), image.convert("RGB")
        except Exception as e:
            print_exception(constants.console, e, prefix=f"Error processing {uri}")
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        loaded = list(executor.map(load_single_image, uris))

    valid_pairs = [item for item in loaded if item is not None]
    return [uri for uri, _ in valid_pairs], [image for _, image in valid_pairs]


def process_batch(images, session, input_name):
    try:
        if isinstance(input_name, Siglip2InferenceContext):
            return process_siglip2_batch(images, session, input_name)

        batch_data = np.ascontiguousarray(np.stack(images))
        outputs = session.run(None, {input_name: batch_data})

        def stable_sigmoid(x):
            return 1 / (1 + np.exp(-np.clip(x, -30, 30)))

        return stable_sigmoid(outputs[0])
    except Exception as e:
        print_exception(constants.console, e, prefix="Batch processing error")
        return None
