# Importing the necessary modules
from typing import Tuple, List
from ops import (
    letterbox,
    non_max_suppression,
    scale_boxes,
    LOGGER,
    PrePostProcessor,
)
import os
import cv2
import numpy as np
import pytpu as tpu
from tqdm.auto import tqdm
import onnxruntime as ort
import time
import click


@click.command()
@click.option(
    '--img_path',
    default='a.jpg',
    help='Input video path.',
)
@click.option(
    '--conf',
    default=0.2,
    help='Confidence NMS threshold.',
)
@click.option(
    '--iou',
    default=0.7,
    help='IoU NMS threshold.',
)
def main(
    img_path: str,
    conf: float,
    iou: float,
):
    classes = ['plane', 'helicopter']
    half = False
    imgsz = [512, 640]
    inferencer = 'ONNX'
    program_path = './tasks/000004/last.onnx'
    # program_path = './tasks/000004/compile_128x128_asic/000004_128x128_asic_b1_def_program.tpu'
    logic = PrePostProcessor(cls_names=classes, imgsz=imgsz, conf=conf, iou=iou, half=half)

    LOGGER.info(f'Init inference')
    if inferencer == 'ONNX':
        # Create an inference session using the ONNX model and specify execution providers
        session = ort.InferenceSession(program_path, providers=["CPUExecutionProvider"])
        # Get the model inputs
        model_inputs = session.get_inputs()

        # Store the shape of the input for later use
        input_shape = model_inputs[0].shape
        tick = time.perf_counter()
        # raw image in BGR format
        img = logic.load_image(img_path)
        LOGGER.info(f'loading an image {img.shape[0]}x{img.shape[1]} takes {(time.perf_counter() - tick):.4f}s')

        # Run inference using the preprocessed image data
        tick = time.perf_counter()
        proc_data = logic.preprocess(img)
        LOGGER.info(f'preproc an image from {input_shape[2]}x{input_shape[3]} to {imgsz[0]}x{imgsz[1]} takes {(time.perf_counter() - tick):.4f}s')

        tick = time.perf_counter()
        outputs = session.run(None, {model_inputs[0].name: proc_data})
        LOGGER.info(f'{inferencer} inference on image {imgsz[0]}x{imgsz[1]} takes {(time.perf_counter() - tick):.4f}s')

        tick = time.perf_counter()
        results = logic.postprocess(
            preds_boxes=outputs[0],
            orig_imgs=[img, ],
        )
        LOGGER.info(f'post proc takes {(time.perf_counter() - tick):.4f}s, has found {len(results[0]["det"])} objects')
        logic.plot(img, results[0]['det'], 'result_img.jpg')

    elif inferencer == 'IVA':
        available_tpu_devices = tpu.Device.list_devices()
        assert available_tpu_devices, 'No TPU device available! Check <ls /dev/tpu*>'

        times = []

        with tpu.Device.open(available_tpu_devices[0]) as tpu_device, \
             tpu_device.load(program_path) as tpu_program, \
             tpu_program.inference() as inference:

            # raw image in BGR format
            tick = time.perf_counter()
            img = logic.load_image(img_path)
            LOGGER.info(f'loading an image {img.shape[0]}x{img.shape[1]} takes {(time.perf_counter() - tick):.4f}s')

            # preprocessed image in BxCxHxW
            tick = time.perf_counter()
            pre_data = {'images:0': logic.preprocess(img)}
            LOGGER.info(f'preproc an image from {img.shape[0]}x{img.shape[1]} to {imgsz[0]}x{imgsz[1]} takes {(time.perf_counter() - tick):.4f}s')

            # Running the neural network on IVA TPU
            # Warming up
            for i in tqdm(range(10), desc='warm up', leave=True):
                output = inference.sync(pre_data)

            tick = time.perf_counter()
            output = inference.sync(pre_data)
            LOGGER.info(f'{inferencer} inference on image {imgsz[0]}x{imgsz[1]} takes {(time.perf_counter() - tick):.4f}s')

            tick = time.perf_counter()
            results = logic.postprocess(
                preds_boxes=output['concat_20:0'],
                orig_imgs=[img, ],
            )
            LOGGER.info(f'post proc takes {(time.perf_counter() - tick):.4f}s, has found {len(results[0]["det"])} objects')

            logic.plot(img, results[0]['det'], 'result_img.jpg')
    else:
        print('Inferencer is not supported')
        
if __name__ == '__main__':
    main()