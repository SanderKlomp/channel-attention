"""Script to measure inference time of networks"""
#%%
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import tensorrt as trt
import onnx
import pycuda.autoinit # This import causes pycuda to automatically manage CUDA context creation and cleanup.
from PIL import Image
import numpy as np

from models.resnet import resnet
import common #Copied from TensorRt python samples folder

# Get test images, models and labels.
test_image = 'timing_files/ILSVRC2012_val_00002338.JPEG'
onnx_file_path = 'timing_files/test.onnx' # where the temporary onnx model will be saved for tensorrt conversion
BATCH_SIZE = 1


def time_tensorrt(model_resnet, fp16=False):
    x = torch.ones((BATCH_SIZE, 3, WIDTH, HEIGHT)).cuda() #TODO: add batch size here?
    torch.onnx.export(model_resnet, x, onnx_file_path, input_names=['input'],
                      output_names=['output'], export_params=True)

    onnx_model = onnx.load(onnx_file_path)
    onnx.checker.check_model(onnx_model)


    # logger to capture errors, warnings, and other information during the build and inference phases
    # Logger writes to stderr, so will not show errors when running in Jupyter notebook.
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    DTYPE = trt.float32 #Just for early normalization. Should always be fp32, regardless of network fp16/32 mode
    trt_runtime = trt.Runtime(TRT_LOGGER)

    def build_engine(onnx_path):
        """
        This is the function to create the TensorRT engine
        Args:
            onnx_path : Path to onnx_file.
            shape : Shape of the input of the ONNX file.
        """
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_workspace_size = (256 << 20)
            # builder.max_batch_size = BATCH_SIZE # DO NOT set this parameter, as it interferes with explicit_batch (create_network(1), the 1 means explicit batch)
            if fp16:
                builder.fp16_mode = True
            with open(onnx_path, 'rb') as model:
                if not parser.parse(model.read()):
                    print('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print (parser.get_error(error))
                    return None
            engine = builder.build_cuda_engine(network)
            return engine

    def normalize_image(image):
        # Resize, antialias and transpose the image to CHW.
        c, h, w = (3, WIDTH, HEIGHT)
        image_arr = np.asarray(image.resize((w, h), Image.ANTIALIAS)).transpose([2, 0, 1]).astype(trt.nptype(DTYPE)).ravel()
        # This particular ResNet50 model requires some preprocessing, specifically, mean normalization.
        return (image_arr / 255.0 - 0.45) / 0.225

    def load_normalized_test_case(test_image, pagelocked_buffer):
        # Normalize the image and copy to pagelocked memory.
        if BATCH_SIZE == 1:
            np.copyto(pagelocked_buffer, normalize_image(Image.open(test_image)))
        else:
            np.copyto(pagelocked_buffer, np.array(BATCH_SIZE * [normalize_image(Image.open(test_image))]).ravel())
        return test_image

    # Test building a TensorRT engine.
    # with build_engine(onnx_file_path) as engine:
    #     # Inference is the same regardless of which parser is used to build the engine, since the model architecture is the same.
    #     # Allocate buffers and create a CUDA stream.
    #     inputs, outputs, bindings, stream = common.allocate_buffers(engine)
    #     # Contexts are used to perform inference.
    #     with engine.create_execution_context() as context:
    #         # Load a normalized test case into the host input page-locked buffer.
    #         test_case = load_normalized_test_case(test_image, inputs[0].host)
    #         # Run the engine. The output will be a 1D tensor of length 1000, where each value represents the
    #         # probability that the image corresponds to that label
    #         trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    #         # We use the highest probability as our prediction. Its index corresponds to the predicted label.
    #         pred = np.argmax(trt_outputs[0]) #labels[np.argmax(trt_outputs[0])]
    #         # if "_".join(pred.split()) in os.path.splitext(os.path.basename(test_case))[0]:
    #         #     print("Correctly recognized " + test_case + " as " + pred)
    #         # else:
    #         #     print("Incorrectly recognized " + test_case + " as " + pred)
    
    # Time tensorrt code:
    time_list = []
    with build_engine(onnx_file_path) as engine:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        # Contexts are used to perform inference.
        with engine.create_execution_context() as context:
            # Load a normalized test case into the host input page-locked buffer.
            test_case = load_normalized_test_case(test_image, inputs[0].host)

            for i in range(1000):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
                end.record()

                # Waits for everything to finish running
                torch.cuda.synchronize()

                time_list.append(start.elapsed_time(end))
    fastest_trt_time = np.min(np.array(time_list))

    return fastest_trt_time

def time_pytorch(model_resnet, fp16=False, cuda=True):
    pytorch_img = np.asarray(Image.open(test_image).resize((WIDTH, HEIGHT), Image.ANTIALIAS)).transpose([2, 0, 1])
    pytorch_img = (pytorch_img / 255.0 - 0.45) / 0.225
    pytorch_img = torch.tensor(pytorch_img)
    pytorch_img = pytorch_img.unsqueeze(0).type(torch.FloatTensor)
    if BATCH_SIZE > 1:
        pytorch_img = torch.cat(BATCH_SIZE * [pytorch_img])
    if cuda:
        pytorch_img = pytorch_img.cuda()
    pred_pytorch = model_resnet(pytorch_img)
    pred_pytorch = torch.max(pred_pytorch, 1)[1]

    time_list = []

    # Time pytorch code:
    for i in range(100):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        if fp16:
            with autocast():
                pred_pytorch = model_resnet(pytorch_img)
        else:
            pred_pytorch = model_resnet(pytorch_img)
        end.record()

        # Waits for everything to finish running
        torch.cuda.synchronize()

        time_list.append(start.elapsed_time(end))
    fastest_pytorch_time = np.min(np.array(time_list))
    return fastest_pytorch_time

def time_pytorch_training_pass(model_resnet, batch_size=256):
    model_resnet.train()
    criterion = nn.CrossEntropyLoss().cuda()
    batch = torch.randn((batch_size, 3, WIDTH, HEIGHT)).cuda()
    targets = torch.randint(0, 999, (batch_size,)).cuda()

    time_list = []
    for i in range(10):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        pred_pytorch = model_resnet(batch)
        loss = criterion(pred_pytorch, targets)
        loss.backward()

        end.record()
        torch.cuda.synchronize()
        time_list.append(start.elapsed_time(end))

    fastest_pytorch_time = np.min(np.array(time_list))
    return fastest_pytorch_time


# %% Do the timing for PyTorch:
BATCH_SIZE = 1
for depth in [18, 34, 50, 101, 152]: # 18, 34, 50, 101, 152
    for recalibration_type in [None, 'meanrew', 'eca', 'srm', 'se', 'multise3']: # None, 'meanrew', 'eca', 'srm', 'se', 'multise3'
        WIDTH, HEIGHT = (224, 224)
        model_resnet = resnet(depth=depth, recalibration_type=recalibration_type).eval().cuda()
        print(f"ResNet-{depth} {recalibration_type}: {time_pytorch(model_resnet)}")

#%% TensorRt:
BATCH_SIZE = 1
for depth in [18, 34, 50, 101, 152]: # 18, 34, 50, 101, 152
    for recalibration_type in [None, 'meanrew', 'eca', 'srm', 'se', 'multise3']: # None, 'meanrew', 'eca', 'srm', 'se', 'multise3'
        WIDTH, HEIGHT = (224, 224)
        model_resnet = resnet(depth=depth, recalibration_type=recalibration_type).eval().cuda()
        print(f"ResNet-{depth} {recalibration_type}: {time_tensorrt(model_resnet, fp16=True)}")


# %% Include resolution sweep
BATCH_SIZE = 1
for depth in [50]: #[18, 34, 50, 101, 152]
    for recalibration_type in [None, 'meanrew', 'eca', 'srm', 'se', 'multise3']: # None, 'meanrew', 'eca', 'srm', 'se', 'multise3'
        for resolution in [32, 64, 128, 256, 512, 768, 1024]: #32, 64, 128, 256, 512, 768, 1024, 1536, 2048
            WIDTH, HEIGHT = (resolution, resolution)
            model_resnet = resnet(depth=depth, recalibration_type=recalibration_type).eval().cuda()
            print(f"ResNet-{depth} {resolution} {recalibration_type}: {time_pytorch(model_resnet, fp16=False)}")

# %% Resolution sweep on TensorRt
BATCH_SIZE = 1
for depth in [50]: #[18, 34, 50, 101, 152]
    for recalibration_type in [None, 'meanrew', 'eca', 'srm', 'se', 'multise3']: # None, 'meanrew', 'eca', 'srm', 'se', 'multise3'
        for resolution in [64, 128, 256, 512, 768, 1024, 1536]: #64, 128, 256, 512, 768, 1024, 1536, 2048
            WIDTH, HEIGHT = (resolution, resolution)
            model_resnet = resnet(depth=depth, recalibration_type=recalibration_type).eval().cuda()
            print(f"ResNet-{depth} {resolution} {recalibration_type}: {time_tensorrt(model_resnet, fp16=False)}")

#%% Batch size sweep on TensorRT
for depth in [50]: #[18, 34, 50, 101, 152]
    for recalibration_type in [None, 'meanrew', 'eca', 'srm', 'se', 'multise3']: # None, 'meanrew', 'eca', 'srm', 'se', 'multise3'
        for BATCH_SIZE in [1, 2, 4, 8, 16, 32, 64]: # 1, 2, 4, 8, 16, 32, 64, 128, 256
            WIDTH, HEIGHT = (224, 224)
            model_resnet = resnet(depth=depth, recalibration_type=recalibration_type).eval().cuda()
            print(f"ResNet-{depth} {BATCH_SIZE} {recalibration_type}: {time_tensorrt(model_resnet, fp16=False)}")

# %%
# PyTorch training pass:
# for depth in [18, 34, 50, 101, 152]: # 18, 34, 50, 101, 152
#     for recalibration_type in [None]: # None, 'se', 'srm', 'meanrew', 'multise3'
#         WIDTH, HEIGHT = (224, 224)
#         model_resnet = resnet(depth=depth, recalibration_type=recalibration_type).eval().cuda()
#         print(f"ResNet-{depth} {recalibration_type}: {time_pytorch_training_pass(model_resnet)}")
