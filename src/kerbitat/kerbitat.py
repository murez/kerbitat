"""
Kerbitat is a profiling tool for pridicting running time of a DNN model on NVIDIA GPUs.
It can predict the performance of the target GPU from the profiling result of the source GPU.
Author: Siyuan Zhang
Email:  zhangsy3@shanghaitech.edu.cn
"""
from typing import Callable
import pickle
from enum import IntEnum
import time
from habitat.profiling.kernel import KernelProfiler
from habitat.analysis import Device
import numpy as np
import torch


class GPU_TYPE(IntEnum):
    """
    The GPU type of NVIDIA GPU.
    Currently, Kerbitat support the following 8 GPU types.
    """
    A40 = 0
    RTX2080Ti = 1
    RTX_TITAN = 2
    TITAN_Xp = 3
    TITAN_V = 4
    GTX1080 = 5
    TITAN_X = 6
    M40 = 7


class Kerbitat:
    """
    Kerbitat is a profiling tool for NVIDIA GPU.
    It can predict the performance of the target GPU from the profiling result of the source GPU.
    1. Kerbitat use KernelProfiler from habitat to get mangled kernel names of running DNN model.
    2. Kerbitat convert the mangled kernel names to feature vector.
    3. Kerbitat use a trained model to predict the performance of the target GPU from the 
       feature vector.
    4. Kerbitat use the predicted performance to estimate the execution time of the target GPU.
    A Big Thanks to habitat team for providing KernelProfiler and all the other great thoughts.
    """

    def __init__(self, test_device: GPU_TYPE) -> None:
        """
        test_device: GPU_TYPE
            The GPU type of the profing device.
        """
        self.profiler = KernelProfiler(device=Device.A40)
        self.all_kernel_to_id = pickle.load(
            open("./data/kernel_to_id.pkl", "rb"))
        self.predictor = torch.load("./data/predictor.pt")
        self.measured_kernel = []
        self.predict_result = np.array(
            len(GPU_TYPE) * [0.0]).astype(np.float32)
        self.test_device = test_device
        self.profile_time = 1.0

    def profiling(self, runable: Callable, run_timer: bool = True) -> None:
        """
        runable: Callable
            The function to be profiled.
        run_timer: bool
            Whether to run the timer. Default is True.
            Otherwise, the profiling result will be just rate not the time.
        This function will run the runable and get the profiling result.
        """
        self.measured_kernel = self.profiler.measure_kernels(runable)
        kernel_names = [m.name for m in self.measured_kernel]
        feature_kernel = np.array(
            len(self.all_kernel_to_id) * [0.0]).astype(np.float32)
        for i, k in enumerate(kernel_names):
            if k in self.all_kernel_to_id:
                feature_kernel[self.all_kernel_to_id[k]] += 1
        input_tensor = torch.tensor(feature_kernel).unsqueeze(0)
        self.predictor.eval()
        predict_result = self.predictor(input_tensor)[0].detach().numpy()[0]
        self.predict_result = predict_result
        if run_timer:
            self.timer(runable)

    def timer(self, runable: Callable) -> None:
        """
        runable: Callable
            The function to be timed.
        This function will run the runable and get the execution time.
        """
        start_time = time.perf_counter()
        runable()
        end_time = time.perf_counter()
        self.profile_time = end_time - start_time

    def predict(self) -> np.ndarray:
        """
        Return the predict result.
        """
        return self.predict_result

    def convert_rate(self, gpu_type_dst: GPU_TYPE) -> float:
        """
        gpu_type_dst: GPU_TYPE
            The GPU type of the target device.
        Return the rate of the target GPU to the profiling GPU.
        """
        return self.predict_result[gpu_type_dst] / self.predict_result[self.test_device]

    def get_target_time(self, gpu_type_dst: GPU_TYPE) -> float:
        """
        gpu_type_dst: GPU_TYPE
            The GPU type of the target device.
        Return the estimated execution time of the target GPU.
        """
        return self.profile_time * self.convert_rate(gpu_type_dst)
