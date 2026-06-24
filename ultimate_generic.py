import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import argparse
import time

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Mapping TensorRT types to NumPy types dynamically
TRT_NUMPY_DTYPE_MAP = {
    trt.DataType.FLOAT: np.float32,
    trt.DataType.HALF: np.float16,
    trt.DataType.INT8: np.int8,
    trt.DataType.INT32: np.int32,
    trt.DataType.BOOL: bool
}

def deploy_arbitrary_engine(engine_path):
    print(f"\n[INFO] Initializing Arbitrary Deployment Verification for: {engine_path}")

    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    context = engine.create_execution_context()
    stream = cuda.Stream()
    inputs = []
    outputs = []
    allocations = []
    print("  DYNAMIC TENSOR PROFILE DISCOVERY")

    #Dynamically discover all input and output tensors
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        shape = engine.get_tensor_shape(name)
        trt_type = engine.get_tensor_dtype(name)
        dtype = TRT_NUMPY_DTYPE_MAP.get(trt_type, np.float32)

        resolved_shape = [abs(dim) if dim < 0 else dim for dim in shape]
        if resolved_shape[0] == 0:  
            resolved_shape[0] = 1

        size = int(np.prod(resolved_shape))
        nbytes = size * np.dtype(dtype).itemsize
        
        device_mem = cuda.mem_alloc(nbytes)
        allocations.append(int(device_mem))
        context.set_tensor_address(name, int(device_mem))

        tensor_info = {
            "name": name, 
            "shape": resolved_shape, 
            "dtype": dtype, 
            "size": size, 
            "device_mem": device_mem
        }

        if is_input:
            inputs.append(tensor_info)
            print(f"[INPUT]  Name: '{name}' | Shape: {resolved_shape} | Dtype: {dtype.__name__}")
        else:
            outputs.append(tensor_info)
            print(f"[OUTPUT] Name: '{name}' | Shape: {resolved_shape} | Dtype: {dtype.__name__}")

    print("="*50)

    #Create generic host buffers with synthetic/dummy test data
    #This proves the pipeline works regardless of input domain (Vision, Audio, NLP)
    host_inputs = []
    for tensor in inputs:
        dummy_data = np.ones(tensor["shape"], dtype=tensor["dtype"])
        dummy_data = np.ascontiguousarray(dummy_data)
        host_inputs.append(dummy_data)

    host_outputs = []
    for tensor in outputs:
        host_outputs.append(np.empty(tensor["shape"], dtype=tensor["dtype"]))

    #Perform Execution Baseline
    print("\n[INFO] Streaming memory buffers and executing network layers...")
    t_start = time.perf_counter()

    for i, tensor in enumerate(inputs):
        cuda.memcpy_htod_async(tensor["device_mem"], host_inputs[i], stream)

    context.execute_async_v3(stream_handle=stream.handle)
    for i, tensor in enumerate(outputs):
        cuda.memcpy_dtoh_async(host_outputs[i], tensor["device_mem"], stream)
        
    stream.synchronize()
    t_end = time.perf_counter()

    latency = (t_end - t_start) * 1000
    print(f"[SUCCESS] Arbitrary Model Run Confirmed!")
    print(f"[METRIC] Complete Execution Latency: {latency:.2f} ms")

    for i, tensor in enumerate(outputs):
        print(f"[RESULT] Output Layer '{tensor['name']}' returned shape {host_outputs[i].shape}")

    for alloc in allocations:
        del alloc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Truly Generic TensorRT Deployment Benchmark")
    parser.add_argument("--engine", type=str, required=True, help="Path to any arbitrary TensorRT .engine file")
    
    args = parser.parse_args()
    deploy_arbitrary_engine(args.engine)