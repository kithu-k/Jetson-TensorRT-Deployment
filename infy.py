import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import os
import time

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def calculate_metrics(pred_mask, gt_mask):
    """Calculates Intersection over Union (IoU) and Pixel Accuracy."""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    iou = intersection / union if union > 0 else 1.0
    
    correct_pixels = (pred_mask == gt_mask).sum()
    accuracy = correct_pixels / pred_mask.size
    return iou, accuracy

def run_evaluation_for_model(engine_path, model_type="static"):
    print(f"\nInitializing Evaluation for: {engine_path}")
    
    # 1. Load and deserialize TensorRT Engine
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    context = engine.create_execution_context()
    stream = cuda.Stream()

    # Define names based on inspected engine properties
    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)
    
    input_shape = engine.get_tensor_shape(input_name)
    output_shape = engine.get_tensor_shape(output_name)
    
    # Allocate GPU Memory space based on shapes
    d_input = cuda.mem_alloc(int(np.prod(input_shape) * np.dtype(np.float32).itemsize))
    d_output = cuda.mem_alloc(int(np.prod(output_shape) * np.dtype(np.float32).itemsize))
    
    # Configure TensorRT 10 execution pointers
    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output_name, int(d_output))

    # Folders Setup
    test_images_dir = "test"
    mask_images_dir = "mask"
    image_extensions = (".jpg", ".jpeg", ".png")
    
    image_files = [f for f in os.listdir(test_images_dir) if f.lower().endswith(image_extensions)]
    
    total_iou = 0.0
    total_acc = 0.0
    latencies = []
    processed_count = 0

    # 2. Iterate through matched dataset
    for filename in image_files:
        img_path = os.path.join(test_images_dir, filename)
        mask_path = os.path.join(mask_images_dir, filename)
        
        if not os.path.exists(mask_path):
            continue  # Skip if no matching ground truth mask exists
            
        # --- Preprocessing Step ---
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if model_type == "static":
            # Models 1-4 expect (1, 256, 256, 3)
            img_resized = cv2.resize(img, (256, 256))
            input_data = np.expand_dims(img_resized.astype(np.float32) / 255.0, axis=0)
        else:
            # CNN-LSTM expects (1, 2, 128, 128, 3)
            img_resized = cv2.resize(img, (128, 128))
            img_normalized = img_resized.astype(np.float32) / 255.0
            # Create sequence by repeating frame twice
            input_data = np.stack([img_normalized, img_normalized], axis=0)
            input_data = np.expand_dims(input_data, axis=0)

        input_data = np.ascontiguousarray(input_data)
        output_data = np.empty(output_shape, dtype=np.float32)

        # --- Inference Step with Precision Timers ---
        t_start = time.perf_counter()
        cuda.memcpy_htod_async(d_input, input_data, stream)
        context.execute_async_v3(stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(output_data, d_output, stream)
        stream.synchronize()
        t_end = time.perf_counter()
        
        # Track latency in milliseconds (omit first image to avoid cold-start warmup anomalies)
        if processed_count > 0:
            latencies.append((t_end - t_start) * 1000)

        # --- Postprocessing and Metric Evaluation ---
        # Squeeze down prediction probabilities to a 2D matrix
        pred_mask_prob = np.squeeze(output_data)
        pred_mask_binary = (pred_mask_prob > 0.5).astype(np.uint8)
        
        # Process matching ground truth mask (handling JPEG interpolation gray pixels)
        gt_mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        target_size = (128, 128) if model_type == "lstm" else (256, 256)
        gt_mask_resized = cv2.resize(gt_mask_raw, target_size, interpolation=cv2.INTER_NEAREST)
        gt_mask_binary = (gt_mask_resized > 127).astype(np.uint8)

        # Compute accuracy scores
        iou, acc = calculate_metrics(pred_mask_binary, gt_mask_binary)
        total_iou += iou
        total_acc += acc
        processed_count += 1

    # Cleanup GPU pointers explicitly
    del d_input
    del d_output

    # --- Final Compilation of Performance Stats ---
    if processed_count == 0:
        print(f"No valid image/mask pairs matched for {engine_path}")
        return None

    avg_iou = (total_iou / processed_count) * 100
    avg_acc = (total_acc / processed_count) * 100
    avg_latency = np.mean(latencies) if latencies else 0.0
    throughput_fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    return {
        "mIoU": f"{avg_iou:.2f}%",
        "Accuracy": f"{avg_acc:.2f}%",
        "Latency": f"{avg_latency:.2f} ms",
        "FPS": f"{throughput_fps:.2f}"
    }

if __name__ == "__main__":
    models_to_test = [
        {"file": "fcnn.engine", "type": "static"},
        {"file": "simple_cnn.engine", "type": "static"},
        {"file": "resnet50_flood_binary.engine", "type": "static"},
        {"file": "u-net.engine", "type": "static"},
        {"file": "CNN_LSTM.engine", "type": "lstm"}
    ]
    
    results_table = {}
    
    for m in models_to_test:
        if os.path.exists(m["file"]):
            metrics = run_evaluation_for_model(m["file"], model_type=m["type"])
            if metrics:
                results_table[m["file"]] = metrics
        else:
            print(f"[WARNING] Engine file not found: {m['file']}")
            
    # Print clean summary report table
    print("\n" + "="*70)
    print(f"{'MODEL ENGINE':<30} | {'mIoU':<8} | {'Accuracy':<10} | {'Latency':<10} | {'FPS':<6}")
    print("="*70)
    for model_name, data in results_table.items():
        print(f"{model_name:<30} | {data['mIoU']:<8} | {data['Accuracy']:<10} | {data['Latency']:<10} | {data['FPS']:<6}")
    print("="*70)
