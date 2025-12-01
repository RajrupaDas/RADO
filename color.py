import onnxruntime as ort
import numpy as np
import cv2
import time
import sys

# --- CONFIGURATION ---
MODEL_PATH = "model7.onnx"
VIDEO_PATH = "istockphoto-1478951186-640_adpp_is.mp4"
CONF_THRESHOLD = 0.25 
IOU_THRESHOLD = 0.45   

CLASS_NAMES = ["Right", "Left", "Cone"] 
TARGET_CLASS_ID = 2                      

# --- SIMPLIFIED HSV COLOR RANGES (OpenCV Hue is 0-179) ---
COLOR_RANGES = {
    # 🚨 CRITICAL FIX: Widen the band for red/orange to catch all variations.
    # We treat orange cones as RED for your simplified classification.
    # Lower Red (0-15) + Upper Red (165-179) + Orange/Yellow (15-30)
    "RED_ORANGE_LOW": (0, 30),        
    "RED_ORANGE_HIGH": (160, 179), # For the Hue wrap-around (pure red)
    
    "GREEN": (40, 80),        
    "BLUE": (100, 130),       
}

def classify_color_hsv(frame, x1, y1, x2, y2):
    """
    Extracts the bounding box, calculates the MEDIAN HSV, and classifies the color.
    """
    # 1. Extract Region of Interest (ROI) - Center 50%
    w = x2 - x1
    h = y2 - y1
    x_start = x1 + int(0.25 * w)
    y_start = y1 + int(0.25 * h)
    x_end = x2 - int(0.25 * w)
    y_end = y2 - int(0.25 * h)
    
    x_start = np.clip(x_start, 0, frame.shape[1])
    y_start = np.clip(y_start, 0, frame.shape[0])
    x_end = np.clip(x_end, 0, frame.shape[1])
    y_end = np.clip(y_end, 0, frame.shape[0])
    
    roi = frame[y_start:y_end, x_start:x_end]
    
    if roi.size == 0:
        return "OTHER"

    # 2. Convert to HSV
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # 3. Calculate MEDIAN HSV (Robust against noise)
    h_median, s_median, v_median = np.median(hsv_roi, axis=(0, 1))
    h, s, v = h_median, s_median, v_median

    # 4. ROVER GATE: Check for saturation and brightness (must be colorful and visible)
    # Tweak S and V thresholds if cones are highly reflective or very dull
    if s < 60 or v < 60:
        return "OTHER"

    # 5. Classify based on Median Hue (H)
    
    # Check for RED (includes Orange)
    is_red_low = (h >= COLOR_RANGES["RED_ORANGE_LOW"][0] and h <= COLOR_RANGES["RED_ORANGE_LOW"][1])
    is_red_high = (h >= COLOR_RANGES["RED_ORANGE_HIGH"][0] and h <= COLOR_RANGES["RED_ORANGE_HIGH"][1])
    
    if is_red_low or is_red_high:
        return "RED"
        
    # Check for GREEN
    elif h >= COLOR_RANGES["GREEN"][0] and h <= COLOR_RANGES["GREEN"][1]:
        return "GREEN"
        
    # Check for BLUE
    elif h >= COLOR_RANGES["BLUE"][0] and h <= COLOR_RANGES["BLUE"][1]:
        return "BLUE"
    
    # Fallback
    return "OTHER"

# --- UTILITY FUNCTION: POST-PROCESSING (Unmodified) ---

def postprocess_and_draw(output, img_w, img_h, conf_threshold, iou_threshold):
    TARGET_CLASS_ID = 2 
    output = output[0].T 
    boxes = output[:, :4]         
    scores_conf = output[:, 4:]   
    class_scores = scores_conf 
    confidences = np.max(class_scores, axis=1)   
    class_ids = np.argmax(class_scores, axis=1)  
    max_conf = np.max(confidences) if len(confidences) > 0 else 0.0

    mask = (confidences >= conf_threshold)
    boxes = boxes[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    if len(boxes) == 0:
        return [], max_conf

    x_center, y_center, width, height = boxes.T
    
    x1 = (x_center - width / 2) * img_w / 640.0
    y1 = (y_center - height / 2) * img_h / 640.0
    x2 = (x_center + width / 2) * img_w / 640.0
    x2 = np.clip(x2, 0, img_w).astype(np.int32)
    y2 = (y_center + height / 2) * img_h / 640.0
    
    x1 = np.clip(x1, 0, img_w).astype(np.int32)
    y1 = np.clip(y1, 0, img_h).astype(np.int32)
    y2 = np.clip(y2, 0, img_h).astype(np.int32)

    nms_boxes = np.column_stack((x1, y1, x2 - x1, y2 - y1))
    indices = cv2.dnn.NMSBoxes(nms_boxes.tolist(), confidences.tolist(), conf_threshold, iou_threshold)
    final_indices = indices.flatten() if len(indices) > 0 else []
    
    final_detections = []
    for i in final_indices:
        if class_ids[i] == TARGET_CLASS_ID:
            final_detections.append([x1[i], y1[i], x2[i], y2[i], confidences[i], class_ids[i]])

    return final_detections, max_conf

# --- MAIN VIDEO INFERENCE SCRIPT (Modified for Color) ---

def run_yolov8_video_inference(model_path, video_path, conf_threshold, iou_threshold, class_names):
    
    try:
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    except Exception as e:
        print(f"ERROR: Could not load ONNX model. Details: {e}")
        sys.exit(1)

    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    input_h, input_w = input_shape[2:] 
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"ERROR: Could not open video file at {video_path}.")
        sys.exit(1)

    frame_count = 0
    print("\n--- Starting Real-Time Inference with Finalized Color Detection ---")
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("End of video stream or error reading frame.")
                break

            frame_count += 1
            original_h, original_w = frame.shape[:2]

            # 2. Preprocess Frame 
            resized_frame = cv2.resize(frame, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
            input_tensor = resized_frame.astype(np.float32) / 255.0
            input_tensor = np.transpose(input_tensor, (2, 0, 1)) 
            input_tensor = np.expand_dims(input_tensor, 0)       
            
            # 3. Run Inference
            start_time = time.time()
            outputs = session.run(None, {input_name: input_tensor})
            inference_time = (time.time() - start_time)
            
            # 4. Post-processing
            final_detections, max_conf = postprocess_and_draw(
                outputs[0], 
                original_w, 
                original_h, 
                conf_threshold, 
                iou_threshold 
            )
            
            # --- COLOR CLASSIFICATION AND PRINTING ---
            print(f"[Frame {frame_count:04d}] Max Confidence Found: {max_conf:.4f}")
            
            current_frame = frame.copy()
            
            if final_detections:
                print(f"  --> DETECTED {len(final_detections)} Cones:")
                for i, det in enumerate(final_detections):
                    x1, y1, x2, y2, conf, cls_id = det
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    # 5. Classify Color 
                    cone_color = classify_color_hsv(current_frame, x1, y1, x2, y2)
                    
                    # 6. Print to Console
                    print(f"    Cone {i+1}: Center ({center_x}, {center_y}), Color: {cone_color}, Conf: {conf:.2f}")

                    # 7. Draw box and label onto the frame
                    label = f"{class_names[int(cls_id)]} - {cone_color}"
                    
                    # Use BGR color for drawing
                    if cone_color == "RED": color_bgr = (0, 165, 255) # Orange/Red for display
                    elif cone_color == "GREEN": color_bgr = (0, 255, 0) 
                    elif cone_color == "BLUE": color_bgr = (255, 0, 0)  
                    else: color_bgr = (255, 255, 255)                  # White/Other

                    cv2.rectangle(current_frame, (x1, y1), (x2, y2), color_bgr, 2)
                    cv2.putText(current_frame, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2)
            
            # 8. Display Result
            fps = 1 / inference_time
            cv2.putText(current_frame, f"FPS: {fps:.2f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("YOLOv8 ONNX Video Inference with Color", current_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\nVideo processing stopped and resources released.")

# Run the main function
if __name__ == "__main__":
    run_yolov8_video_inference(
        MODEL_PATH, 
        VIDEO_PATH, 
        CONF_THRESHOLD, 
        IOU_THRESHOLD, 
        CLASS_NAMES
    )
