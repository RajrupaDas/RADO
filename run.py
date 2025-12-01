import onnxruntime as ort
import numpy as np
import cv2
import time
import sys

# --- CONFIGURATION (FINALIZED) ---
MODEL_PATH = "model7.onnx"
VIDEO_PATH = "istockphoto-1478951186-640_adpp_is.mp4"
CONF_THRESHOLD = 0.25 # Use a standard, higher confidence now that the structure is fixed
IOU_THRESHOLD = 0.45   

# CRITICAL FIX 1: Update CLASS_NAMES to match the model metadata
CLASS_NAMES = ["Right", "Left", "Cone"] 
# CRITICAL FIX 2: Define the cone's Class ID based on the metadata {'2': 'Cone'}
TARGET_CLASS_ID = 2 

# --- UTILITY FUNCTION: POST-PROCESSING (FINAL CORRECTED VERSION) ---

# CRITICAL FIX: Removed 'target_class_id' from the function signature
def postprocess_and_draw(output, img_w, img_h, conf_threshold, iou_threshold):
    """
    Handles post-processing (NMS) using the confirmed [1, 7, 8400] structure.
    """
    
    # The cone class ID is 2, and is defined here based on the confirmed structure.
    TARGET_CLASS_ID = 2 
    
    # 1. Transpose and reshape the output
    # Output shape is [1, 7, N]. Transpose to [N, 7].
    output = output[0].T 
    
    # 2. Separate confidence, boxes, and scores
    boxes = output[:, :4]         # Bounding box coordinates (cx, cy, w, h)
    scores_conf = output[:, 4:]   # The 3 Class Scores (Index 4, 5, 6)
    
    # *** CRITICAL FIX: ASSUME NO SEPARATE OBJECTNESS SCORE (scores_conf IS class_scores) ***
    class_scores = scores_conf 
    
    confidences = np.max(class_scores, axis=1)   # Max class score is the detection confidence
    class_ids = np.argmax(class_scores, axis=1)  # Index of the max class score (0, 1, or 2)
    
    # --- DEBUGGING RETURN: Get the maximum confidence score before filtering ---
    max_conf = np.max(confidences) if len(confidences) > 0 else 0.0

    # 3. Filter by confidence threshold 
    mask = (confidences >= conf_threshold)
    
    boxes = boxes[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    if len(boxes) == 0:
        return [], max_conf

    # 4. Convert boxes from (cx, cy, w, h) to (x1, y1, x2, y2)
    # Correct scaling back to original image size
    x_center, y_center, width, height = boxes.T
    
    x1 = (x_center - width / 2) * img_w / 640.0
    y1 = (y_center - height / 2) * img_h / 640.0
    x2 = (x_center + width / 2) * img_w / 640.0
    y2 = (y_center + height / 2) * img_h / 640.0
    
    x1 = np.clip(x1, 0, img_w).astype(np.int32)
    y1 = np.clip(y1, 0, img_h).astype(np.int32)
    x2 = np.clip(x2, 0, img_w).astype(np.int32)
    y2 = np.clip(y2, 0, img_h).astype(np.int32)

    # 5. Non-Maximum Suppression (NMS)
    nms_boxes = np.column_stack((x1, y1, x2 - x1, y2 - y1))
    indices = cv2.dnn.NMSBoxes(nms_boxes.tolist(), confidences.tolist(), conf_threshold, iou_threshold)
    
    final_indices = indices.flatten() if len(indices) > 0 else []
    
    # 6. Assemble final detections (FILTER BY TARGET CLASS ID = 2)
    final_detections = []
    for i in final_indices:
        if class_ids[i] == TARGET_CLASS_ID:
            # Format: [x1, y1, x2, y2, confidence, class_id]
            final_detections.append([x1[i], y1[i], x2[i], y2[i], confidences[i], class_ids[i]])

    return final_detections, max_conf

def draw_boxes_on_image(frame, final_detections, class_names):
    """Draws the final bounding boxes onto the frame."""
    for det in final_detections:
        x1, y1, x2, y2, conf, cls_id = det
        cls_id = int(cls_id)
        
        # Use the correct class name from the updated list
        label = f"{class_names[cls_id]}: {conf:.2f}"
        color = (0, 0, 255) # Red (BGR)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    return frame

# --- MAIN VIDEO INFERENCE SCRIPT (Modified) ---

# CRITICAL FIX: Removed 'target_class_id' from the function signature
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
    print("\n--- Starting Real-Time Inference ---")
    
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
            
            # 4. Post-processing and Drawing
            # CRITICAL FIX: Removed 'target_class_id' from the call
            final_detections, max_conf = postprocess_and_draw(
                outputs[0], 
                original_w, 
                original_h, 
                conf_threshold, 
                iou_threshold 
            )
            
            # --- DEBUGGING PRINT ---
            print(f"[Frame {frame_count:04d}] Max Confidence Found: {max_conf:.4f}")
            # ---------------------------------
            
            # Print Detected Locations
            if final_detections:
                print(f"  --> DETECTED {len(final_detections)} Cones! Bounding Boxes Drawn.")
                for i, det in enumerate(final_detections):
                    x1, y1, x2, y2, conf, cls_id = det
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    print(f"    Cone {i+1}: Center ({center_x}, {center_y}), Conf: {conf:.2f}")
            
            # 5. Display Result
            processed_frame = draw_boxes_on_image(frame, final_detections, class_names)
            
            fps = 1 / inference_time
            cv2.putText(processed_frame, f"FPS: {fps:.2f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("YOLOv8 ONNX Video Inference", processed_frame)
            
            # Check for 'q' key press to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\nVideo processing stopped and resources released.")

# Run the main function
if __name__ == "__main__":
    # CRITICAL FIX: Removed 'TARGET_CLASS_ID' from the function call
    run_yolov8_video_inference(
        MODEL_PATH, 
        VIDEO_PATH, 
        CONF_THRESHOLD, 
        IOU_THRESHOLD, 
        CLASS_NAMES
    )
