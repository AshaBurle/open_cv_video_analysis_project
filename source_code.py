import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Define the classes you are interested in
target_classes = ["snake", "centipede", "scorpion", "elephant", "lion", "leopard", "crocodile", "person"]

layer_names = net.getLayerNames()
try:
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
except TypeError:
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Define bright colors for target classes
bright_colors = {
    "snake": (0, 255, 255),         # Cyan
    "centipede": (255, 0, 255),     # Magenta
    "scorpion": (255, 255, 0),      # Yellow
    "elephant": (0, 128, 255),      # Orange
    "lion": (0, 255, 0),            # Green
    "leopard": (255, 0, 0),         # Blue
    "crocodile": (255, 255, 255),   # White
    "person": (0, 0, 255)           # Red (for thief detection)
}

# Function to process each frame and detect objects
def process_frame(frame):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = bright_colors.get(label, (0, 255, 255))  # Default to bright yellow if not found
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return frame

# Main function to capture video and apply processing
def main():
    cap = cv2.VideoCapture('input_video.mp4') # Change to 0 for webcam

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = process_frame(frame)

        cv2.imshow('Frame', processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
