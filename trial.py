import streamlit as st
import cv2
from ultralytics import YOLO

model6 = YOLO('runs/detect/train5/weights/best.pt')
model7 = YOLO('runs/detect/train7/weights/best.pt')
model5 = YOLO('runs/detect/train15/weights/best.pt')

def detect_with_model(model, frame, required_classes):
    results = model.predict(source=frame, conf=0.25, save=False, verbose=False)
    detected_classes = set()
    for box in results[0].boxes:
        class_id = int(box.cls)
        detected_classes.add(model.names[class_id])
    all_classes_detected = required_classes.issubset(detected_classes)
    return detected_classes, all_classes_detected

def classify_image_as_aadhaar(image_path, model1, model2, required_classes1, required_classes2):
    
    # Step 1: Load the image
    frame = cv2.imread(image_path)
    if frame is None:
        st.error("Image not found.")
        return

    # Step 2: Perform detection
    detected_classes, is_aadhaar1 = detect_with_model(model1, frame, required_classes1)

    # Step 3: Annotate the image
    annotated_frame = model1.predict(source=frame, conf=0.25, save=False)[0].plot()
    if is_aadhaar1:
        detected_classes, is_aadhaar2 = detect_with_model(model2, frame, required_classes2)
        if is_aadhaar2:
            st.success("The image has been classified as an Aadhaar card.")
        else:
            st.error("1The image does not meet Aadhaar card criteria.")
    else:
            st.error("The image does not meet Aadhaar card criteria.")

    # Step 4: Display results
    st.image(annotated_frame, channels="BGR", use_column_width=True)
    
# Example usage
image_path = "adhar.jpg"
required_classes_model1 = {"PIC", "Satyamav Jayate"}  # Aadhaar detection classes
# required_classes_model2 = {"DOB", "Income Logo", "Name", "Pan Number"}
required_classes_model3 = {"GOV", "Name", "DOB", "Gender"}
classify_image_as_aadhaar(image_path, model6, model5, required_classes_model1, required_classes_model3)
