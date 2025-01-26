import cv2
import json
import time
from ultralytics import YOLO
import streamlit as st
import os
import re
from google.cloud import vision
import numpy as np

# Initialize models 
model6 = YOLO(r'runs/detect/train5/weights/best.pt')
model7 = YOLO(r'runs/detect/train7/weights/best.pt')
model5 = YOLO(r'runs/detect/train15/weights/best.pt')

# Aadhaar number pattern
aadhaar_pattern = r"\b[2-9]{1}[0-9]{3}\s?[0-9]{4}\s?[0-9]{4}\b"
pan_pattern = r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"

# Set Google Vision API credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"C:\Users\KIIT0001\ocr\ocr_id\samples\new_preprocessed\ocr-recognition-440606-b6f682bf20bb.json"
client = vision.ImageAnnotatorClient()

# Load the JSON database
def load_aadhar_database():
    with open(r"C:\Users\KIIT0001\ocr\ocr_id\samples\new_preprocessed\aadhaar-full-dataset.json", "r") as f:
        return json.load(f)
    
def load_pan_database():
    with open(r"C:\Users\KIIT0001\ocr\ocr_id\samples\new_preprocessed\pan-card-database.json", "r") as f:
        return json.load(f)

def validate_aadhaar_number(number):
    # Example: Add a basic length check (12 digits) or Luhn check for Aadhaar.
    clean_number = number.replace(" ", "")
    return len(clean_number) == 12 and clean_number.isdigit()

def validate_pan_number(number):
    clean_number = number.strip()
    return len(clean_number) == 10 and re.match(pan_pattern, clean_number)

def detect_text_in_frame(frame):
    _, encoded_image = cv2.imencode('.jpg', frame)
    content = encoded_image.tobytes()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    return texts[0].description if texts else "No text detected"

def detect_text_in_image(image_path):
    with open(image_path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    
    response = client.text_detection(image=image)
    texts = response.text_annotations

    if response.error.message:
        raise Exception(f"API Error: {response.error.message}")
    
    return texts[0].description if texts else "No text detected"

def detect_with_model(model, frame, required_classes):
    results = model.predict(source=frame, conf=0.25, save=False, verbose=False)
    detected_classes = set()
    for box in results[0].boxes:
        class_id = int(box.cls)
        detected_classes.add(model.names[class_id])
    all_classes_detected = required_classes.issubset(detected_classes)
    return detected_classes, all_classes_detected

def extract_aadhaar_number(text):
    matches = re.findall(aadhaar_pattern, text)
    return matches if matches else []

def extract_pan_number(text):
    matches = re.findall(pan_pattern, text)
    return matches if matches else []

def classify_image(image_path, model1, model2, model3, required_classes1, required_classes3, required_classes2):
    # Step 1: Load the image
    id_type = "Null"
    ad_no = 0
    is_aadhaar = False
    is_pan = False

    frame = cv2.imread(image_path)
    if frame is None:
        st.error("Image not found.")
        return False, id_type, ad_no

    detected_classes, is_success1 = detect_with_model(model1, frame, required_classes1)

    if is_success1:
        # Aadhaar check
        annotated_frame1 = model1.predict(source=frame, conf=0.25, save=False)[0].plot()
        detected_classes, is_aadhaar = detect_with_model(model2, frame, required_classes3)
        if is_aadhaar:
            annotated_frame2 = model2.predict(source=frame, conf=0.25, save=False)[0].plot()
            st.success("The image has been classified as an Aadhaar card.")
            height1, width1, _ = annotated_frame1.shape
            height2, width2, _ = annotated_frame2.shape

            if height1 != height2 or width1 != width2:
                annotated_frame2 = cv2.resize(annotated_frame2, (width1, height1))

            combined_frame = np.hstack((annotated_frame1, annotated_frame2))
            st.image(combined_frame, channels="BGR", use_column_width=True)
            id_type = "Aadhaar"
            text_detected = detect_text_in_image(image_path)
            aadhaar_numbers = extract_aadhaar_number(text_detected)
            if aadhaar_numbers and validate_aadhaar_number(aadhaar_numbers[0]):
                ad_no = aadhaar_numbers[0]
                st.write(f"Aadhaar Numbers Detected: {ad_no}")
            else:
                st.error("The image does not meet Aadhaar card criteria.")
                is_aadhaar = False

        # PAN check
        detected_classes, is_pan = detect_with_model(model3, frame, required_classes2)
        if is_pan:
            annotated_frame3 = model3.predict(source=frame, conf=0.25, save=False)[0].plot()
            st.success("The image has been classified as a PAN card.")
            height1, width1, _ = annotated_frame1.shape
            height3, width3, _ = annotated_frame3.shape

            if height1 != height3 or width1 != width3:
                annotated_frame3 = cv2.resize(annotated_frame3, (width1, height1))

            combined_frame2 = np.hstack((annotated_frame1, annotated_frame3))
            st.image(combined_frame2, channels="BGR", use_column_width=True)
            id_type = "PAN"
            text_detected = detect_text_in_image(image_path)
            pan_number = extract_pan_number(text_detected)  # You need to implement this function
            if pan_number and validate_pan_number(pan_number):  # You need to implement this function
                ad_no = pan_number[0]
                st.write(f"PAN Number Detected: {ad_no[0]}")
            else:
                st.error("The image does not meet PAN card criteria.")
                is_pan = False

    if not is_aadhaar and not is_pan:
        st.error("The image does not meet either Aadhaar or PAN card criteria.")
    elif is_aadhaar and not is_pan:
        st.success("The image has been classified as an Aadhaar card but not PAN.")
    else:
        st.success("The image has been classified as a PAN card but not Aadhaar.")
    return (is_aadhaar or is_pan), id_type, ad_no

def main():

    if "metrics" not in st.session_state:
        st.session_state.metrics = {"aadhaar_scanned": 0, "pan_scanned": 0, "aadhaar_success": 0, "pan_success": 0}
    if "id_type" not in st.session_state:
        st.session_state.id_type = None
    if "id_number" not in st.session_state:
        st.session_state.id_number = None

    st.title("Aadhaar & PAN Card Detection System")

    id_type = "None"
    id_number = 0
    required_classes_model1 = {"PIC", "Satyamav Jayate"}  # Aadhaar detection classes
    required_classes_model2 = {"DOB", "Income Logo", "Name", "Pan Number"}
    required_classes_model3 = {"GOV", "Name", "DOB", "Gender"}
    # Metrics
    metrics = {"aadhaar_scanned": 0, "pan_scanned": 0, "aadhaar_success": 0, "pan_success": 0}
    aadhaar_col, pan_col = st.columns(2)
    aadhaar_col.metric("Aadhaar Scanned", metrics["aadhaar_scanned"])
    pan_col.metric("PAN Scanned", metrics["pan_scanned"])

    # Choice for user: Image or Live Feed
    choice = st.radio("Choose Verification Method:", ["Image Upload", "Live Feed"])

    if choice == "Image Upload":
        uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"], key="image_uploader1")
        if uploaded_file:
            # Save uploaded file temporarily
            temp_file_path = f"temp_image.{uploaded_file.type.split('/')[-1]}"
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(uploaded_file.read())

            result, id_type1, id_number  = classify_image(temp_file_path, model6, model7, model5, required_classes_model1, required_classes_model3, required_classes_model2)
            id_type = id_type1
            if result and id_type1 == "Aadhaar":
                metrics["aadhaar_scanned"] += 1
                metrics["aadhaar_success"] += 1
                aadhaar_col.metric("Aadhaar Scanned", metrics["aadhaar_scanned"])
            elif result and id_type1 == "PAN":
                metrics["pan_scanned"] += 1
                metrics["pan_success"] += 1
                pan_col.metric("PAN Scanned", metrics["pan_scanned"])
            else:
                st.error("Aadhaar verification failed.")

    elif choice == "Live Feed":
        live_feed = st.checkbox("Enable Live Detection", key="live_detection_checkbox")
        frame_placeholder = st.empty()
        info_placeholder = st.empty()
        cap = None

        if live_feed:
            if cap is None:
                cap = cv2.VideoCapture(0)  # Use the default camera

            while live_feed:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture video feed.")
                    break

                # Aadhaar Detection
                detected_classes_1, step1_success = detect_with_model(model6, frame, required_classes_model1)
                
                # Annotate frame with detected classes
                annotated_frame = frame.copy()
                cv2.putText(annotated_frame, f"Detected: {', '.join(detected_classes_1)}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if step1_success:
                    detected_classes_3, aadhar_success = detect_with_model(model7, frame, required_classes_model3)
                    
                    # Update annotation with more specific classes
                    cv2.putText(annotated_frame, f"Specific: {', '.join(detected_classes_3)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    if aadhar_success:
                        annotated_frame = model6.predict(source=frame, conf=0.25, save=False)[0].plot()
                        annotated_frame = model7.predict(source=frame, conf=0.25, save=False)[0].plot()
                        metrics["aadhaar_scanned"] += 1
                        
                        # Add text to indicate successful detection
                        cv2.putText(annotated_frame, "Aadhaar Card Detected", (10, 90), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


                    detected_classes_2, pan_success = detect_with_model(model5, frame, required_classes_model2)
                    cv2.putText(annotated_frame, f"Specific: {', '.join(detected_classes_2)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    if pan_success:
                        annotated_frame = model6.predict(source=frame, conf=0.25, save=False)[0].plot()
                        annotated_frame = model5.predict(source=frame, conf=0.25, save=False)[0].plot()
                        metrics["pan_scanned"] += 1
                        
                        # Add text to indicate successful detection
                        cv2.putText(annotated_frame, "PAN Card Detected", (10, 90), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                # Display the annotated frame
                frame_placeholder.image(annotated_frame, channels="BGR", use_column_width=True)

                # # Display additional information
                # info_placeholder.text(f"Detected classes: {', '.join(detected_classes_1)}")
                # if step1_success:
                #     info_placeholder.text(f"Specific classes: {', '.join(detected_classes_3)}")

                # Stop the feed if the checkbox is unchecked
                live_feed = st.session_state.get("live_detection_checkbox", False)

            if cap:
                cap.release()
            cv2.destroyAllWindows()

    phone = st.sidebar.text_input("Enter your Registered Phone Number")

    if st.sidebar.button("Verify"):
        aadhar_db = load_aadhar_database()  # Load Aadhar database
        pan_db = load_pan_database()  # Load PAN database
        
        if id_type == "Aadhaar":
            db = aadhar_db
            id_number = id_number
        elif id_type == "PAN":
            db = pan_db
            id_number = id_number
        else:
            st.error("Invalid ID type")
            return

        for user in db["users"]:
            if user["id_number"] == id_number:
                if user["phone"] == phone:
                    st.success(f"Your {id_type} Number is verified")
                    break
                else:
                    st.error("Phone number does not match.")
                    return
        else:
            st.error(f"{id_type} not found in the database.")

    # if st.button("Scan Another"):
    #     for key in ["id_type", "id_number", "image_uploader1"]:
    #         if key in st.session_state:
    #             del st.session_state[key]
    #     st.set_query_params()
if __name__ == "__main__":
    main()

