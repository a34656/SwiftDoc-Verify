import json
import random
import streamlit as st

# Load the JSON database
def load_database():
    with open(r"C:\Users\KIIT0001\ocr\ocr_id\samples\new_preprocessed\aadhaar-full-dataset.json", "r") as f:
        return json.load(f)

# Save updated database
# def save_database(data):
#     with open("database.json", "w") as f:
#         json.dump(data, f)

from google.cloud import vision
import os
import re

# Set the path to your credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"C:\Users\KIIT0001\ocr\ocr_id\samples\new_preprocessed\ocr-recognition-440606-b6f682bf20bb.json"

# Initialize Vision API client
client = vision.ImageAnnotatorClient()
image_path = "adhar.jpg"

# Test text detection
def find_aadhar(image_path):
    with open(image_path, "rb") as image_file:
        content = image_file.read()
        image = vision.Image(content=content)

    aadhaar_pattern = r"\b[2-9]{1}[0-9]{3}\s?[0-9]{4}\s?[0-9]{4}\b"
    response = client.text_detection(image=image)
    if response.text_annotations:
        # print("Detected text:", response.text_annotations[0].description)
        matches = re.findall(aadhaar_pattern, response.text_annotations[0].description)
        if matches:
            return matches[0]

aadhar_number = find_aadhar(image_path)
# Generate OTP
def generate_otp():
    return str(random.randint(100000, 999999))

# Main function
def main():
    st.title("Aadhaar/PAN Verification System")

    if "otp" not in st.session_state:
        st.session_state["otp"] = None

    st.sidebar.header("Input Details")
    id_type = st.sidebar.selectbox("Select ID Type", ["Aadhaar", "PAN"])
    # id_number = st.sidebar.text_input("Enter ID Number")
    phone = st.sidebar.text_input("Enter your Registered Phone Number")

    if st.sidebar.button("Verify"):
        db = load_database()
        for user in db["users"]:
            if user["id_type"] == id_type and user["id_number"] == aadhar_number:
                if user["phone"] == phone:
                    otp = generate_otp()
                    st.session_state["otp"] = otp  # Save OTP to session state
                    st.success(f"OTP sent to your registered phone number: {otp}")
                    st.write("Enter the OTP below to complete verification:")
                    break
                else:
                    st.error("Phone number does not match.")
                    return
        else:
            st.error("ID not found in the database.")

    if st.session_state["otp"]:
        user_otp = st.text_input("OTP", type="password")
        if st.button("Submit OTP"):
            if user_otp == st.session_state["otp"]:
                st.success("Verification Successful!")
                st.session_state["otp"] = None  # Reset OTP after successful verification
            else:
                st.error("Invalid OTP. Please try again.")

if __name__ == "__main__":
    main()
