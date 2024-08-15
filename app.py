import streamlit as st
import cv2
from deepface import DeepFace
import pandas as pd
import os
from datetime import datetime

# Initialize face detector
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Prepare CSV file
csv_file = 'face_emotions.csv'
csv_data = []

# Define emotion to product mapping
emotion_to_products = {
  "happy": ["Joyful Juice", "Cheerful Chocolate", "Happy Hoodie"],
  "sad": ["Comfort Blanket", "Warm Tea", "Inspirational Book"],
  "angry": ["Stress Ball", "Calming Tea", "Meditation App"],
  "surprised": ["Exciting Gadgets", "Adventure Gear", "Surprise Box"],
  "neutral": ["Laptop- www.google.com ❤️‍🔥", "Healthy Snacks", "Relaxing Music"],
  "fear": ["Safety Kit", "Comfort Food", "Stress Relief Kit"],
  "disgust": ["Refreshing Drink", "Cleanser", "Aromatherapy Kit"]
}

def detect_emotion(frame):
  emotion = "Unknown"
  products = []
  try:
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = DeepFace.analyze(rgb_frame, actions=['emotion'])
    emotion = result[0]['dominant_emotion']  # Extract dominant emotion
    products = emotion_to_products.get(emotion, ["No products available"])
  except Exception as e:
    st.error(f"Error analyzing frame: {e}")
  return emotion, products

def detect_faces_and_emotions():
  # Create a directory to store images if it doesn't exist
  if not os.path.exists('saved_faces'):
    os.makedirs('saved_faces')

  # Start video capture
  cap = cv2.VideoCapture(0)
  if not cap.isOpened():
    st.error("Cannot open webcam!")
    return

  stframe = st.empty()
  while True:
    ret, frame = cap.read()
    if not ret:
      break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    emotion, products = detect_emotion(frame)

    for (x, y, w, h) in faces:
      cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, emotion, (50, 50), font, 1, (0, 0, 255), 2, cv2.LINE_4)
    stframe.image(frame, channels="BGR")

    st.write(f"*Emotion Detected:* {emotion}")
    st.write("*Recommended Products:*")
    st.write(", ".join(products))

    # Button for saving image with unique key
    if st.button("Save Image", key=f"save_image_{datetime.now().strftime('%Y%m%d%H%M%S')}"):
      timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
      face_filename = f'saved_faces/face_{timestamp}.jpg'
      cv2.imwrite(face_filename, frame)

      csv_data.append({
          'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
          'Emotion': emotion,
          'File Path': face_filename
      })
      print(f"Image saved and data logged to CSV: {face_filename}")

    if st.button("Quit", key="quit_app_button"):
      break

  cap.release()
  cv2.destroyAllWindows()

  # Save CSV file with emotion and product data
  df = pd.DataFrame(csv_data)
  df.to_csv(csv_file, index=False)
  st.success(f"Data saved to {csv_file}")

def display_csv_data():
  if os.path.exists(csv_file):
    # Check if the file is empty
    if os.path.getsize(csv_file) > 0:
      try:
        df = pd.read_csv(csv_file)
        st.write("### Logged Emotions and Product Recommendations")
        st.dataframe(df)
      except pd.errors.EmptyDataError:
        st.write("The file is empty or cannot be read.")
    else:
      st.write("The file is empty.")
  else:
    st.write("No data available. Start detection to log emotions.")

st.title("Welcome To NeuroSphere!!")
st.write("This web app detects emotions from a live webcam feed and suggests products based on the detected emotion. To make your shopping experience happy and more reliable")

if st.button("Start Detection", key="start_detection_button"):
  detect_faces_and_emotions()

