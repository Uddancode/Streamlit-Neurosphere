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
  "happy": ["Minute maid fruit juice - https://www.walmart.com/ip/Minute-Maid-Premium-Mango-Punch-Fruit-Juice-59-fl-oz-Carton/187385053?classType=REGULAR&athbdg=L1600&from=/search", "Chocolate - https://www.walmart.com/ip/Ferrero-Rocher-Premium-Chocolate-Bar-Milk-Chocolate-Hazelnut-3-1-oz/260060982?classType=REGULAR&athbdg=L1600&from=/search", "Hoodies - https://www.walmart.com/ip/Cadmus-Men-s-Workout-Long-Sleeve-Fishing-shirts-UPF-50-Sun-Protection-Dry-Fit-Hoodies-1-Pack-096-White-Large/5027554790?classType=VARIANT&adsRedirect=true","Dress - https://www.walmart.com/search?q=dress","Chocolate - https://www.walmart.com/search?q=chocolate","Clothing -  https://www.walmart.com/cp/clothing/5438"],
  "sad": ["comfort blanket - https://www.walmart.com/search?q=comfort+blanket", "Warm tea - https://www.walmart.com/search?q=warm+tea", "Inspirational books - https://www.walmart.com/search?q=inspirational+books"],
  "angry": ["Stress ball - https://www.walmart.com/search?q=stress+ball", "Calming tea - https://www.walmart.com/search?q=calming+tea", "Mediataion Books - https://www.walmart.com/search?q=meditation+books","Water bottle - https://www.walmart.com/search?q=beautiful+beautiful+water+bottle"],
  "surprised": ["Soft toys - https://www.walmart.com/search?q=soft+toys","Mobile - https://www.walmart.com/search?q=mobile", "Adventure gear - https://www.walmart.com/search?q=adventure+gear", "Surprise box -https://www.walmart.com/search?q=surprise+box","Handbags - https://www.walmart.com/search?q=handbags"],
  "neutral": ["Earings-  https://www.walmart.com/search?q=earing%2014%20k%20gold%20yellow%20for%20wom%20e&typeahead=earin","Watch - https://www.walmart.com/search?q=watch" ,"Healthy Snacks - https://www.walmart.com/cp/health-inspired-snacks/6275730?q=healthy+snacks", "Soundbox - https://www.walmart.com/search?q=soundbox"],
  "fear": ["Pepper Spray - https://www.walmart.com/search?q=pepper+spray","Safety kit - https://www.walmart.com/search?q=safety+kit","Taser - https://www.walmart.com/search?q=taser", "Comfort food- https://www.walmart.com/search?q=comfort+food", "Stress relief kit - https://www.walmart.com/search?q=stress+relief+kit"],
  "disgust": ["Tissue paper - https://www.walmart.com/search?q=tissue+paper","Room freshener - https://www.walmart.com/search?q=room%20freshener", "Cleanser - https://www.walmart.com/search?q=cleanser", "Aromatherapy - https://www.walmart.com/search?q=aromatherapy"]
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

    st.write(f"Emotion Detected: {emotion}")
    st.write("Recommended Products:")
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

st.title("Face Emotion Detection Web App")
st.write("This web app detects emotions from a live webcam feed and suggests products based on the detected emotion.")

if st.button("Start Detection", key="start_detection_button"):
  detect_faces_and_emotions()