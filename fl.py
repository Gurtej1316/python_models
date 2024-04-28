from flask import Flask, request, jsonify
import cv2
import speech_recognition as sr
import threading
import queue
import time
from collections import Counter
from mtcnn import MTCNN
from deepface import DeepFace
from langdetect import detect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Initialize the recognizer
r = sr.Recognizer()

def calculate_accuracy(transcript, topic):
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Calculate TF-IDF matrices for transcript and topic
    tfidf_transcript = vectorizer.fit_transform([transcript])
    tfidf_topic = vectorizer.transform([topic])

    # Calculate cosine similarity between transcript and topic
    similarity_score = cosine_similarity(tfidf_transcript, tfidf_topic)

    # Accuracy is the cosine similarity score (range: 0 to 1)
    accuracy = similarity_score[0][0] if similarity_score.shape[0] > 0 else 0.0
    return accuracy

def detect_emotions(frame, emotions_counter, mtcnn_detector):
    try:
        # Detect faces in the frame
        result = mtcnn_detector.detect_faces(frame)

        if result:
            # Loop through detected faces
            for face_data in result:
                x, y, w, h = face_data['box']
                roi = frame[y:y+h, x:x+w]

                # Recognize emotion
                emotions = DeepFace.analyze(roi, actions=['emotion'], enforce_detection=False)

                # Loop through each detected face in the emotions list
                for emotion_data in emotions:
                    if 'emotion' in emotion_data:
                        # Extract dominant emotion and face confidence for this face
                        dominant_emotion = emotion_data['dominant_emotion']
                        facial_confidence = max(emotion_data['emotion'].values())  # Use maximum confidence

                        # Cap facial confidence at 100% for each emotion
                        facial_confidence = min(facial_confidence, 100.0)

                        # Increment emotion count and update total confidence
                        if dominant_emotion in emotions_counter:
                            emotions_counter[dominant_emotion]['count'] += 1
                            emotions_counter[dominant_emotion]['total_confidence'] += facial_confidence
                        else:
                            emotions_counter[dominant_emotion] = {
                                'count': 1,
                                'total_confidence': facial_confidence
                            }

    except Exception as e:
        print("Error in detect_emotions:", str(e))

@app.route('/analyze_video_and_audio', methods=['POST'])
def analyze_video_and_audio():
    try:
        timeout = int(request.form.get('timeout', 10))
        video_filename = request.form.get('video_filename', 'output.mp4')

        # Function to process audio and put into the queue
        def process_audio(audio_queue):
            start_time = time.time()
            while (time.time() - start_time) < timeout:
                try:
                    with sr.Microphone() as source:
                        r.adjust_for_ambient_noise(source, duration=0.2)
                        audio = r.listen(source)
                        text = r.recognize_google(audio)
                        
                        # Detect language of the transcribed text
                        detected_language = detect(text)
                        print("Detected Language:", detected_language)
                        topic = "Self Introduction"

                        accuracy = calculate_accuracy(text, topic)
                        print("Accuracy:", accuracy)

                        audio_queue.put((text, detected_language, accuracy))

                except sr.RequestError as e:
                    print(f"Could not request result: {e}")
                except sr.UnknownValueError:
                    print("Unknown error occurred")

            print("Timeout reached for audio recording.")

        # Start audio processing in a separate thread
        audio_queue = queue.Queue()
        audio_thread = threading.Thread(target=process_audio, args=(audio_queue,))
        audio_thread.start()

        # Open default camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return jsonify({'error': 'Failed to open camera'}), 500

        print("Camera is opened successfully")

        # Define the codec and create VideoWriter object for MP4 format
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Using MP4V codec for MP4 format
        out = cv2.VideoWriter(video_filename, fourcc, 20.0, (640, 480))

        # Initialize MTCNN face detector
        mtcnn_detector = MTCNN()

        # Dictionary to store emotion counts and total confidence
        emotions_counter = {}

        # Start video recording
        start_time = time.time()
        while (time.time() - start_time) < timeout:
            ret, frame = cap.read()
            if not ret:
                return jsonify({'error': 'Failed to capture frame'}), 500

            # Write the frame into the video file
            out.write(frame)

            # Detect emotions in the frame
            detect_emotions(frame, emotions_counter, mtcnn_detector)

        # Wait for audio processing thread to finish
        audio_thread.join()

        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # Calculate overall dominant emotion and confidence
        if emotions_counter:
            most_common_emotion, _ = max(emotions_counter.items(), key=lambda x: x[1]['count'])
            overall_confidence = sum(data['total_confidence'] / data['count'] for data in emotions_counter.values() if 'count' in data and data['count'] > 0)

            # Get transcript, detected language, and accuracy from the audio_queue
            transcript, detected_language, accuracy = audio_queue.get()

            return jsonify({
                'most_common_emotion': most_common_emotion,
                'overall_confidence': min(overall_confidence, 100.0),
                'transcript': transcript,
                'detected_language': detected_language,
                'accuracy': accuracy
            })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000)
