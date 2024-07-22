import cv2
import mediapipe as mp
import numpy as np

def initialize_face_mesh():
    mp_face_mesh = mp.solutions.face_mesh
    return mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))

def is_looking_at_camera(landmarks):
    left_eye_outer = landmarks[33]
    right_eye_outer = landmarks[263]
    nose_tip = landmarks[1]

    left_eye_distance = euclidean_distance(left_eye_outer, nose_tip)
    right_eye_distance = euclidean_distance(right_eye_outer, nose_tip)

    ratio = left_eye_distance / right_eye_distance
    return 0.9 < ratio < 1.1

def detect_gaze(video_path):
    face_mesh = initialize_face_mesh()
    cap = cv2.VideoCapture(video_path)

    tatapan_menghadap_kamera = 0
    was_looking_at_camera = False

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = [(lm.x * frame.shape[1], lm.y * frame.shape[0], lm.z) for lm in face_landmarks.landmark]
                is_looking = is_looking_at_camera(landmarks)
                
                if is_looking and not was_looking_at_camera:
                    tatapan_menghadap_kamera += 1
                    cv2.putText(frame, 'Looking at Camera', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
                was_looking_at_camera = is_looking

                for landmark in landmarks:
                    cv2.circle(frame, (int(landmark[0]), int(landmark[1])), 1, (0, 255, 0), -1)


        # cv2.imshow('Frame', frame)

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    cv2.destroyAllWindows()

    return tatapan_menghadap_kamera
