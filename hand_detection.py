import cv2
import mediapipe as mp
from google.protobuf.json_format import MessageToDict

def detect_hands(video_path):
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.75,
        min_tracking_confidence=0.75,
        max_num_hands=2)

    cap = cv2.VideoCapture(video_path)

    hand_count = 0
    prev_hand_status = False

    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        current_hand_status = False
        if results.multi_hand_landmarks:
            current_hand_status = True
            if len(results.multi_handedness) == 2:
                cv2.putText(img, 'Both Hands', (250, 50),
                            cv2.FONT_HERSHEY_COMPLEX, 0.9,
                            (0, 255, 0), 2)
            else:
                for i in results.multi_handedness:
                    label = MessageToDict(i)['classification'][0]['label']

                    if label == 'Left':
                        cv2.putText(img, label + ' Hand', (20, 50),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.9,
                                    (0, 255, 0), 2)

                    if label == 'Right':
                        cv2.putText(img, label + ' Hand', (460, 50),
                                    cv2.FONT_HERSHEY_COMPLEX,
                                    0.9, (0, 255, 0), 2)

        if current_hand_status and not prev_hand_status:
            hand_count += 1

        prev_hand_status = current_hand_status

       
        cv2.imshow('Frame', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return hand_count

print(detect_hands("video.mp4"))