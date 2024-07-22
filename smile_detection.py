import cv2

def detect_smiles(video_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

    video_capture = cv2.VideoCapture(video_path)
    total_smiles = 0
    prev_smile_status = False

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        smile_detected = False
        
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]

            smiles = smile_cascade.detectMultiScale(roi_gray, 1.3, 5)
            if len(smiles) > 0:
                smile_detected = True
        
        if smile_detected and not prev_smile_status:
            total_smiles += 1

        prev_smile_status = smile_detected
        
        # # Display the frame
        # cv2.imshow('Frame', frame)
        
        # # Break the loop if 'q' is pressed
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    video_capture.release()
    cv2.destroyAllWindows()
    return total_smiles

