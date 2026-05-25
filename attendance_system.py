from deepface import DeepFace
import cv2
import pandas as pd
from datetime import datetime
import os
import winsound

# =========================
# SETTINGS
# =========================

DATABASE_PATH = "database"
ATTENDANCE_FILE = "attendance.csv"

# =========================
# CREATE CSV IF NOT EXISTS
# =========================

if not os.path.exists(ATTENDANCE_FILE):
    df = pd.DataFrame(columns=["Name", "Time", "Date"])
    df.to_csv(ATTENDANCE_FILE, index=False)

# =========================
# PREVENT DUPLICATE ENTRY
# =========================

marked_names = set()

# =========================
# START CAMERA
# =========================

cam = cv2.VideoCapture("http://10.24.202.195:8080/video")

print("Attendance System Started")
print("Press Q to Quit")

# =========================
# MAIN LOOP
# =========================

while True:

    ret, frame = cam.read()

    if not ret:
        break

    # =========================
    # FACE DETECTION
    # =========================

    try:

        faces = DeepFace.extract_faces(
            img_path=frame,
            detector_backend='opencv',
            enforce_detection=False
        )

        for face in faces:

            facial_area = face['facial_area']

            x = facial_area['x']
            y = facial_area['y']
            w = facial_area['w']
            h = facial_area['h']

            # Yellow rectangle while detecting
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 255),
                2
            )

            # Save temporary image
            temp_face = frame[y:y+h, x:x+w]

            cv2.imwrite("temp.jpg", temp_face)

            try:

                # =========================
                # FACE RECOGNITION
                # =========================

                result = DeepFace.find(
                    img_path="temp.jpg",
                    db_path=DATABASE_PATH,
                    enforce_detection=False,
                    silent=True
                )

                if len(result) > 0 and not result[0].empty:

                    identity_path = result[0]['identity'][0]

                    name = identity_path.split("\\")[1]

                    # Green rectangle if recognized
                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x + w, y + h),
                        (0, 255, 0),
                        3
                    )

                    # Display name
                    cv2.putText(
                        frame,
                        name,
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2
                    )

                    # =========================
                    # MARK ATTENDANCE
                    # =========================

                    if name not in marked_names:

                        now = datetime.now()

                        current_time = now.strftime("%H:%M:%S")
                        current_date = now.strftime("%d-%m-%Y")

                        new_data = pd.DataFrame({
                            "Name": [name],
                            "Time": [current_time],
                            "Date": [current_date]
                        })

                        new_data.to_csv(
                            ATTENDANCE_FILE,
                            mode='a',
                            header=False,
                            index=False
                        )

                        marked_names.add(name)

                        # =========================
                        # BEEP SOUND
                        # =========================

                        winsound.Beep(1000, 500)

                        print(f"Attendance Marked for {name}")

            except Exception as e:
                print("Recognition Error:", e)

    except Exception as e:
        print("Detection Error:", e)

    # =========================
    # SHOW WINDOW
    # =========================

    cv2.imshow("Attendance Detection System", frame)

    # =========================
    # EXIT
    # =========================

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

# =========================
# CLEANUP
# =========================

cam.release()
cv2.destroyAllWindows()