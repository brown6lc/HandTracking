import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
import serial

# ---------------- Arduino Setup ----------------
arduino = serial.Serial('COM3', 9600, timeout=1)  # change COM3 to your Arduino port
time.sleep(2)

# ---------------- MediaPipe Setup ----------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def calc_angle(a, b, c):
    """Calculate angle (in degrees) between three points a-b-c."""
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle

# Define fingers with landmark triplets
FINGERS = {
    "Thumb":  [(1, 2, 3), (2, 3, 4)],
    "Index":  [(5, 6, 7), (6, 7, 8)],
    "Middle": [(9, 10, 11), (10, 11, 12)],
    "Ring":   [(13, 14, 15), (14, 15, 16)],
    "Pinky":  [(17, 18, 19), (18, 19, 20)],
}

cap = cv2.VideoCapture(0)

# Set capture resolution to 1024x768
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)

# Storage for CSV data
data_records = []

with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        # Ensure frame is always 1024x768
        frame = cv2.resize(frame, (1024, 768))
        h, w, _ = frame.shape
        frame_time = time.time()

        frame_record = {"timestamp": frame_time}

        if result.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
                # Draw landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                handedness_label = result.multi_handedness[hand_idx].classification[0].label

                # Text position: left or right
                x_pos = 10 if handedness_label == "Left" else w - 350
                y_offset = 40

                print(f"\n--- {handedness_label} Hand ---")

                bent_count = 0  # count bent joints for gesture detection

                for finger_name, joints in FINGERS.items():
                    for j, (a, b, c) in enumerate(joints):
                        angle = calc_angle(hand_landmarks.landmark[a],
                                           hand_landmarks.landmark[b],
                                           hand_landmarks.landmark[c])
                        angle_int = int(angle)

                        # Save value to record
                        frame_record[f"{handedness_label}_{finger_name}{j+1}"] = angle_int

                        # Print to console
                        print(f"{handedness_label} {finger_name}{j+1}: {angle_int}")

                        # Draw on frame
                        cv2.putText(frame,
                                    f"{handedness_label} {finger_name}{j+1}: {angle_int}",
                                    (x_pos, y_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (0, 255, 0), 2)
                        y_offset += 25

                        # Count if bent
                        if angle_int < 60:
                            bent_count += 1

                # ---------------- Servo State Logic ----------------
                if bent_count >= 7:       # most joints bent = fist
                    servo_angle = 180
                    gesture = "Closed Fist"
                elif bent_count <= 2:     # most joints straight = open hand
                    servo_angle = 0
                    gesture = "Open Hand"
                else:                     # in between
                    servo_angle = 90
                    gesture = "Half Closed"

                # Send to Arduino
                arduino.write(f"{servo_angle}\n".encode())
                print(f"Gesture: {gesture}, Servo -> {servo_angle}")

                # Draw gesture label
                cv2.putText(frame, f"{gesture}: {servo_angle} deg",
                            (x_pos, y_offset + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 255), 2)

        data_records.append(frame_record)

        cv2.imshow("Two-Hand Tracking with Angles", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.getWindowProperty("Two-Hand Tracking with Angles", cv2.WND_PROP_VISIBLE) < 1:
            break

cap.release()
arduino.close()
cv2.destroyAllWindows()

# Save to CSV
df = pd.DataFrame(data_records)
df.to_csv(r"C:\Users\lando\Documents\VSCODE-Dir\.Proj\Handtracking\hand_tracking_angles.csv", index=False)
print("Saved angles to hand_tracking_angles.csv")
