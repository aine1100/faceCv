import cv2
import time
import math

# ---------- CONFIGURATION ----------
KNOWN_FACE_WIDTH_CM = 16.0       # average adult face width (cheek-to-cheek)
FOCAL_LENGTH_PX = 650.0          # approximate webcam focal length (can adjust)
SMOOTHING_ALPHA = 0.4            # distance smoothing
TOO_CLOSE_CM = 40                # alert threshold
# ----------------------------------

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

prev_center = None
prev_time = time.time()
direction = "Center"
speed_px_s = 0.0
speed_cm_s = 0.0
smoothed_distance_cm = None

print("[INFO] Auto Face Direction + Distance Tracker Started (press 'q' to quit)")

def estimate_distance_cm(face_pixel_width, focal_len_px):
    """Estimate distance from camera based on apparent face width."""
    if face_pixel_width == 0:
        return 0
    return (KNOWN_FACE_WIDTH_CM * focal_len_px) / face_pixel_width

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Camera not accessible.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # pick the largest detected face (closest to camera)
    primary_face = None
    max_area = 0
    for (x, y, w, h) in faces:
        if w * h > max_area:
            max_area = w * h
            primary_face = (x, y, w, h)

    if primary_face:
        x, y, w, h = primary_face
        cx, cy = x + w // 2, y + h // 2

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)

        # --- compute direction and speed ---
        current_time = time.time()
        dt = current_time - prev_time if prev_time else 1e-6
        if prev_center:
            dx = cx - prev_center[0]
            dy = cy - prev_center[1]
            if abs(dx) > abs(dy) and abs(dx) > 2:
                direction = "Right" if dx > 0 else "Left"
            elif abs(dy) > abs(dx) and abs(dy) > 2:
                direction = "Down" if dy > 0 else "Up"
            else:
                direction = "Center"

            distance_px = math.hypot(dx, dy)
            speed_px_s = distance_px / dt
        prev_center = (cx, cy)
        prev_time = current_time

        # --- compute distance and cm/s ---
        distance_cm = estimate_distance_cm(w, FOCAL_LENGTH_PX)
        if smoothed_distance_cm is None:
            smoothed_distance_cm = distance_cm
        else:
            smoothed_distance_cm = (SMOOTHING_ALPHA * distance_cm +
                                    (1 - SMOOTHING_ALPHA) * smoothed_distance_cm)

        pixel_to_cm = KNOWN_FACE_WIDTH_CM / w  # approx cm per pixel
        speed_cm_s = speed_px_s * pixel_to_cm

        # --- display info on screen ---
        cv2.putText(frame, f"Direction: {direction}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Speed: {speed_cm_s:.2f} cm/s", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, f"Distance: {smoothed_distance_cm:.1f} cm", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # --- warning if too close ---
        if smoothed_distance_cm < TOO_CLOSE_CM:
            cv2.putText(frame, "TOO CLOSE!", (20, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    else:
        cv2.putText(frame, "No face detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # show frame
    cv2.imshow("Auto Face Direction & Distance Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
