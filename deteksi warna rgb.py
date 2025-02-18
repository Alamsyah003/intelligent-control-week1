import cv2
import numpy as np

# Inisialisasi kamera
cap = cv2.VideoCapture(1)

def detect_color(frame, lower_bound, upper_bound, color_name, box_color):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    best_contour = None
    best_accuracy = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:  # Filter area minimum untuk menghindari noise
            x, y, w, h = cv2.boundingRect(cnt)
            accuracy = (area / (w * h)) * 100  # Perhitungan akurasi sederhana
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_contour = (x, y, w, h)
    
    if best_contour:
        x, y, w, h = best_contour
        cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
        text = f"{color_name}: {best_accuracy:.2f}%"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Rentang warna dalam HSV
    colors = [
        (np.array([0, 120, 70]), np.array([10, 255, 255]), "Red", (0, 0, 255)),
        (np.array([36, 100, 100]), np.array([86, 255, 255]), "Green", (0, 255, 0)),
        (np.array([100, 150, 0]), np.array([140, 255, 255]), "Blue", (255, 0, 0)),
    ]

    for lower, upper, name, color in colors:
        detect_color(frame, lower, upper, name, color)

    cv2.imshow("Color Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
