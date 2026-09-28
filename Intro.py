# Import OpenCV
import cv2
import numpy as np

# 1. Read image
img = cv2.imread("canon.png")

# 2. Show Image
cv2.imshow("Original Image", img)

# 3. Filter Color Image
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # Konversi BGR-HSV 

# Filter Range Warna Merah-Kuning
lower_end = np.array([0, 0, 0])
upper_end = np.array([35, 255, 255])

# Mask
mask_img = cv2.inRange(hsv_img, lower_end, upper_end)
filtered_img = cv2.bitwise_and(img, img, mask=mask_img)
    
cv2.imshow('Filtered Color', filtered_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 4. Filter Color Video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break   
    hsv_video = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # Konversi BGR-HSV
    
    # Mask
    mask_video = cv2.inRange(hsv_video, lower_end, upper_end)
    filtered_video = cv2.bitwise_and(frame, frame, mask=mask_video)
    
    # Show Video
    cv2.imshow("Video Original", frame)
    cv2.imshow("Video Filtered Color", filtered_video)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
