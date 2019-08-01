import cv2
import sys

image = cv2.imread('images/spainfc.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faceCascade = cv2.CascadeClassifier('open_cv/haarcascade_frontalface_default.xml')
faces = faceCascade.detectMultiScale(gray, 1.2 , 5)

print("[ALERT] Found {0} Faces.".format(len(faces)))

for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0,0,255), 2)
    roi_color = image[y:y + h, x:x + w]
    print("[INFO] Faces Detected -- Will be saved locally.")
    cv2.imwrite('scraped_images/spainfc/'+str(w) + str(h)+ '_faces.jpg', roi_color)
    
status = cv2.imwrite('faces_detected.jpg', image)
print("[ALERT] Image faces_detected.jpg written to filesystem: ", status)