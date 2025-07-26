
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import cv2

# Load trained model
mymodel = load_model('mymodel.h5')

# Load face detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load image
img_path = 'testimg.png'  # Change to your image file path
img = cv2.imread(img_path)

# Resize if image is too large (optional: to 800x800 max)
MAX_DIM = 800
h, w = img.shape[:2]
if max(h, w) > MAX_DIM:
    scaling_factor = MAX_DIM / float(max(h, w))
    img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

orig = img.copy()

# Convert to grayscale for face detection
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

if len(faces) == 0:
    print("❌ No faces detected in the image.")
else:
    for (x, y, w, h) in faces:
        face_img = img[y:y+h, x:x+w]

        # Resize the face to match model input
        face_resized = cv2.resize(face_img, (150, 150))
        test_image = image.img_to_array(face_resized)
        test_image = np.expand_dims(test_image, axis=0)
        pred = mymodel.predict(test_image)[0][0]

        if pred >= 0.5:
            label = 'NO MASK'
            color = (0, 0, 255)
        else:
            label = 'MASK'
            color = (0, 255, 0)

        # Draw rectangle and label
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
        cv2.putText(img, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Show final image with predictions
    cv2.imshow('Mask Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
