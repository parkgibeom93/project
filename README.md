# project
import cv2
import tensorflow.keras
import numpy as np
import serial
import time

capture = cv2.VideoCapture(0)
ans = ['non_label', 'label']

ser = serial.Serial('COM4', 9600)
# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
size = (224, 224)
img_num = 1

while True:
    if ser.readable():
        val = ser.readline()
        flag = int(val.decode()[:len(val) - 1])
        
        if flag == 1:

            print("Trash detected")
            ret, frame = capture.read()
            print("캡쳐")
            cv2.imwrite("temp.jpg", frame)
            image = cv2.resize(frame, dsize=size, interpolation=cv2.INTER_AREA)
            # Normalize the image
            normalized_image_array = (image.astype(np.float32) / 127.0) - 1
            # Load the image into the array
            data[0] = normalized_image_array
            # run the inference
            prediction = model.predict(data)
            index = np.argmax(prediction)
            print(prediction)
            cv2.imwrite(str(ans[index]) + str(img_num) + ".jpg", image)
            img_num = img_num + 1
            
            if index == 0:
                send = '2'
                send = send.encode('utf-8')
                ser.write(send)
                print(ans[index])
                time.sleep(0.5)

            elif index == 1:
                send = '1'
                send = send.encode('utf-8')
                ser.write(send)
                print(ans[index])
                time.sleep(0.5)
            flag = 0



capture.release()
cv2.destroyAllWindows()
