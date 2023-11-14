import cv2
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# MobileNetV2 모델 로드 (이미 사전 훈련된 가중치 포함)
classification_model = MobileNetV2(weights="imagenet")

# 카메라 초기화
cap = cv2.VideoCapture(0)

# 모델 로드 (본인이 학습한 모델 경로로 수정)
pet_bottle_model = load_model('path/to/your/pet_bottle_label_classification_model.h5')

while True:
    # 프레임 읽기
    ret, frame = cap.read()

    # 이미지 전처리
    frame = cv2.resize(frame, (224, 224))
    frame_for_classification = img_to_array(frame)
    frame_for_classification = preprocess_input(frame_for_classification)
    frame_for_classification = np.expand_dims(frame_for_classification, axis=0)

    # 플라스틱 병 라벨 분류
    classification_predictions = classification_model.predict(frame_for_classification)
    classification_label = decode_predictions(classification_predictions)
    classification_label = classification_label[0][0][1]

    # 플라스틱 병 라벨에 따라 예측
    if classification_label == 'plastic_bottle':
        # 플라스틱 병 라벨이면 딥러닝 모델로 예측
        classification_for_pet_bottle = img_to_array(frame)
        classification_for_pet_bottle = np.expand_dims(classification_for_pet_bottle, axis=0)
        pet_bottle_prediction = pet_bottle_model.predict(classification_for_pet_bottle)

        # 예측 결과 디코딩
        pet_bottle_label = "Plastic Bottle" if pet_bottle_prediction[0][0] > 0.5 else "Not a Plastic Bottle"

        # 화면에 표시
        cv2.putText(frame, f"Label: {pet_bottle_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 화면에 플라스틱 병 라벨 표시
    cv2.putText(frame, f"Plastic Bottle Label: {classification_label}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 화면에 표시
    cv2.imshow("Plastic Bottle Label Classifier", frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 카메라 해제 및 윈도우 닫기
cap.release()
cv2.destroyAllWindows()
