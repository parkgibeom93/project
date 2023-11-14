import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 데이터 경로 설정 (본인의 데이터 경로로 수정)
train_data_dir = 'path/to/your/train_data'
validation_data_dir = 'path/to/your/validation_data'

# 이미지 데이터 생성기 설정
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

validation_datagen = ImageDataGenerator(rescale=1./255)

# 학습 데이터 로드 및 전처리
train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    target_size=(150, 150),
                                                    batch_size=32,
                                                    class_mode='binary')

# 검증 데이터 로드 및 전처리
validation_generator = validation_datagen.flow_from_directory(validation_data_dir,
                                                              target_size=(150, 150),
                                                              batch_size=32,
                                                              class_mode='binary')

# 모델 구성
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# 모델 컴파일
model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
              metrics=['accuracy'])

# 모델 학습
model.fit(train_generator,
          steps_per_epoch=train_generator.samples // train_generator.batch_size,
          epochs=50,
          validation_data=validation_generator,
          validation_steps=validation_generator.samples // validation_generator.batch_size)

# 모델 저장 (본인이 원하는 경로로 수정)
model.save('path/to/your/pet_bottle_label_classification_model.h5')
