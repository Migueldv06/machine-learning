import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

EPOCHS = 50     # Teste com valores maiores
IMG_HEIGHT = 224
IMG_WIDTH = 244
BATCH_SIZE = 32

# Augmentation para aumentar o dataset artificialmente
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 80% treino, 20% validação
)

# Carregar imagens
train_generator = datagen.flow_from_directory(
    "dataset",
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

validation_generator = datagen.flow_from_directory(
    "dataset",
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# Criar modelo CNN
model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1, activation='softmax')  # 3 classes: antes, meio, dentro
])

# Compilar modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Treinar modelo
model.fit(train_generator, validation_data=validation_generator, epochs=EPOCHS)

# Salvar modelo treinado
model.save("modelo_detecta_avião.h5")

# Imprimir as classes do dataset
print("Classes detectadas:", train_generator.class_indices)