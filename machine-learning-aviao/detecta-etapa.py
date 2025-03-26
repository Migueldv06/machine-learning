import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Carregar modelo treinado
model = load_model("modelo_detecta_aviao.h5")

# Mapeamento das classes
classes = ["airplanes"]

def classify_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (393, 187))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    if img is None:
        print(f"Erro: Não foi possível carregar a imagem '{img_path}'. Verifique o caminho do arquivo.")
        return
    else:
        print(f"\n\n o Arquivo carregou corretamente '{img_path}'.\n\n")

    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    
    if classes[class_index] == "airplanes":
        print(f"A imagem tem um avião")
    else: 
        print(f"a imagem não tem avião")

    print(f"A imagem é: {classes[class_index]}")

# Teste com uma imagem nova
classify_image("image_0004.png")
