import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.applications import ResNet50

# Definir las rutas de las imágenes
img_path_tank = 'TankshermanM4.jpg'
img_path_f16 = 'F-16_June_2008.jpg'
img_path_747 = '747.jpeg'
img_path_bmw1 = 'bmw1.jpg'

# Función para procesar una imagen dada su ruta
def procesar_imagen(img_path):
    # Cargar la imagen con el tamaño adecuado para el modelo
    img = image.load_img(img_path, target_size=(224, 224))
    # Convertir la imagen en un array
    img_array = image.img_to_array(img)
    # Expandir las dimensiones de la imagen para que se ajuste al formato esperado por el modelo
    img_expanded = np.expand_dims(img_array, axis=0)
    # Preprocesar la imagen de la misma manera que se preprocesaron las imágenes originales
    img_ready = preprocess_input(img_expanded)
    return img_ready

# Procesar las imágenes de los vehículos
img_ready_tank = procesar_imagen(img_path_tank)
img_ready_f16 = procesar_imagen(img_path_f16)
img_ready_747 = procesar_imagen(img_path_747)

# Procesar las imágenes de los hombres y el automóvil BMW
img_ready_bmw1 = procesar_imagen(img_path_bmw1)



# Instanciar un modelo ResNet50 con pesos 'imagenet'
model_tank = ResNet50(weights='imagenet')
model_f16 = ResNet50(weights='imagenet')
model_747 = ResNet50(weights='imagenet')


# Predecir con ResNet50 en las imágenes ya procesadas
preds_tank = model_tank.predict(img_ready_tank)
preds_f16 = model_f16.predict(img_ready_f16)
preds_747 = model_747.predict(img_ready_747)



# Instanciar un modelo ResNet50 con pesos 'imagenet' para las imágenes de los hombres y el automóvil BMW
model_bmw1 = ResNet50(weights='imagenet')
#model_blackM = ResNet50(weights='imagenet')
#model_whiteM = ResNet50(weights='imagenet')

# Predecir con ResNet50 en las imágenes ya procesadas
preds_bmw1 = model_bmw1.predict(img_ready_bmw1)




# Decodificar las primeras 5 predicciones para cada imagen
print('Predicciones para el Tanque Sherman M4:')
print(decode_predictions(preds_tank, top=1)[0])
print("*******")
print('Predicciones para el F-16:')
print(decode_predictions(preds_f16, top=1)[0])
print("*******")
print('Predicciones para 747:')
print(decode_predictions(preds_747, top=1)[0])
print("*******")
print('Predicciones para el BMW:')
print(decode_predictions(preds_bmw1, top=1)[0])
print("*******")


# Crear un DataFrame para almacenar los resultados
import pandas as pd

data = {
    "Foto": ["Tanque Sherman M4", "F-16", "747", "BMW"],
    "Predicción": [
        decode_predictions(preds_tank, top=1)[0][0][1],
        decode_predictions(preds_f16, top=1)[0][0][1],
        decode_predictions(preds_747, top=1)[0][0][1],
        decode_predictions(preds_bmw1, top=1)[0][0][1],
 
        
  
    ],
    "Probabilidad": [
        decode_predictions(preds_tank, top=1)[0][0][2],
        decode_predictions(preds_f16, top=1)[0][0][2],
        decode_predictions(preds_747, top=1)[0][0][2],
        decode_predictions(preds_bmw1, top=1)[0][0][2],


    ]
}

df = pd.DataFrame(data)

# Crear el gráfico de barras
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x="Foto", y="Probabilidad")
plt.title('Probabilidad de la predicción para cada foto')
plt.xlabel('Foto')
plt.ylabel('Probabilidad')
plt.savefig('Probabilidad de la predicción para cada foto.jpg')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
