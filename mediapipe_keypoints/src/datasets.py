import os
import pandas as pd
from sklearn.model_selection import train_test_split
import re

# Ruta de los videos
video_dir = 'elegidoss'

# Listar los archivos .mp4
videos = [f for f in os.listdir(video_dir) if f.endswith('.pkl')]
print(len(videos))
# Extraer las clases (lo que aparece antes del primer '_')
classes = []
for video in videos:
    nombre_sin_extension = os.path.splitext(video)[0]  # Nombre sin la extensión
    base_name = nombre_sin_extension.split('_')[0]  # Parte antes del primer '_'

    # Encontrar todas las secuencias de letras mayúsculas
    mayusculas = re.findall(r'[A-ZÀ-Ú]+', base_name)

    # Obtener la secuencia más larga
    ma = max(mayusculas, key=len) if mayusculas else base_name
    classes.append(ma)



   
# Crear un mapeo de clases a números
class_names = sorted(set(classes))  # Lista de clases únicas

print("cantidad de clases ",len(class_names))
class_mapping = {name: idx + 1 for idx, name in enumerate(class_names)}  # Asignar un número a cada clase
for i in class_mapping:
    print(i)

# Asignar los números de clase a los videos
#video_class = [(video.split('.')[0], class_mapping[cls]) for video, cls in zip(videos, classes)]

# Crear un DataFrame
#df = pd.DataFrame(video_class)

# Dividir los datos en train, test, y val
#train_df, test_val_df = train_test_split(df, test_size=0.3, stratify=df[1])
#test_df, val_df = train_test_split(test_val_df, test_size=0.5, stratify=test_val_df[1])

# Guardar los CSVs
#train_df.to_csv('train_labels.csv', header=False,index=False)
#test_df.to_csv('test_labels.csv', header=False,index=False)
#val_df.to_csv('val_labels.csv',header=False, index=False)