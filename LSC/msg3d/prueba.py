import numpy as np

# Carga el archivo
data_path = "/home/naiara/Documentos/GitHub/SWL-LSE/mediapipe_keypoints/src/generate_dataset/train_joints_C3_xyc.npy"
data_path = "/home/naiara/Documentos/GitHub/SWL-LSE/mediapipe_keypoints/src/generate_arr_keypoints/VOLER_5_SK7.npy"
data = np.load(data_path, allow_pickle=True)

# Imprime información sobre el array
print("Shape del array:", data.shape)
print("Tamaño total:", data.size)

# Intenta acceder al índice 5630
try:
    if data.size > 0 and 5630 < len(data):
        print("Datos en índice 5630:", data[5630])
        print("Shape de los datos en 5630:", data[5630].shape)
    else:
        print("El índice 5630 está fuera de los límites")
        print("Tamaño máximo del array:", len(data) if data.size > 0 else 0)
except Exception as e:
    print("Error al acceder a los datos:", str(e))

# Si quieres ver los primeros elementos para verificar la estructura
if data.size > 0:
    print("\nPrimeros elementos:")
    print(data[0].shape if data.size > 0 else "Array vacío")