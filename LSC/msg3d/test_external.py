import torch
import numpy as np
from model.msg3d import Model  # Importa la arquitectura usada en el entrenamiento
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import import_class, count_params
from model.ms_gcn import MultiScale_GraphConv as MS_GCN
from model.ms_tcn import MultiScale_TemporalConv as MS_TCN
from model.ms_gtcn import SpatialTemporal_MS_GCN, UnfoldTemporalWindows
from model.mlp import MLP
from model.activation import activation_factory


# Configurar paths
MODEL_PATH = "work_dir/TRAIN_TEST/E8_pre_juntos_105/42/joints_C3_xyc-T1/weights/weights-36.pt"
NEW_VIDEO_NPY = "/home/naiara/Documentos/GitHub/SWL-LSE/mediapipe_keypoints/src/features_output/joints_C3_xyc/LSC - Ocell [XgvobAcMsMk].npy"

# Cargar modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(
        num_class=105,
        num_point=61,
        num_person=1,
        num_gcn_scales=8,
        num_g3d_scales=8,
        graph='graph.mediapipe61.AdjMatrixGraph'
        
    
    )
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)

model.eval()

# Cargar los datos del nuevo video
data = np.load(NEW_VIDEO_NPY)  # (12, 543, 2)
data = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device)  # Agregar batch dimension

# Hacer la predicción
with torch.no_grad():
    output = model(data)  # Output sin aplicar softmax

# Aplicar softmax para obtener probabilidades
probabilities = torch.softmax(output, dim=1)

# Obtener las 5 clases con mayor probabilidad
top5_prob, top5_classes = torch.topk(probabilities, 5, dim=1)

# Convertir a listas de Python para imprimir
top5_prob = top5_prob.squeeze().tolist()
top5_classes = top5_classes.squeeze().tolist()

# Mostrar resultados
print("Top 5 clases más probables:")
for i in range(5):
    print(f"Clase {top5_classes[i]} - Probabilidad: {top5_prob[i]:.4f}")
