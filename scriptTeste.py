import os
import sys
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from mrcnn.config import Config
from mrcnn import model as modellib
import matplotlib.pyplot as plt
from mrcnn import visualize

# Configuração para inferência
class InferenceConfig(Config):
    NAME = "object_detection"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # Background + Número de classes
    DETECTION_MIN_CONFIDENCE = 0.7 # Confiança mínima para detecção

# Diretório raiz do projeto
ROOT_DIR = "C:/Users/anafl/Códigos/Mask-R-CNN-main"

# Caminho para o arquivo de pesos pré-treinado
MODEL_WEIGHTS_PATH = os.path.join(ROOT_DIR, "logs/object20240830T1508/mask_rcnn_object_0003.h5")

# Inicializando configuração e modelo
config = InferenceConfig()
model = modellib.MaskRCNN(mode="inference", config=config, model_dir=ROOT_DIR)

# Carregando os pesos pré-treinados
model.load_weights(MODEL_WEIGHTS_PATH, by_name=True)

# Carregando a imagem para inferência
image_path = os.path.join(ROOT_DIR, "Dataset/train/images/Cópia de Cópia de IMG_20211229_110514.jpg")
image = imread(image_path)
image_resized = resize(image, (config.IMAGE_MIN_DIM, config.IMAGE_MIN_DIM), preserve_range=True).astype(np.uint8)

# Fazendo a inferência
results = model.detect([image_resized], verbose=1)
r = results[0]  # Resultados da primeira imagem

# Mostrando os resultados
visualize.display_instances(image_resized, r['rois'], r['masks'], r['class_ids'], 
                            ['BG', 'object'], r['scores'], figsize=(5,5))

plt.show()
