import os
import sys
import numpy as np
from skimage.io import imread
from skimage import img_as_float
from mrcnn.config import Config
from mrcnn import model as modellib, utils
import tensorflow as tf
import gc
from imgaug import augmenters as iaa 

# Configuração para melhorar o desempenho em CPUs multicore
tf_config = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=8,
    inter_op_parallelism_threads=2,
    allow_soft_placement=True,
    device_count={'CPU': 1}
)
tf.compat.v1.Session(config=tf_config)

# Root directory of the project
ROOT_DIR = "C:/Users/anafl/Códigos/Mask-R-CNN-main"

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

class CustomConfig(Config):
    NAME = "object"
    IMAGES_PER_GPU = 2
    NUM_CLASSES = 1 + 1 # fundo + objeto 
    STEPS_PER_EPOCH = 22 # Número de passos por época
    DETECTION_MIN_CONFIDENCE = 0.7 # Confiança mínima para considerar detecções

class CustomDataset(utils.Dataset):
    def load_custom(self, dataset_dir, subset):
        self.add_class("object", 1, "atroporpureum") # Adiciona a classe de objeto
        image_dir = os.path.join(dataset_dir, subset, "images")
        mask_dir = os.path.join(dataset_dir, subset, "masks")

        image_files = os.listdir(image_dir)
        mask_files = os.listdir(mask_dir)
        
        print(f"Found {len(image_files)} images in {image_dir}")
        print(f"Found {len(mask_files)} masks in {mask_dir}")

        for image_file in image_files:
            if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                print(f"Skipping non-image file: {image_file}")
                continue

            image_path = os.path.join(image_dir, image_file)
            possible_mask_file = image_file.replace(".jpg", ".png").replace(".jpeg", ".png").replace(".png", "_mascara.png")
            mask_path = os.path.join(mask_dir, possible_mask_file)
            if os.path.exists(mask_path):
                print(f"Using image for training: {image_path}")
                self.add_image(
                    "object",
                    image_id=image_file,
                    path=image_path,
                    mask_path=mask_path
                )
            else:
                print(f"Mask file not found for image: {image_file}")

        print(f"Loaded {len(self.image_info)} images")

    # Carrega uma imagem pelo seu ID
    def load_image(self, image_id):
        info = self.image_info[image_id]
        image = imread(info['path'])
        image = img_as_float(image).astype(np.float32)  # converte para o formato de ponto flutuante float
        return image

    # Carrega uma máscara pelo ID da imagem
    def load_mask(self, image_id):
        info = self.image_info[image_id]
        mask_path = info['mask_path']
        mask = imread(mask_path).astype(np.bool_) # True indica a presença do objeto e False o fundo)
        mask = np.expand_dims(mask, axis=-1) # adiciona uma nova dimensão à matriz da máscara (necessária para compatibilidade com o modelo Mask R-CNN)
        class_ids = np.ones([mask.shape[-1]], dtype=np.int32)
        return mask, class_ids

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "object":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

def train(model): 
    dataset_train = CustomDataset()
    dataset_train.load_custom(os.path.join(ROOT_DIR, "dataset"), "train")
    dataset_train.prepare()

    dataset_val = CustomDataset()
    dataset_val.load_custom(os.path.join(ROOT_DIR, "dataset"), "val")
    dataset_val.prepare()

    if len(dataset_train.image_info) == 0:
        raise ValueError("Nenhuma imagem de treinamento carregada.")
    if len(dataset_val.image_info) == 0:
        raise ValueError("Nenhuma imagem de validação carregada.")
    
    # Definindo a sequência data argumentation
    augmentation = iaa.Sequential([
        iaa.Fliplr(0.5),  # 50% de chance de inverter horizontalmente
        iaa.Flipud(0.2),  # 20% de chance de inverter verticalmente
        iaa.Multiply((0.7, 1.3)),  # Alterar o brilho
        iaa.ContrastNormalization((0.75, 1.5)),  # Ajustar o contraste
        iaa.Affine(scale=(0.8, 1.2)),  # Escalar as imagens aleatoriamente
    ])

    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=10,
                layers='heads',
                augmentation=augmentation)  # Passar a sequência de aumentação aqui

    # Limpar a memória
    tf.keras.backend.clear_session()
    gc.collect()

# Inicializa a configuração e o modelo
config = CustomConfig()
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=DEFAULT_LOGS_DIR)

# Carrega os pesos iniciais se disponíveis
weights_path = COCO_WEIGHTS_PATH
if not os.path.exists(weights_path):
    utils.download_trained_weights(weights_path)

model.load_weights(weights_path, by_name=True, exclude=[
    "mrcnn_class_logits", "mrcnn_bbox_fc",
    "mrcnn_bbox", "mrcnn_mask"])

# Chama a função de treinamento
train(model)
