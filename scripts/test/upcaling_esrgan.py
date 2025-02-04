import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import time
import numpy as np
import cv2

# ================================
#       MODELO ESRGAN
# ================================
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))

class ESRGAN(nn.Module):
    def __init__(self):
        super(ESRGAN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(8)])
        self.conv2 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        return self.conv2(self.res_blocks(self.relu(self.conv1(x))))

# ================================
#       FUNÇÕES AUXILIARES
# ================================
def apply_post_processing(image):
    """
    Aplica melhorias visuais na imagem upscaleada.
    """
    image = np.array(image)
    
    # Anti-Aliasing
    image = cv2.GaussianBlur(image, (3, 3), 0)
    
    # Detecção de bordas para sombras e realces
    edges = cv2.Canny(image, 100, 200)
    image = cv2.addWeighted(image, 0.9, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), 0.1, 0)
    
    # Simulação de reflexos em água e espelhos
    reflection_mask = detect_reflective_surfaces(image)
    image[reflection_mask] = cv2.addWeighted(image[reflection_mask], 0.5, cv2.flip(image[reflection_mask], 1), 0.5, 0)
    
    return Image.fromarray(image)

def detect_reflective_surfaces(image):
    """
    Método simples para detectar áreas reflexivas (espelhos, água, etc.).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
    return mask.astype(bool)

# ================================
#       CONFIGURAÇÃO
# ================================
input_image_path = "/mnt/d/Desenv/Python/frames/Dream/Data/Images/Test/input_image.png"
output_image_path = "/mnt/d/Desenv/Python/frames/Dream/Data/Images/Upscaling/output_image_04_02.png"
model_path = "/mnt/d/Desenv/Python/frames/Dream/models/esrgan_adversarial.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carregar modelo treinado
generator = ESRGAN().to(device)
generator.load_state_dict(torch.load(model_path, map_location=device))
generator.eval()

# Transformação de entrada
transform = transforms.Compose([
    transforms.Resize((2048, 1536)),  # Redimensionar para a resolução desejada
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ================================
#       PROCESSAMENTO
# ================================
print("Carregando imagem de entrada...")
image = Image.open(input_image_path).convert("RGB")
tensor_image = transform(image).unsqueeze(0).to(device)

print("Iniciando upscaling...")
start_time = time.time()

with torch.no_grad():
    upscale_tensor = generator(tensor_image)
    upscale_tensor = (upscale_tensor * 0.5 + 0.5).clamp(0, 1)  # Desnormalização
    upscale_image = transforms.ToPILImage()(upscale_tensor.squeeze(0).cpu())

# Pós-processamento
upscale_image = apply_post_processing(upscale_image)

# Salvar saída
upscale_image.save(output_image_path)

end_time = time.time()
print(f"Upscaling concluído em {end_time - start_time:.2f} segundos")
