import torch

def check_gpu():
    print("=== Diagnóstico da GPU com PyTorch ===")
    
    # Verifica se o PyTorch detecta a GPU
    gpu_available = torch.cuda.is_available()
    print(f"CUDA disponível: {gpu_available}")
    
    if not gpu_available:
        print("ERRO: CUDA não está disponível. Verifique sua instalação do PyTorch e drivers da GPU.")
        return
    
    # Obtém informações básicas da GPU
    device_id = torch.cuda.current_device()
    print(f"Dispositivo atual: {device_id}")
    print(f"Nome da GPU: {torch.cuda.get_device_name(device_id)}")
    print(f"Memória total da GPU: {torch.cuda.get_device_properties(device_id).total_memory / 1e9:.2f} GB")
    
    # Verifica o uso de memória atual na GPU
    print(f"Memória usada antes de alocar tensor: {torch.cuda.memory_allocated(device_id) / 1e9:.2f} GB")
    
    try:
        # Testa a alocação de um pequeno tensor na GPU
        x = torch.randn(1, 3, 8, 8).to("cuda")
        print("✅ Tensor pequeno movido para a GPU com sucesso!")
        
        # Testa a alocação de um tensor maior
        y = torch.randn(1, 3, 64, 64).to("cuda")
        print("✅ Tensor grande movido para a GPU com sucesso!")
    except Exception as e:
        print(f"❌ ERRO ao mover tensor para GPU: {e}")
    
    # Exibe o uso de memória após a alocação dos tensores
    print(f"Memória usada após alocação: {torch.cuda.memory_allocated(device_id) / 1e9:.2f} GB")
    
    # Limpa cache da GPU
    torch.cuda.empty_cache()
    print("🗑 Cache da GPU liberada!")

if __name__ == "__main__":
    check_gpu()
