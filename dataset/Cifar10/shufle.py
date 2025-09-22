import numpy as np
import os

# Defina o diretório onde estão os arquivos .npz
diretorio = 'train_mal'

# Lista de todos os arquivos .npz na pasta
arquivos = [f for f in os.listdir(diretorio) if f.endswith('.npz')]

# Para cada arquivo .npz na pasta
for arquivo in arquivos:
    # Carregar o arquivo com allow_pickle=True
    caminho_arquivo = os.path.join(diretorio, arquivo)
    dados = np.load(caminho_arquivo, allow_pickle=True)  # Habilita o pickle
    
    # Acesse a chave 'data'
    dados_data = dados['data']  # 'data' é um array NumPy
    
    print(f"Processando o arquivo: {arquivo}")
    print(f"Tipo de 'data': {type(dados_data)}")
    print(f"Forma de 'data': {getattr(dados_data, 'shape', 'N/A')}")
    print(f"Conteúdo de 'data': {dados_data}")  # Imprime o conteúdo de 'data'
    
    # Verifique a forma de dados_data
    if dados_data.ndim == 1:
        # Se for um array 1D, podemos tentar separar as imagens e os rótulos manualmente
        print(f"'data' é 1D. Tamanho: {dados_data.shape[0]}")
        # Verifique os primeiros valores para entender a estrutura
        print(f"Primeiros elementos de 'data': {dados_data[:10]}")
    elif dados_data.ndim == 2:
        # Se for 2D, talvez possamos separar as imagens e rótulos diretamente
        print(f"'data' é 2D. Forma: {dados_data.shape}")
        # Tente separar as imagens e rótulos, assumindo que as últimas colunas sejam rótulos
        # (ajuste conforme necessário se a estrutura for diferente)
        images = dados_data[:, :-1]  # Todas as colunas exceto a última
        labels = dados_data[:, -1]   # Última coluna (rótulos)
        print(f"Imagens (primeiros 10): {images[:10]}")
        print(f"Rótulos (primeiros 10): {labels[:10]}")
    else:
        print(f"'data' tem um formato inesperado. Dimensão: {dados_data.ndim}")

    # Realize o label flipping se as separações foram bem-sucedidas
    if 'images' in locals() and 'labels' in locals():
        labels_flipped = np.random.permutation(labels)  # Embaralha os rótulos aleatoriamente
        
        # Crie um novo dicionário com as imagens e os rótulos modificados
        dados_modificados = {'data': dados_data}  # Mantém os dados intactos
        dados_modificados['y'] = labels_flipped  # Substitua os rótulos pela versão modificada
        
        # Salve o arquivo modificado com os rótulos alterados
        np.savez(caminho_arquivo, data=dados_modificados)  # Agora, usa o novo dicionário
        
        # Imprimir para confirmar
        print(f"Rótulos após flipping (primeiros 10): {labels_flipped[:10]}")
