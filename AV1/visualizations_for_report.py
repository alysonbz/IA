import pandas as pd
import matplotlib.pyplot as plt
import io

# Verifique o caminho correto do dataset
dataset_path = 'dataset/Cancer_Data.csv'  # Substitua pelo caminho real do seu arquivo

# Carregar o dataset
df = pd.read_csv(dataset_path)

import matplotlib.pyplot as plt
import io

# Capturar o output da função info()
buffer = io.StringIO()
df.info(buf=buffer)
info_str = buffer.getvalue()

# Converter o output em uma lista de linhas para plotagem
info_lines = info_str.splitlines()

# Configurar a plotagem
fig, ax = plt.subplots(figsize=(10, 8))  # Ajuste o tamanho da figura conforme necessário
ax.axis('off')

# Exibir o texto no gráfico com ajustes
text = "\n".join(info_lines)
ax.text(0.5, 1, text, fontsize=12, va="top", ha="center", family='monospace', wrap=True)

# Ajustar margens
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

# Salvar a imagem gerada
plt.savefig('dataset_info_visualization.png', bbox_inches='tight')

# Mostrar a visualização
plt.show()
