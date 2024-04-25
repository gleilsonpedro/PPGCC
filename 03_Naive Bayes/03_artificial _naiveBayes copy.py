import numpy as np

# Criar um array para armazenar os dados
dados = np.zeros((100, 3))

# Gerar dados aleatórios para cada classe
for i in range(100):
    classe = np.random.randint(3)
    if classe == 0:
        dados[i, 0] = np.random.rand()
        dados[i, 1] = np.random.rand()
    elif classe == 1:
        dados[i, 0] = np.random.rand() + 0.5
        dados[i, 1] = np.random.rand() + 0.5
    else:
        dados[i, 0] = np.random.rand()
        dados[i, 1] = np.random.rand() * 0.5

# Salvar o conjunto de dados em um arquivo
np.savetxt('dados.csv', dados, delimiter=',')

import matplotlib.pyplot as plt
import numpy as np

# Carregar o conjunto de dados
dados = np.loadtxt('dados.csv', delimiter=',')

# Separar os dados por classe
classe_a = dados[dados[:, 2] == 0]  # Classe a
classe_b = dados[dados[:, 2] == 1]  # Classe b
classe_c = dados[dados[:, 2] == 2]  # Classe c

# Visualizar as classes
plt.plot(classe_a[:, 0], classe_a[:, 1], 'ro', label='Classe a')
plt.plot(classe_b[:, 0], classe_b[:, 1], 'bo', label='Classe b')
plt.plot(classe_c[:, 0], classe_c[:, 1], 'go', label='Classe c')

# Adicionar rótulos e legenda
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Conjunto de dados')
plt.legend()

# Mostrar o gráfico
plt.show()


# Mostrar o gráfico
plt.show()
