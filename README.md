# Mínimos quadrados: implementação e casos práticos

Neste repositório estão compiladas implementações na linguagem Python para diversas
abordagens de resolução do método dos mínimos quadrados. Como o objetivo aqui é
implementar mínimos quadrados, e não conceitos matriciais e vetoriais, boa parte das
operações é apoiada pelo módulo `numpy`.

A pasta "src" contém exemplos práticos, usando conjuntos de dados diversos, de
aproximações utilizando mínimos quadrados. Apesar de, por padrão, estarem definidos
para serem resolvidos com o método `linalg.lstsq` do `numpy`, os resultados também
podem ser encontrados pelas implementações aqui contidas, localizadas no arquivo `leastsquares.py`.
O tempo de execução é avaliado para cada um dos casos. Cada um dos exemplos é acompanhado
por um arquivo `.pdf`, que contém uma explicação numérica do caso (pasta `exp`).

A maioria dos nomes, de pastas, variáveis e arquivos, está em Inglês para padronização com
a linguagem Python. Os comentários de código e explicações, contudo, estão em Português.

## Estrutura dos arquivos
**Diretórios**

* `/src` - contém os códigos Python.
* `/exp` - contém o desenvolvimento numérico dos casos implementados em `/src`.

**Arquivos de** `/src`

* `atime.py` - função para calcular o tempo de execução de soluções diversas
de mínimos qudrados.
* `data.py` - contém conjuntos de dados utilizados nas implementações práticas.
* `leastsquares.py` - contém a implementação de diversas abordagens para solução
de mínimos quadrados.
* `test.py` - contém um teste básico para saber se as funções de `leastsquares.py`
estão funcionando corretamente.
* `vmls_ex13-3.py` - implementa/resolve o exercício 13.3 do livro VMLS. Trata-se de uma aproximação de uma função afim.
* `vmls_fig13-3.py` - implementa o caso exibido na figura 13.3 do livro VMLS. Também trata-se de uma aproximação de uma função afim.
* `parabola.py` - cria uma aproximação de uma função quadrática a partir de postos
distribuídos de forma similar a uma parábola.
* `utcfr_exercicio.py` - resolve um exercício disponível [neste material](http://www.utc.fr/~mottelet/mt94/leastSquares.pdf). Nele, encontra-se a aproximação de um círculo para pontos dispostos em "formato" circular.

---

Soluções desenvolvidas como parte do projeto de iniciação científica
"Estudos teóricos e computacionais do método dos mínimos quadrados e suas variações", desenvolvido por
Daniel H. S. Pereira, orientado pelo professor Cassio M. Oishi da FCT/UNESP.

EN: Least squares solutions and implementations. Part of the project
"Theoretical and computational studies of the least squares method and its variations"
by Daniel Serezane and Cassio M. Oishi.