import streamlit as st
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================

st.set_page_config(
    page_title="Aplicação AHP",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# CONSTANTES
# ============================================================

# Índices Aleatórios de Saaty
RI_VALUES = {
    1: 0.00,
    2: 0.00,
    3: 0.58,
    4: 0.90,
    5: 1.12,
    6: 1.24,
    7: 1.32,
    8: 1.41,
    9: 1.45,
    10: 1.49
}

SAATY_VALUES = [
    1/9, 1/8, 1/7, 1/6, 1/5, 1/4, 1/3, 1/2,
    1, 2, 3, 4, 5, 6, 7, 8, 9
]


# ============================================================
# FUNÇÃO PARA VALIDAR MATRIZ AHP
# ============================================================

def validar_matriz_ahp(matriz, tolerancia=0.001):

    n = matriz.shape[0]

    # Verifica elementos positivos
    if np.any(matriz <= 0):
        return False, "Todos os elementos da matriz devem ser maiores que zero."

    # Verifica diagonal igual a 1
    for fileira in range(n):
        if not np.isclose(matriz[fileira, fileira], 1.0, atol=tolerancia):
            return False, (
                f"O elemento da diagonal [{fileira + 1}, {fileira + 1}] deve ser igual a 1."
            )

    # Verifica reciprocidade
    for fileira in range(n):
        for coluna in range(n):

            if fileira != coluna:
                if not np.isclose(
                        matriz[fileira, coluna] * matriz[coluna, fileira],
                    1.0,
                    atol=tolerancia
                ):
                    return False, (
                        f"A matriz não é recíproca nas posições "
                        f"[{fileira + 1}, {coluna + 1}] e [{coluna + 1}, {fileira + 1}]."
                    )

    return True, "Matriz válida."


# ============================================================
# FUNÇÃO PARA CALCULAR O AUTOVETOR PRINCIPAL
# ============================================================

def calcular_pesos(matriz):

    # Calcula autovalores e autovetores
    autovalores, autovetores = np.linalg.eig(matriz)

    # Índice do maior autovalor
    indice_principal = np.argmax(autovalores.real)

    # Maior autovalor
    lambda_max = autovalores[indice_principal].real

    # Autovetor associado ao maior autovalor
    vetor_principal = autovetores[:, indice_principal].real

    # Elimina possíveis sinais negativos
    vetor_principal = np.abs(vetor_principal)

    # Normalização para soma igual a 1
    pesos = vetor_principal / np.sum(vetor_principal)

    return pesos, lambda_max


# ============================================================
# FUNÇÃO PARA CALCULAR A CONSISTÊNCIA
# ============================================================

def calcular_consistencia(matriz, pesos):

    n = matriz.shape[0]

    # Produto matriz-vetor
    aw = matriz @ pesos

    # Vetor lambda
    lambda_vetor = aw / pesos

    # Média dos valores
    lambda_max = np.mean(lambda_vetor)

    # Para matrizes 1x1 e 2x2
    if n <= 2:
        return lambda_max, 0.0, 0.0

    # Índice de consistência
    ci = (lambda_max - n) / (n - 1)

    # Índice aleatório
    ri = RI_VALUES.get(n, None)

    if ri is None:
        cr = None
    elif ri == 0:
        cr = 0.0
    else:
        cr = ci / ri

    return lambda_max, ci, cr


# ============================================================
# FUNÇÃO PARA FORMATAR O STATUS DO CR
# ============================================================

def status_consistencia(cr):

    if cr is None:
        return "Índice RI não disponível para esta ordem."

    if cr < 0.10:
        return "Consistência aceitável"
    else:
        return "Consistência acima de 10%"


# ============================================================
# FUNÇÃO PARA CRIAR MATRIZ AHP COM RECÍPROCOS NA MESMA TABELA
# ============================================================

def criar_editor_matriz(nomes, chave, titulo, matriz_inicial=None):
    """
    Apresenta a matriz AHP completa em uma única tabela visual.
    Apenas os elementos acima da diagonal são editáveis.
    A diagonal é igual a 1 e os elementos abaixo da diagonal
    mostram automaticamente os recíprocos dos valores inseridos.
    """

    n = len(nomes)

    if matriz_inicial is None:
        matriz_base = np.ones((n, n), dtype=float)
    else:
        matriz_base = np.asarray(matriz_inicial, dtype=float).copy()
        if matriz_base.shape != (n, n):
            raise ValueError("A matriz inicial possui dimensões incompatíveis.")

    st.subheader(titulo)
    st.caption(
        "Edite os valores acima da diagonal principal. "
        "A diagonal permanece igual a 1 e os valores abaixo dela "
        "são atualizados automaticamente com os recíprocos, na mesma matriz."
    )

    # Cabeçalho da matriz
    colunas = st.columns([1.35] + [1] * n)
    with colunas[0]:
        st.markdown("**Critério/opção**")
    for j, nome in enumerate(nomes):
        with colunas[j + 1]:
            st.markdown(f"**{nome}**")

    matriz_completa = np.ones((n, n), dtype=float)

    # Apresenta todas as células na mesma estrutura de matriz.
    for linha in range(n):
        colunas = st.columns([1.35] + [1] * n)

        with colunas[0]:
            st.markdown(f"**{nomes[linha]}**")

        for coluna in range(n):
            with colunas[coluna + 1]:
                if linha == coluna:
                    st.text("1")
                    matriz_completa[linha, coluna] = 1.0

                elif linha < coluna:
                    valor_inicial = float(matriz_base[linha, coluna])
                    if not np.isfinite(valor_inicial) or valor_inicial <= 0:
                        valor_inicial = 1.0

                    # Para matrizes geradas por valores nominais, o sufixo
                    # acompanha o valor inicial; assim, alterações nos dados
                    # nominais atualizam os valores iniciais dos campos.
                    sufixo = f"_{valor_inicial:.8g}" if matriz_inicial is not None else ""

                    valor = st.number_input(
                        f"{nomes[linha]} × {nomes[coluna]}",
                        min_value=0.000001,
                        value=valor_inicial,
                        format="%.6f",
                        key=f"{chave}_superior_{linha}_{coluna}{sufixo}",
                        label_visibility="collapsed"
                    )

                    matriz_completa[linha, coluna] = float(valor)
                    matriz_completa[coluna, linha] = 1.0 / float(valor)

                else:
                    # Mostra o recíproco na célula correspondente da mesma matriz.
                    valor_reciproco = matriz_completa[linha, coluna]
                    st.text(f"{valor_reciproco:.6f}")

    return pd.DataFrame(
        matriz_completa,
        index=nomes,
        columns=nomes
    )


# ============================================================
# FUNÇÃO PARA GERAR MATRIZ AHP A PARTIR DE VALORES NOMINAIS
# ============================================================

def gerar_matriz_valores_nominais(valores, tipo_criterio):
    """
    Converte valores técnicos em uma matriz de comparação AHP.

    Normalização: z = (x - mínimo) / (máximo - mínimo)
    Para critérios de custo, a preferência é invertida.
    Conversão: intensidade = arredondar(1 + 8 * |z_i - z_j|)

    A matriz é recíproca e usa intensidades inteiras de 1 a 9.
    """
    valores = np.asarray(valores, dtype=float)
    n = len(valores)
    matriz = np.ones((n, n), dtype=float)

    minimo = np.min(valores)
    maximo = np.max(valores)
    amplitude = maximo - minimo

    # Se todos os valores forem iguais, todas as opções são equivalentes.
    if np.isclose(amplitude, 0.0):
        return matriz

    normalizados = (valores - minimo) / amplitude

    # Para critérios de custo, menor valor significa maior preferência.
    if tipo_criterio == "Custo (menor é melhor)":
        normalizados = 1.0 - normalizados

    for i in range(n):
        for j in range(i + 1, n):
            diferenca = abs(normalizados[i] - normalizados[j])
            intensidade_bruta = 1.0 + 8.0 * diferenca
            intensidade = int(np.floor(intensidade_bruta + 0.5))
            intensidade = max(1, min(9, intensidade))

            if normalizados[i] > normalizados[j]:
                matriz[i, j] = intensidade
                matriz[j, i] = 1.0 / intensidade
            elif normalizados[j] > normalizados[i]:
                matriz[i, j] = 1.0 / intensidade
                matriz[j, i] = intensidade
            else:
                matriz[i, j] = 1.0
                matriz[j, i] = 1.0

    return matriz


# ============================================================
# TÍTULO
# ============================================================

st.title("Aplicação do Método AHP")

st.write(
    """
    Esta aplicação calcula:

    - pesos dos critérios;
    - prioridades locais das opções;
    - índice de consistência;
    - razão de consistência;
    - peso global das opções;
    - ranking final.
    """
)


# ============================================================
# ETAPA 1 - DEFINIÇÃO DO PROBLEMA
# ============================================================

st.header("1. Definição do problema")

col1, col2 = st.columns(2)

with col1:

    numero_criterios = st.number_input(
        "Número de critérios",
        min_value=2,
        max_value=10,
        value=3,
        step=1
    )

with col2:

    numero_opcoes = st.number_input(
        "Número de opções",
        min_value=2,
        max_value=10,
        value=3,
        step=1
    )


st.subheader("Critérios")

criterios = []

# noinspection PyTypeChecker
for i in range(numero_criterios):

    criterio = st.text_input(
        f"Critério {i + 1}",
        value=f"Critério {i + 1}",
        key=f"criterio_{i}"
    )

    criterios.append(criterio)


st.subheader("Opções")

opcoes = []

# noinspection PyTypeChecker
for i in range(numero_opcoes):

    opcao = st.text_input(
        f"Opção {i + 1}",
        value=f"Opção {i + 1}",
        key=f"opcao_{i}"
    )

    opcoes.append(opcao)


# ============================================================
# ETAPA 2 - MATRIZ DOS CRITÉRIOS
# ============================================================

st.header("2. Matriz de comparação dos critérios")

st.info(
    """
    Preencha a matriz utilizando a escala AHP.
    A diagonal principal deve ser igual a 1 e as comparações
    recíprocas devem obedecer à regra:

    a[i,j] = 1 / a[j,i]
    """
)

matriz_criterios_df = criar_editor_matriz(
    criterios,
    "matriz_criterios",
    "Comparação entre critérios"
)


# ============================================================
# ETAPA 3 - MATRIZES DAS OPÇÕES
# ============================================================

st.header("3. Matrizes de comparação das opções")

st.write(
    "Para cada critério, mantenha o preenchimento manual da matriz inteira "
    "ou gere uma matriz inicial a partir de valores técnicos nominais. "
    "No modo nominal, a matriz gerada também poderá ser editada integralmente."
)

matrizes_opcoes = {}

for indice_criterio, criterio in enumerate(criterios):

    st.markdown(f"### Opções em relação ao critério: {criterio}")

    modo_avaliacao = st.radio(
        f"Como deseja montar a matriz para '{criterio}'?",
        options=[
            "Preencher matriz AHP inteira manualmente",
            "Gerar matriz a partir de valores técnicos nominais"
        ],
        key=f"modo_avaliacao_{indice_criterio}"
    )

    if modo_avaliacao == "Preencher matriz AHP inteira manualmente":
        # Mantém o editor de matriz completa da versão original.
        matriz_df = criar_editor_matriz(
            opcoes,
            f"matriz_{criterio}",
            f"Matriz das opções - {criterio}"
        )

    else:
        tipo_criterio = st.radio(
            f"Sentido de preferência para '{criterio}'",
            options=[
                "Benefício (maior é melhor)",
                "Custo (menor é melhor)"
            ],
            key=f"tipo_criterio_nominal_{indice_criterio}",
            horizontal=True
        )

        st.caption(
            "Informe valores na mesma unidade. Exemplos: eficiência em %, "
            "custo em R$/kWh ou densidade energética em Wh/kg. "
            "A normalização considera o conjunto de opções informado."
        )

        valores_nominais = []
        for indice_opcao, opcao in enumerate(opcoes):
            valor = st.number_input(
                f"{opcao} — valor de '{criterio}'",
                value=0.0,
                key=f"valor_nominal_{indice_criterio}_{indice_opcao}",
                format="%.6f"
            )
            valores_nominais.append(valor)

        matriz_gerada = gerar_matriz_valores_nominais(
            valores_nominais,
            tipo_criterio
        )

        st.caption(
            "A matriz gerada serve como ponto de partida. Você pode ajustar "
            "os julgamentos na metade superior; a metade inferior será "
            "recalculada automaticamente pelos recíprocos."
        )

        matriz_df = criar_editor_matriz(
            opcoes,
            f"matriz_{criterio}_nominal",
            f"Matriz das opções - {criterio} (gerada)",
            matriz_inicial=matriz_gerada
        )

    matrizes_opcoes[criterio] = matriz_df


# ============================================================
# BOTÃO DE CÁLCULO
# ============================================================

st.header("4. Cálculo dos resultados")

if st.button("Calcular AHP", type="primary"):

    try:

        # ====================================================
        # MATRIZ DOS CRITÉRIOS
        # ====================================================

        # noinspection PyUnresolvedReferences
        matriz_criterios = matriz_criterios_df.to_numpy(
            dtype=float
        )

        valido, mensagem = validar_matriz_ahp(
            matriz_criterios
        )

        if not valido:
            st.error(
                f"Erro na matriz dos critérios: {mensagem}"
            )
            st.stop()


        # ====================================================
        # PESOS DOS CRITÉRIOS
        # ====================================================

        pesos_criterios, lambda_criterios = calcular_pesos(
            matriz_criterios
        )

        lambda_max_criterios, ci_criterios, cr_criterios = (
            calcular_consistencia(
                matriz_criterios,
                pesos_criterios
            )
        )


        # ====================================================
        # RESULTADO DOS CRITÉRIOS
        # ====================================================

        st.subheader("Pesos dos critérios")

        resultado_criterios = pd.DataFrame({
            "Critério": criterios,
            "Peso": pesos_criterios,
            "Percentual": pesos_criterios * 100
        })

        resultado_criterios = resultado_criterios.sort_values(
            by="Peso",
            ascending=False
        )

        st.dataframe(
            resultado_criterios,
            use_container_width=True
        )


        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Lambda máximo",
                f"{lambda_max_criterios:.6f}"
            )

        with col2:
            st.metric(
                "CI",
                f"{ci_criterios:.6f}"
            )

        with col3:
            if cr_criterios is not None:
                st.metric(
                    "CR",
                    f"{cr_criterios:.2%}"
                )

        with col4:
            if cr_criterios is not None:

                if cr_criterios < 0.10:
                    st.success("CR aceitável")
                else:
                    st.warning("CR acima de 10%")


        # ====================================================
        # PRIORIDADES LOCAIS
        # ====================================================

        st.header("5. Prioridades das opções por critério")

        # noinspection PyTypeChecker
        prioridades_locais = np.zeros(
            (numero_opcoes, numero_criterios)
        )

        resultados_consistencia = []


        for j, criterio in enumerate(criterios):

            matriz_opcoes = (
                matrizes_opcoes[criterio]
                .to_numpy(dtype=float)
            )

            valido, mensagem = validar_matriz_ahp(
                matriz_opcoes
            )

            if not valido:

                st.error(
                    f"Erro na matriz das opções para o critério "
                    f"'{criterio}': {mensagem}"
                )

                st.stop()


            # Cálculo dos pesos locais
            pesos_opcoes, lambda_opcoes = calcular_pesos(
                matriz_opcoes
            )

            lambda_max_opcoes, ci_opcoes, cr_opcoes = (
                calcular_consistencia(
                    matriz_opcoes,
                    pesos_opcoes
                )
            )


            # Armazena prioridades locais
            prioridades_locais[:, j] = pesos_opcoes


            # Exibe resultado local
            st.subheader(
                f"Prioridades das opções - {criterio}"
            )

            resultado_opcoes = pd.DataFrame({
                "Opção": opcoes,
                "Prioridade local": pesos_opcoes,
                "Percentual": pesos_opcoes * 100
            })

            resultado_opcoes = resultado_opcoes.sort_values(
                by="Prioridade local",
                ascending=False
            )

            st.dataframe(
                resultado_opcoes,
                use_container_width=True
            )


            # Armazena consistência
            resultados_consistencia.append({
                "Matriz": criterio,
                "Lambda máximo": lambda_max_opcoes,
                "CI": ci_opcoes,
                "CR": cr_opcoes,
                "Status": status_consistencia(cr_opcoes)
            })


        # ====================================================
        # MATRIZ DE PRIORIDADES LOCAIS
        # ====================================================

        st.header("6. Matriz de prioridades locais")

        matriz_prioridades = pd.DataFrame(
            prioridades_locais,
            index=opcoes,
            columns=criterios
        )

        st.dataframe(
            matriz_prioridades,
            use_container_width=True
        )


        # ====================================================
        # CÁLCULO DOS PESOS GLOBAIS
        # ====================================================

        st.header("7. Pesos globais e ranking final")

        pesos_globais = (
            prioridades_locais @ pesos_criterios
        )

        resultado_final = pd.DataFrame({
            "Opção": opcoes,
            "Peso global": pesos_globais,
            "Percentual": pesos_globais * 100
        })

        resultado_final = resultado_final.sort_values(
            by="Peso global",
            ascending=False
        )

        resultado_final["Ranking"] = range(
            1,
            len(resultado_final) + 1
        )

        resultado_final = resultado_final[
            [
                "Ranking",
                "Opção",
                "Peso global",
                "Percentual"
            ]
        ]

        st.dataframe(
            resultado_final,
            use_container_width=True
        )


        # ====================================================
        # CÁLCULO DETALHADO DO PESO GLOBAL
        # ====================================================

        st.subheader(
            "Composição dos pesos globais"
        )

        contribuicoes = (
            prioridades_locais
            * pesos_criterios
        )

        contribuicoes_df = pd.DataFrame(
            contribuicoes,
            index=opcoes,
            columns=criterios
        )

        contribuicoes_df["Peso global"] = (
            contribuicoes_df.sum(axis=1)
        )

        st.dataframe(
            contribuicoes_df,
            use_container_width=True
        )


        # ====================================================
        # CONSISTÊNCIA DE TODAS AS MATRIZES
        # ====================================================

        st.header("8. Resumo da consistência")

        consistencia_criterios = {
            "Matriz": "Critérios",
            "Lambda máximo": lambda_max_criterios,
            "CI": ci_criterios,
            "CR": cr_criterios,
            "Status": status_consistencia(cr_criterios)
        }

        resultados_consistencia.insert(
            0,
            consistencia_criterios
        )

        tabela_consistencia = pd.DataFrame(
            resultados_consistencia
        )

        st.dataframe(
            tabela_consistencia,
            use_container_width=True
        )


        # ====================================================
        # OPÇÃO VENCEDORA
        # ====================================================

        melhor_opcao = resultado_final.iloc[0]

        st.success(
            f"Melhor opção: {melhor_opcao['Opção']} | "
            f"Peso global: {melhor_opcao['Peso global']:.4f} | "
            f"Percentual: {melhor_opcao['Percentual']:.2f}%"
        )


    except ValueError:

        st.error(
            """
            Erro na conversão dos valores.

            Verifique se todos os elementos da matriz
            possuem valores numéricos válidos.
            """
        )

    except Exception as erro:

        st.error(
            f"Ocorreu um erro durante o cálculo: {erro}"
        )