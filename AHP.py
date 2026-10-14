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
# FUNÇÃO PARA CRIAR MATRIZ EDITÁVEL
# ============================================================

def criar_editor_matriz(nomes, chave, titulo):

    n = len(nomes)

    matriz_inicial = np.ones((n, n))

    # Preenche valores padrão de reciprocidade
    for fileira in range(n):
        for coluna in range(n):

            if fileira < coluna:
                matriz_inicial[fileira, coluna] = 1
                matriz_inicial[coluna, fileira] = 1

    dataframe = pd.DataFrame(
        matriz_inicial,
        index=nomes,
        columns=nomes
    )

    st.subheader(titulo)

    matriz_editada = st.data_editor(
        dataframe,
        key=chave,
        use_container_width=True,
        num_rows="fixed"
    )

    return matriz_editada


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

matrizes_opcoes = {}

for criterio in criterios:

    st.markdown(f"### Opções em relação ao critério: {criterio}")

    matriz_df = criar_editor_matriz(
        opcoes,
        f"matriz_{criterio}",
        f"Matriz das opções - {criterio}"
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