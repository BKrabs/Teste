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


# Escala fundamental de Saaty
ESCALA_AHP = [
    "1/9",
    "1/8",
    "1/7",
    "1/6",
    "1/5",
    "1/4",
    "1/3",
    "1/2",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9"
]


# Conversão da escala textual para valores numéricos
VALORES_AHP = {
    "1/9": 1/9,
    "1/8": 1/8,
    "1/7": 1/7,
    "1/6": 1/6,
    "1/5": 1/5,
    "1/4": 1/4,
    "1/3": 1/3,
    "1/2": 1/2,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9
}


# ============================================================
# CÁLCULO DO AUTOVETOR PRINCIPAL
# ============================================================

def calcular_pesos(matriz):

    autovalores, autovetores = np.linalg.eig(matriz)

    # Identifica o maior autovalor
    indice_principal = np.argmax(autovalores.real)

    lambda_max = autovalores[indice_principal].real

    # Autovetor associado
    vetor_principal = autovetores[
        :, indice_principal
    ].real

    # Garante valores positivos
    vetor_principal = np.abs(vetor_principal)

    # Normalização
    pesos = vetor_principal / np.sum(vetor_principal)

    return pesos, lambda_max


# ============================================================
# CÁLCULO DA CONSISTÊNCIA
# ============================================================

def calcular_consistencia(matriz, pesos):

    n = matriz.shape[0]

    # Produto matriz-vetor
    aw = matriz @ pesos

    # Valores individuais de lambda
    lambda_vetor = aw / pesos

    # Lambda máximo
    lambda_max = np.mean(lambda_vetor)

    # Matrizes 1x1 e 2x2
    if n <= 2:
        return lambda_max, 0.0, 0.0

    # Índice de consistência
    ci = (lambda_max - n) / (n - 1)

    # Índice aleatório
    ri = RI_VALUES.get(n)

    if ri is None:
        cr = None
    elif ri == 0:
        cr = 0
    else:
        cr = ci / ri

    return lambda_max, ci, cr


# ============================================================
# STATUS DO CR
# ============================================================

def status_consistencia(cr):

    if cr is None:
        return "RI não disponível"

    if cr < 0.10:
        return "Consistência aceitável"

    return "CR acima de 10%"


# ============================================================
# EDITOR DE COMPARAÇÕES PAR A PAR
# ============================================================

def criar_comparacoes_par_a_par(
    elementos,
    chave,
    titulo
):

    st.subheader(titulo)

    st.info(
        """
        Compare cada par de elementos utilizando a escala
        fundamental do AHP.

        Valores maiores que 1 indicam preferência pelo
        primeiro elemento.

        Valores menores que 1 indicam preferência pelo
        segundo elemento.

        A comparação inversa é preenchida automaticamente.
        """
    )

    comparacoes = {}

    contador = 0

    for i in range(len(elementos)):

        for j in range(i + 1, len(elementos)):

            elemento_i = elementos[i]
            elemento_j = elementos[j]

            chave_comparacao = (
                f"{chave}_{i}_{j}"
            )

            selecao = st.selectbox(
                f"{elemento_i}  ×  {elemento_j}",
                ESCALA_AHP,
                index=8,
                key=chave_comparacao
            )

            valor = VALORES_AHP[selecao]

            comparacoes[(i, j)] = valor

            # Explicação da reciprocidade
            if valor == 1:
                texto = (
                    f"{elemento_i} e {elemento_j} "
                    "possuem a mesma importância."
                )

            elif valor > 1:
                texto = (
                    f"{elemento_i} recebe {selecao} "
                    f"em relação a {elemento_j}."
                )

            else:
                texto = (
                    f"{elemento_j} recebe "
                    f"{1 / valor:g} em relação a "
                    f"{elemento_i}."
                )

            st.caption(texto)

            contador += 1

    # Criação da matriz
    n = len(elementos)

    matriz = np.ones((n, n))

    for (i, j), valor in comparacoes.items():

        matriz[i, j] = valor

        # Reciprocidade automática
        matriz[j, i] = 1 / valor

    return matriz


# ============================================================
# EXIBIÇÃO DE MATRIZ
# ============================================================

def exibir_matriz(matriz, nomes):

    df = pd.DataFrame(
        matriz,
        index=nomes,
        columns=nomes
    )

    # Formatação
    df_formatado = df.map(
        lambda x: f"{x:.4f}"
    )

    st.dataframe(
        df_formatado,
        use_container_width=True
    )


# ============================================================
# TÍTULO
# ============================================================

st.title("Aplicação do Método AHP")

st.write(
    """
    Aplicação do Analytic Hierarchy Process (AHP) para
    seleção e ranqueamento de alternativas.
    """
)


# ============================================================
# DEFINIÇÃO DO PROBLEMA
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
        value=4,
        step=1
    )


# ============================================================
# CRITÉRIOS
# ============================================================

st.subheader("Critérios")

criterios = []

for i in range(numero_criterios):

    criterio = st.text_input(
        f"Critério {i + 1}",
        value=f"Critério {i + 1}",
        key=f"criterio_{i}"
    )

    criterios.append(criterio)


# ============================================================
# OPÇÕES
# ============================================================

st.subheader("Opções")

opcoes = []

for i in range(numero_opcoes):

    opcao = st.text_input(
        f"Opção {i + 1}",
        value=f"Opção {i + 1}",
        key=f"opcao_{i}"
    )

    opcoes.append(opcao)


# ============================================================
# MATRIZ DE CRITÉRIOS
# ============================================================

st.header("2. Comparação dos critérios")

matriz_criterios = criar_comparacoes_par_a_par(
    criterios,
    "criterios",
    "Comparações entre critérios"
)


# ============================================================
# MATRIZES DAS OPÇÕES
# ============================================================

st.header("3. Comparação das opções")

matrizes_opcoes = {}

for criterio in criterios:

    st.markdown(
        f"### Critério: {criterio}"
    )

    matriz = criar_comparacoes_par_a_par(
        opcoes,
        f"opcoes_{criterio}",
        f"Comparações das opções para {criterio}"
    )

    matrizes_opcoes[criterio] = matriz


# ============================================================
# BOTÃO DE CÁLCULO
# ============================================================

st.header("4. Processamento AHP")

calcular = st.button(
    "Calcular AHP",
    type="primary"
)


if calcular:

    # ========================================================
    # PESOS DOS CRITÉRIOS
    # ========================================================

    pesos_criterios, _ = calcular_pesos(
        matriz_criterios
    )

    (
        lambda_criterios,
        ci_criterios,
        cr_criterios
    ) = calcular_consistencia(
        matriz_criterios,
        pesos_criterios
    )


    # ========================================================
    # RESULTADOS DOS CRITÉRIOS
    # ========================================================

    st.header("5. Pesos dos critérios")

    resultado_criterios = pd.DataFrame({

        "Critério": criterios,

        "Peso": pesos_criterios,

        "Percentual":
            pesos_criterios * 100
    })

    resultado_criterios = (
        resultado_criterios
        .sort_values(
            "Peso",
            ascending=False
        )
        .reset_index(drop=True)
    )

    st.dataframe(
        resultado_criterios,
        use_container_width=True
    )


    # ========================================================
    # CONSISTÊNCIA DOS CRITÉRIOS
    # ========================================================

    st.subheader(
        "Consistência da matriz dos critérios"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Lambda máximo",
            f"{lambda_criterios:.6f}"
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


    if cr_criterios is not None:

        if cr_criterios < 0.10:

            st.success(
                "A matriz dos critérios apresenta "
                "consistência aceitável."
            )

        else:

            st.warning(
                "A matriz dos critérios apresenta "
                "CR superior a 10%."
            )


    # ========================================================
    # PRIORIDADES LOCAIS
    # ========================================================

    st.header(
        "6. Prioridades locais das opções"
    )

    prioridades_locais = np.zeros(
        (
            numero_opcoes,
            numero_criterios
        )
    )

    resultados_consistencia = []


    for j, criterio in enumerate(criterios):

        matriz = matrizes_opcoes[criterio]

        pesos, _ = calcular_pesos(
            matriz
        )

        (
            lambda_max,
            ci,
            cr
        ) = calcular_consistencia(
            matriz,
            pesos
        )

        prioridades_locais[:, j] = pesos


        # --------------------------------------------
        # Resultado
        # --------------------------------------------

        resultado_local = pd.DataFrame({

            "Opção": opcoes,

            "Prioridade":
                pesos,

            "Percentual":
                pesos * 100
        })

        resultado_local = (
            resultado_local
            .sort_values(
                "Prioridade",
                ascending=False
            )
            .reset_index(drop=True)
        )

        st.subheader(
            f"Prioridades para {criterio}"
        )

        st.dataframe(
            resultado_local,
            use_container_width=True
        )


        resultados_consistencia.append({

            "Matriz": criterio,

            "Lambda máximo":
                lambda_max,

            "CI": ci,

            "CR": cr,

            "Status":
                status_consistencia(cr)
        })


    # ========================================================
    # MATRIZ DE PRIORIDADES
    # ========================================================

    st.header(
        "7. Matriz de prioridades locais"
    )

    matriz_prioridades_df = pd.DataFrame(
        prioridades_locais,
        index=opcoes,
        columns=criterios
    )

    st.dataframe(
        matriz_prioridades_df,
        use_container_width=True
    )


    # ========================================================
    # PESOS GLOBAIS
    # ========================================================

    st.header(
        "8. Peso final das opções"
    )

    pesos_globais = (
        prioridades_locais
        @ pesos_criterios
    )


    resultado_final = pd.DataFrame({

        "Opção": opcoes,

        "Peso global":
            pesos_globais,

        "Percentual":
            pesos_globais * 100
    })


    resultado_final = (
        resultado_final
        .sort_values(
            "Peso global",
            ascending=False
        )
        .reset_index(drop=True)
    )


    resultado_final.insert(
        0,
        "Ranking",
        range(
            1,
            len(resultado_final) + 1
        )
    )


    st.dataframe(
        resultado_final,
        use_container_width=True
    )


    # ========================================================
    # CONTRIBUIÇÃO DE CADA CRITÉRIO
    # ========================================================

    st.subheader(
        "Contribuição de cada critério"
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


    # ========================================================
    # CONSISTÊNCIA DAS MATRIZES
    # ========================================================

    st.header(
        "9. Análise de consistência"
    )

    tabela_consistencia = pd.DataFrame(
        resultados_consistencia
    )

    st.dataframe(
        tabela_consistencia,
        use_container_width=True
    )


    # ========================================================
    # MATRIZ DOS CRITÉRIOS
    # ========================================================

    st.header(
        "10. Matriz de comparação dos critérios"
    )

    exibir_matriz(
        matriz_criterios,
        criterios
    )


    # ========================================================
    # MATRIZES DAS OPÇÕES
    # ========================================================

    st.header(
        "11. Matrizes de comparação das opções"
    )

    for criterio in criterios:

        st.subheader(
            criterio
        )

        exibir_matriz(
            matrizes_opcoes[criterio],
            opcoes
        )


    # ========================================================
    # VENCEDOR
    # ========================================================

    melhor = resultado_final.iloc[0]

    st.success(
        f"Melhor opção: {melhor['Opção']} | "
        f"Peso global: {melhor['Peso global']:.4f} | "
        f"{melhor['Percentual']:.2f}%"
    )