import streamlit as st
import pandas as pd
import pickle
import ast
from sklearn.metrics.pairwise import cosine_similarity

# 1. CONFIGURAÇÃO DA PÁGINA
st.set_page_config(page_title="BoraVer - Recomendador Inteligente", layout="wide", page_icon="🍿")

# Estilização para melhorar o visual
# Estilização completa: Fundo, Barra Lateral e Inputs
st.markdown("""
    <style>
    /* 1. Fundo da página e do cabeçalho */
    .stApp, .stHeader {
        background-color: #0e1117 !important;
    }
    
    /* 2. Barra Lateral */
    [data-testid="stSidebar"] {
        background-color: #161b22 !important;
    }

    /* 3. Cor do texto digitado no Prompt (O MAIS IMPORTANTE) */
    input {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }

    /* 4. Estilização dos campos (Fundo e Borda) */
    .stTextInput>div>div>input, .stMultiSelect>div>div {
        background-color: #262730 !important;
        border: 1px solid #4b4b4b !important;
    }

    /* 5. Cor dos rótulos e textos gerais */
    .stApp p, .stApp h1, .stApp h2, .stApp h3, .stApp span, .stApp label, .stMarkdown {
        color: #ffffff !important;
    }

    /* 6. Tags do Multiselect */
    span[data-baseweb="tag"] {
        background-color: #ff4b4b !important;
        color: white !important;
    }

    /* 7. Botão */
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border: none;
    }

    header {
        visibility: hidden;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. FUNÇÃO PARA CARREGAR E LIMPAR O MODELO
@st.cache_resource
def carregar_modelo():
    # Carrega o arquivo gerado no Notebook 2
    with open('archive/modelo_completo.pkl', 'rb') as f:
        dados = pickle.load(f)
    
    df = dados['df']
    
    # CORREÇÃO DOS GÊNEROS: Converte string de lista para lista real do Python
    if isinstance(df['genres_list'].iloc[0], str):
        df['genres_list'] = df['genres_list'].apply(ast.literal_eval)
    
    return {
        'df': df,
        'matriz_tfidf': dados['matriz_tfidf'],
        'tfidf_vectorizer': dados['tfidf_vectorizer']
    }

# Inicializa os dados
dados = carregar_modelo()
df = dados['df']
matriz_tfidf = dados['matriz_tfidf']
tfidf = dados['tfidf_vectorizer']

# 3. BARRA LATERAL - VARIÁVEIS EXPOSITIVAS (Filtros)
st.sidebar.title("🎬 Configurações")

st.sidebar.subheader("🔞 Classificação")
filtro_adulto = st.sidebar.checkbox("Conteúdo Adulto (+18)", value=False)

st.sidebar.subheader("🌍 Idioma Original")
idiomas_disponiveis = sorted(df['original_language'].dropna().unique().tolist())
idiomas_selecionados = st.sidebar.multiselect("Selecione os idiomas:", idiomas_disponiveis, default=['English', 'Portuguese'])

st.sidebar.subheader("🎭 Gêneros")
# Agora os gêneros aparecem como palavras, não letras soltas
todos_generos = sorted(list(set([g for lista in df['genres_list'] for g in lista])))
generos_selecionados = st.sidebar.multiselect("Quais géneros prefere?", todos_generos)

# 4. ÁREA PRINCIPAL - INTERFACE E PROMPT
st.title("🤖 BoraVer: O Seu Próximo Filme Favorito")
st.write("Seu recomendador inteligente de filmes baseado em preferências e sinopses!")

prompt_usuario = st.text_input(
    "Descreve aí que tipo de filme te interessa:", 
    placeholder="Ex: A sad story about a dog or an action movie with robots",
)

# 5. LÓGICA DE RECOMENDAÇÃO (Unindo Filtros + Texto + Variáveis Discretas)
if st.button("Encontrar Sugestões"):
    
    # A. Aplicação das 'Hard Constraints' (Filtros do Sidebar)
    mask = (df['adult'] == filtro_adulto) & (df['original_language'].isin(idiomas_selecionados))
    
    if generos_selecionados:
        mask = mask & df['genres_list'].apply(lambda x: any(g in x for g in generos_selecionados))
    
    df_filtrado = df[mask].copy()

    if df_filtrado.empty:
        st.error("Ops! Não encontrei nenhum filme com esses filtros. Tenta mudar os idiomas ou gêneros!")
    else:
        # B. Cálculo do Score Semântico (Processamento do Prompt)
        if prompt_usuario:
            # Transforma o prompt em vetor
            vetor_usuario = tfidf.transform([prompt_usuario.lower()])
            
            # Calcula similaridade apenas nos filmes que passaram no filtro
            indices_filtrados = df_filtrado.index
            sim_textual = cosine_similarity(vetor_usuario, matriz_tfidf[indices_filtrados]).flatten()
            
            # C. Ranking Híbrido (Texto com peso 3x + Variáveis Discretas: Pop, ROI, Nota)
            df_filtrado['final_score'] = (sim_textual * 3) + df_filtrado['score_discreto']
        else:
            # Se não houver texto, ordena apenas pela "qualidade" (score_discreto)
            df_filtrado['final_score'] = df_filtrado['score_discreto']

        # D. EXIBIÇÃO EM GRID
        resultados = df_filtrado.sort_values('final_score', ascending=False).head(20)
        
        st.divider()
        st.subheader(f"🍿 Sugestões ideais para ti:")

        cols = st.columns(3)
        for i, (idx, row) in enumerate(resultados.iterrows()):
            with cols[i % 3]:
                st.info(f"**{row['title']}**")
                
                # Exibe métricas interessantes
                st.caption(f"⭐ Nota: {row['vote_average']} | 📅 {str(row['release_date'])[:4]}")
                
                # Badge de ROI para destacar o teu diferencial do Notebook 1
                if row['roi_relativo'] > 1.5:
                    st.success("🔥 Alto Retorno/Sucesso")
                
                with st.expander("Ver Sinopse"):
                    st.write(row['overview'])
                st.write("---")