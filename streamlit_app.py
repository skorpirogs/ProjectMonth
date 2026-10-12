from pathlib import Path
import os
import re

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
from scipy import sparse
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


SEED = 42
MODEL_NAME = "intfloat/multilingual-e5-base"
ALPHA_E5 = 0.75
ALPHA_TFIDF = 0.25
SELECTED_CATEGORIES = ["Одежда", "Обувь"]

RAW_PRODUCTS_PATH = Path(
    os.environ.get("RAW_PRODUCTS_PATH", "wb_products_raw_sample.parquet")
)
ARTIFACT_DIR = Path(
    os.environ.get("STREAMLIT_ARTIFACT_DIR", "streamlit_artifacts")
)

CATALOG_PATH = ARTIFACT_DIR / "catalog_search.parquet"
TFIDF_VECTORIZER_PATH = ARTIFACT_DIR / "tfidf_vectorizer.joblib"
TFIDF_MATRIX_PATH = ARTIFACT_DIR / "tfidf_matrix_full.npz"
E5_EMBEDDINGS_PATH = ARTIFACT_DIR / "e5_embeddings_full.npy"


def clean_text(text):
    """Очищает текст тем же способом, что и основной notebook."""
    text = "" if pd.isna(text) else str(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^\w\s-]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def prepare_catalog(raw_path: Path) -> pd.DataFrame:
    """Формирует полный рабочий каталог на уровне imt_id."""
    df_raw = pd.read_parquet(raw_path)

    required_columns = {
        "imt_id",
        "imt_name",
        "subj_name",
        "subj_root_name",
        "description",
    }
    missing_columns = required_columns - set(df_raw.columns)
    if missing_columns:
        raise ValueError(
            "В каталоге отсутствуют обязательные колонки: "
            + ", ".join(sorted(missing_columns))
        )

    catalog = df_raw[
        df_raw["subj_root_name"].isin(SELECTED_CATEGORIES)
    ].copy()

    catalog["_description_len"] = (
        catalog["description"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.len()
    )

    catalog = (
        catalog
        .sort_values(
            ["imt_id", "_description_len"],
            ascending=[True, False],
            kind="mergesort",
        )
        .drop_duplicates(subset="imt_id", keep="first")
        .copy()
    )

    empty_name = (
        catalog["imt_name"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
    )
    empty_description = (
        catalog["description"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
    )

    catalog = catalog.loc[
        ~(empty_name & empty_description)
    ].copy()

    catalog["imt_name_clean"] = catalog["imt_name"].apply(clean_text)
    catalog["subj_name_clean"] = catalog["subj_name"].apply(clean_text)
    catalog["description_clean"] = catalog["description"].apply(clean_text)
    catalog["product_text"] = (
        catalog["imt_name_clean"]
        + " "
        + catalog["subj_name_clean"]
    ).str.replace(r"\s+", " ", regex=True).str.strip()

    catalog = catalog.drop(columns="_description_len")

    keep_columns = [
        "imt_id",
        "imt_name",
        "subj_name",
        "subj_root_name",
        "description",
        "product_text",
    ]
    if "brand_name" in catalog.columns:
        keep_columns.insert(3, "brand_name")

    catalog = catalog[keep_columns].reset_index(drop=True)

    if catalog["imt_id"].duplicated().any():
        raise ValueError("После подготовки каталога найдены дубликаты imt_id.")
    if (catalog["product_text"] == "").any():
        raise ValueError("После подготовки найдены пустые product_text.")

    return catalog


def build_artifacts():
    """Создаёт локальные артефакты финального гибридного поиска."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    catalog = prepare_catalog(RAW_PRODUCTS_PATH)

    tfidf_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=50_000,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )
    tfidf_matrix = tfidf_vectorizer.fit_transform(catalog["product_text"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    e5_model = SentenceTransformer(MODEL_NAME, device=device)

    passages = [
        f"passage: {text}"
        for text in catalog["product_text"].tolist()
    ]
    e5_embeddings = e5_model.encode(
        passages,
        batch_size=64 if device == "cuda" else 32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    catalog.to_parquet(CATALOG_PATH, index=False)
    joblib.dump(tfidf_vectorizer, TFIDF_VECTORIZER_PATH)
    sparse.save_npz(TFIDF_MATRIX_PATH, tfidf_matrix)
    np.save(E5_EMBEDDINGS_PATH, e5_embeddings)

    return (
        catalog,
        tfidf_vectorizer,
        tfidf_matrix,
        e5_model,
        e5_embeddings,
        device,
    )


@st.cache_resource(show_spinner=False)
def load_resources():
    """Загружает артефакты или создаёт их при первом запуске."""
    artifact_files = [
        CATALOG_PATH,
        TFIDF_VECTORIZER_PATH,
        TFIDF_MATRIX_PATH,
        E5_EMBEDDINGS_PATH,
    ]

    if not all(path.exists() for path in artifact_files):
        return build_artifacts()

    catalog = pd.read_parquet(CATALOG_PATH)
    tfidf_vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
    tfidf_matrix = sparse.load_npz(TFIDF_MATRIX_PATH)
    e5_embeddings = np.load(E5_EMBEDDINGS_PATH)

    if len(catalog) != tfidf_matrix.shape[0]:
        raise ValueError("TF-IDF-матрица не соответствует сохранённому каталогу.")
    if len(catalog) != e5_embeddings.shape[0]:
        raise ValueError("E5-эмбеддинги не соответствуют сохранённому каталогу.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    e5_model = SentenceTransformer(MODEL_NAME, device=device)

    return (
        catalog,
        tfidf_vectorizer,
        tfidf_matrix,
        e5_model,
        e5_embeddings,
        device,
    )


def minmax_normalize(scores: np.ndarray) -> np.ndarray:
    """Нормализует одномерный массив скоров в диапазон [0, 1]."""
    score_min = float(scores.min())
    score_max = float(scores.max())
    if score_max == score_min:
        return np.zeros_like(scores, dtype=np.float32)
    return ((scores - score_min) / (score_max - score_min)).astype(np.float32)


def search_hybrid(
    query: str,
    top_k: int,
    catalog: pd.DataFrame,
    tfidf_vectorizer,
    tfidf_matrix,
    e5_model,
    e5_embeddings: np.ndarray,
):
    """Выполняет финальный гибридный поиск TF-IDF + E5."""
    clean_query = clean_text(query)
    if not clean_query:
        return pd.DataFrame()

    query_tfidf = tfidf_vectorizer.transform([clean_query])
    lexical_scores = (
        query_tfidf @ tfidf_matrix.T
    ).toarray().ravel().astype(np.float32)

    query_embedding = e5_model.encode(
        [f"query: {clean_query}"],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)[0]
    semantic_scores = (e5_embeddings @ query_embedding).astype(np.float32)

    lexical_norm = minmax_normalize(lexical_scores)
    semantic_norm = minmax_normalize(semantic_scores)
    hybrid_scores = (
        ALPHA_E5 * semantic_norm
        + ALPHA_TFIDF * lexical_norm
    )

    item_ids = catalog["imt_id"].to_numpy().astype(np.int64)
    sorted_indices = np.lexsort((item_ids, -hybrid_scores))
    top_indices = sorted_indices[: min(top_k, len(sorted_indices))]

    result = catalog.iloc[top_indices].copy()
    result.insert(0, "rank", np.arange(1, len(result) + 1))
    result["hybrid_score"] = hybrid_scores[top_indices]
    result["e5_score"] = semantic_norm[top_indices]
    result["tfidf_score"] = lexical_norm[top_indices]
    return result.reset_index(drop=True)


def shorten_text(text, max_length=350):
    """Ограничивает длину описания для интерфейса."""
    text = "" if pd.isna(text) else str(text).strip()
    if len(text) <= max_length:
        return text
    return text[: max_length - 1].rstrip() + "…"


st.set_page_config(
    page_title="CalmFruits — гибридный поиск",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 CalmFruits — гибридный поиск товаров")
st.caption(
    "Финальная конфигурация проекта: "
    "0.75 × multilingual-e5-base + 0.25 × TF-IDF."
)

if not RAW_PRODUCTS_PATH.exists():
    st.error(
        "Не найден файл `wb_products_raw_sample.parquet`.\n\n"
        "Поместите исходный каталог в папку проекта "
        "или задайте путь через переменную окружения `RAW_PRODUCTS_PATH`."
    )
    st.stop()

with st.spinner(
    "Загрузка поискового пайплайна. "
    "При первом запуске создаются TF-IDF и E5-артефакты..."
):
    (
        catalog,
        tfidf_vectorizer,
        tfidf_matrix,
        e5_model,
        e5_embeddings,
        device,
    ) = load_resources()

with st.sidebar:
    st.header("Параметры")
    top_k = st.slider(
        "Количество результатов",
        min_value=5,
        max_value=20,
        value=10,
        step=1,
    )
    show_scores = st.checkbox(
        "Показывать технические скоры",
        value=False,
    )

    st.divider()
    st.write(f"Каталог: **{len(catalog):,}** карточек")
    st.write("Категории: **Одежда + Обувь**")
    st.write(f"Модель: **{MODEL_NAME}**")
    st.write(f"Устройство: **{device.upper()}**")

query = st.text_input(
    "Поисковый запрос",
    placeholder="Например: белые классические кеды",
)
search_clicked = st.button(
    "Найти",
    type="primary",
    use_container_width=True,
)

if search_clicked:
    if not query.strip():
        st.warning("Введите поисковый запрос.")
    else:
        with st.spinner("Поиск..."):
            results = search_hybrid(
                query=query,
                top_k=top_k,
                catalog=catalog,
                tfidf_vectorizer=tfidf_vectorizer,
                tfidf_matrix=tfidf_matrix,
                e5_model=e5_model,
                e5_embeddings=e5_embeddings,
            )

        if results.empty:
            st.info("По запросу не удалось сформировать выдачу.")
        else:
            st.subheader(f"Результаты по запросу: «{query.strip()}»")

            for _, row in results.iterrows():
                title = str(row.get("imt_name", "")).strip() or "Без названия"
                subject = str(row.get("subj_name", "")).strip() or "Категория не указана"
                brand = (
                    str(row.get("brand_name", "")).strip()
                    if "brand_name" in results.columns
                    else ""
                )
                description = shorten_text(row.get("description", ""))

                with st.container(border=True):
                    st.markdown(f"### {int(row['rank'])}. {title}")

                    metadata_parts = [
                        f"Категория: **{subject}**",
                        f"`imt_id`: **{int(row['imt_id'])}**",
                    ]
                    if brand and brand.lower() != "nan":
                        metadata_parts.insert(1, f"Бренд: **{brand}**")
                    st.markdown(" · ".join(metadata_parts))

                    if description:
                        st.write(description)

                    if show_scores:
                        col_1, col_2, col_3 = st.columns(3)
                        col_1.metric("Hybrid", f"{row['hybrid_score']:.4f}")
                        col_2.metric("E5", f"{row['e5_score']:.4f}")
                        col_3.metric("TF-IDF", f"{row['tfidf_score']:.4f}")

st.divider()
st.caption(
    "MVP проекта semantic search. "
    "Финальная конфигурация выбрана на validation; "
    "test использован только для итоговой оценки."
)
