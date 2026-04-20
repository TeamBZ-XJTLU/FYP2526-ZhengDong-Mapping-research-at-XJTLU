"""
Topic Analysis Module — BERTopic-based research theme extraction.

Functions
---------
build_corpus          Build a unified corpus from publications, projects, activities.
train_bertopic_model  Train a global BERTopic model on the corpus.
extract_teacher_topic_trends   Per-teacher keyword-weight time series.
extract_department_topic_trends  Per-department keyword-weight time series.
save_topic_results    Persist all artefacts to data/ as pkl.
"""

import re
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict

from utils.project_config import (
    BERTOPIC_EMBEDDING_MODEL,
    BERTOPIC_MIN_TOPIC_SIZE,
    BERTOPIC_NR_TOPICS,
    BERTOPIC_TOP_N_WORDS,
    BERTOPIC_NGRAM_RANGE,
    BERTOPIC_MIN_ABSTRACT_LENGTH,
    TREND_TOP_KEYWORDS_PER_TEACHER,
    TREND_MIN_DOCS_FOR_TEACHER,
    TREND_TOP_KEYWORDS_PER_DEPARTMENT,
)

# ---------------------------------------------------------------------------
# Academic stop words: administrative / procedural terms that pollute topics
# ---------------------------------------------------------------------------
_ACADEMIC_STOPWORDS: set[str] = {
    # --- Roles & people ---
    "reviewer", "committee", "member", "chair", "co-chair", "cochair",
    "supervisor", "examiner", "assessor", "panelist", "judge",
    "student", "phd", "master", "msc", "mdes", "undergraduate", "postgraduate",
    "pg", "ug", "candidate", "advisor", "mentor", "mentee", "fellow",
    "lecturer", "professor", "instructor", "tutor", "teacher", "staff",
    "principal", "dean", "director", "coordinator",
    # --- Administrative actions ---
    "submitted", "completed", "reviewed", "invited", "accepted", "rejected",
    "assigned", "appointed", "attended", "organized", "chaired", "hosted",
    "supervised", "examined", "awarded", "nominated", "selected",
    "published", "presented", "delivered", "participated",
    # --- Administrative nouns ---
    "committee", "conference", "workshop", "symposium", "seminar", "session",
    "meeting", "report", "paper", "manuscript", "dissertation", "thesis",
    "journal", "editorial", "abstract", "abstracts", "proceeding",
    "submission", "review", "revision", "progress", "supervision",
    "assessment", "panel", "board", "course", "module", "teaching",
    "learning teaching", "award", "grant", "funding", "scholarship",
    "programme", "program", "project", "degree", "diploma",
    # --- Academic-admin compound phrases ---
    "technical program", "program committee", "committee member",
    "paper reviewer", "reviewer submitted", "submitted report",
    "report paper", "member technical", "hoc reviewer", "reviewer ad",
    "reviewer journal", "journal reviewer", "reviewer academic",
    "ad hoc", "tpc member", "committee tpc", "tpc",
    "completed student", "project completed", "course completed",
    "online course", "pg student", "phd student", "msc project",
    "student progress", "supervisor upd", "upd pg", "upd",
    "peer reviewed", "reviewed journal", "international peer",
    "academic international", "assessment panel",
    "progress supervisor", "project student", "design student",
    "member independent", "decision manuscript",
    "portal online", "coursera portal", "completed datacamp",
    "datacamp online", "using coursera",
    # --- Platforms & institutions ---
    "mdpi", "mdpi reviewer", "coursera", "datacamp", "edx", "xjtlu",
    "ibss", "baleap", "ceapa", "ceapa baleap",
    # --- Generic / filler ---
    "using", "based", "new", "study", "research", "analysis",
    "results", "approach", "method", "proposed", "used",
    "different", "provide", "show", "high", "two", "first",
    "also", "one", "well", "however", "within", "use",
    "including", "work", "various", "paper", "number",
    "recent", "et", "al", "fig", "figure", "table",
    "respectively", "total", "given", "made", "important",
}

def _build_custom_stopwords() -> list[str]:
    """Merge sklearn English stop words with academic-admin terms."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    return list(ENGLISH_STOP_WORDS | _ACADEMIC_STOPWORDS)


def _is_academic_keyword(kw: str) -> bool:
    """Return True if keyword looks like a genuine research term."""
    kw_lower = kw.lower().strip()
    # Reject if entirely in stopword set
    if kw_lower in _ACADEMIC_STOPWORDS:
        return False
    # Reject single character or purely numeric
    if len(kw_lower) <= 1 or kw_lower.replace(" ", "").isdigit():
        return False
    # Reject if every token is a stopword
    tokens = kw_lower.split()
    if all(t in _ACADEMIC_STOPWORDS for t in tokens):
        return False
    return True


def _consolidate_keywords(
    kw_year_weight: dict[str, dict[int, float]],
) -> dict[str, dict[int, float]]:
    """
    Merge unigrams into their longer n-gram counterparts when both exist.

    For example, if both "reinforcement" and "reinforcement learning" exist,
    merge the weight of "reinforcement" into "reinforcement learning" and
    remove "reinforcement" as a standalone keyword.
    """
    keywords = list(kw_year_weight.keys())
    absorbed: set[str] = set()

    for short_kw in keywords:
        if short_kw in absorbed:
            continue
        short_tokens = short_kw.split()
        if len(short_tokens) >= 3:
            continue  # only merge unigrams/bigrams into longer phrases

        # Find longer keywords that contain this shorter one
        for long_kw in keywords:
            if long_kw == short_kw or long_kw in absorbed:
                continue
            long_tokens = long_kw.split()
            if len(long_tokens) <= len(short_tokens):
                continue
            # Check if short_kw tokens form a contiguous sub-sequence of long_kw
            if short_kw in long_kw:
                # Merge weights: add short's year-weights into long
                for year, weight in kw_year_weight[short_kw].items():
                    kw_year_weight[long_kw][year] = (
                        kw_year_weight[long_kw].get(year, 0) + weight
                    )
                absorbed.add(short_kw)
                break  # short_kw absorbed into the first matching long_kw

    for kw in absorbed:
        del kw_year_weight[kw]

    return kw_year_weight


# ---------------------------------------------------------------------------
# 1. Corpus construction
# ---------------------------------------------------------------------------

def build_corpus(datas: dict, datalinks: dict) -> pd.DataFrame:
    """
    Collects abstracts from publication_df, project_df, activity_df and
    resolves teacher associations via the relation DataFrames.

    Returns a DataFrame with columns:
        text, source_type, source_link, year, teacher_urls (list)
    """
    rows = []

    source_configs = [
        ("publication", "publication_df", "author_publication_relation_df"),
        ("project",     "project_df",     "author_project_relation_df"),
        ("activity",    "activity_df",    "author_activity_relation_df"),
    ]

    for source_type, df_key, rel_key in source_configs:
        df = datas.get(df_key)
        rel = datalinks.get(rel_key)
        if df is None:
            continue

        # Build link -> teacher_urls mapping from the relation table
        link_to_teachers: dict[str, list[str]] = {}
        if rel is not None:
            for _, r in rel.iterrows():
                link = r.get("source_link", "")
                teacher_url = r.get("teacher_url", "")
                if link and teacher_url:
                    link_to_teachers.setdefault(link, []).append(teacher_url)

        for idx, row in df.iterrows():
            # Try 'abstract' first, then fall back to 'desc' (activity may use it)
            text = row.get("abstract") or row.get("desc") or ""
            if not isinstance(text, str) or len(text.strip()) < BERTOPIC_MIN_ABSTRACT_LENGTH:
                continue

            link = row.get("link", "")
            year = row.get("year")
            if pd.isna(year) if isinstance(year, float) else (year is None):
                continue
            year = int(year)

            teacher_urls = link_to_teachers.get(link, [])

            rows.append({
                "text": text.strip(),
                "source_type": source_type,
                "source_link": link,
                "year": year,
                "teacher_urls": teacher_urls,
            })

    corpus_df = pd.DataFrame(rows)
    # Deduplicate by source_link (a document may appear only once)
    if not corpus_df.empty:
        corpus_df = corpus_df.drop_duplicates(subset="source_link").reset_index(drop=True)

    print(f"[Topic] Corpus size: {len(corpus_df)} documents "
          f"(pub/proj/act breakdown: "
          f"{(corpus_df['source_type'] == 'publication').sum()}/"
          f"{(corpus_df['source_type'] == 'project').sum()}/"
          f"{(corpus_df['source_type'] == 'activity').sum()})")
    return corpus_df


# ---------------------------------------------------------------------------
# 2. BERTopic training
# ---------------------------------------------------------------------------

def train_bertopic_model(corpus_texts: list[str]):
    """
    Train a global BERTopic model with academic-focused keyword extraction.

    Uses:
    - Custom stop words to filter administrative language
    - KeyBERTInspired representation for coherent, embedding-aligned keywords
    - Ngram range (1,3) to capture multi-word research terms

    Returns
    -------
    model : BERTopic
    topics : list[int]
    probs : np.ndarray
    embeddings : np.ndarray   (sentence embeddings for reuse)
    """
    from bertopic import BERTopic
    from bertopic.representation import KeyBERTInspired
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from hdbscan import HDBSCAN

    print(f"[Topic] Loading embedding model: {BERTOPIC_EMBEDDING_MODEL}")
    embedding_model = SentenceTransformer(BERTOPIC_EMBEDDING_MODEL)

    print(f"[Topic] Encoding {len(corpus_texts)} documents …")
    embeddings = embedding_model.encode(corpus_texts, show_progress_bar=True)

    # Custom stop words: English defaults + academic-admin terms
    custom_stops = _build_custom_stopwords()

    vectorizer = CountVectorizer(
        ngram_range=BERTOPIC_NGRAM_RANGE,
        stop_words=custom_stops,
        min_df=2,               # terms must appear in >= 2 documents
        max_df=0.85,            # skip terms in > 85% of documents
    )

    # KeyBERTInspired: re-ranks topic words by cosine similarity to
    # the topic embedding, producing more coherent academic keywords
    representation_model = KeyBERTInspired(top_n_words=BERTOPIC_TOP_N_WORDS)

    # HDBSCAN with a larger min_cluster_size merges micro-topics
    hdbscan_model = HDBSCAN(
        min_cluster_size=BERTOPIC_MIN_TOPIC_SIZE,
        min_samples=3,
        metric="euclidean",
        prediction_data=True,
    )

    nr_topics = None if BERTOPIC_NR_TOPICS == "auto" else int(BERTOPIC_NR_TOPICS)

    topic_model = BERTopic(
        embedding_model=embedding_model,
        hdbscan_model=hdbscan_model,
        min_topic_size=BERTOPIC_MIN_TOPIC_SIZE,
        nr_topics=nr_topics,
        top_n_words=BERTOPIC_TOP_N_WORDS,
        vectorizer_model=vectorizer,
        representation_model=representation_model,
        verbose=True,
    )

    print("[Topic] Fitting BERTopic …")
    topics, probs = topic_model.fit_transform(corpus_texts, embeddings)

    topic_info = topic_model.get_topic_info()
    n_topics = len(topic_info[topic_info["Topic"] != -1])
    print(f"[Topic] Discovered {n_topics} topics (excl. outlier topic -1)")

    # --- Post-hoc: filter non-academic keywords from each topic ---
    _filter_topic_keywords(topic_model)

    return topic_model, topics, probs, embeddings


def _filter_topic_keywords(model):
    """
    Remove non-academic keywords from every topic in-place and log results.
    Topics left with zero keywords are removed entirely.
    """
    cleaned = 0
    empty_topics = []
    for tid in list(model.topic_representations_.keys()):
        if tid == -1:
            continue
        original = model.topic_representations_[tid]
        filtered = [(w, s) for w, s in original if _is_academic_keyword(w)]
        if len(filtered) == 0:
            # Topic has no academic keywords — remove it entirely
            empty_topics.append(tid)
            del model.topic_representations_[tid]
            cleaned += len(original)
        elif len(filtered) < len(original):
            cleaned += len(original) - len(filtered)
            model.topic_representations_[tid] = filtered
    if empty_topics:
        print(f"[Topic] Post-filter: removed {len(empty_topics)} topics with no academic keywords")
    print(f"[Topic] Post-filter: removed {cleaned} non-academic keywords across topics")


# ---------------------------------------------------------------------------
# 3. Per-teacher topic trends
# ---------------------------------------------------------------------------

def extract_teacher_topic_trends(
    model,
    corpus_df: pd.DataFrame,
    topics: list[int],
    teacher_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each teacher, extract keyword weight time series.

    Returns DataFrame: [teacher_url, teacher_name, year, keyword, weight]
    """
    # Map teacher_url -> name for labelling
    url_to_name = dict(zip(teacher_df["url"], teacher_df["name"]))

    # Get topic -> keyword mapping (list of (word, score) tuples)
    topic_keywords: dict[int, list[tuple[str, float]]] = {}
    for tid in set(topics):
        if tid == -1:
            continue
        kw = model.get_topic(tid)  # list of (word, score)
        if kw:
            topic_keywords[tid] = kw

    # Assign topic id to each corpus row
    corpus_df = corpus_df.copy()
    corpus_df["topic_id"] = topics

    records = []
    teacher_urls = teacher_df["url"].dropna().unique()

    for teacher_url in teacher_urls:
        name = url_to_name.get(teacher_url, "Unknown")

        # Documents associated with this teacher
        mask = corpus_df["teacher_urls"].apply(lambda urls: teacher_url in urls)
        teacher_docs = corpus_df[mask]

        if len(teacher_docs) < TREND_MIN_DOCS_FOR_TEACHER:
            continue

        # Aggregate keyword weights per year
        # keyword -> year -> accumulated weight
        kw_year_weight: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))

        for _, doc in teacher_docs.iterrows():
            tid = doc["topic_id"]
            year = doc["year"]
            if tid == -1 or tid not in topic_keywords:
                continue
            for word, score in topic_keywords[tid]:
                if _is_academic_keyword(word):
                    kw_year_weight[word][year] += abs(score)

        if not kw_year_weight:
            continue

        # Consolidate: merge short keywords into longer ngrams
        kw_year_weight = _consolidate_keywords(kw_year_weight)

        # Select top-N keywords by total weight across all years
        kw_total = {kw: sum(ydict.values()) for kw, ydict in kw_year_weight.items()}
        top_keywords = sorted(kw_total, key=kw_total.get, reverse=True)[:TREND_TOP_KEYWORDS_PER_TEACHER]

        for kw in top_keywords:
            for year, weight in kw_year_weight[kw].items():
                records.append({
                    "teacher_url": teacher_url,
                    "teacher_name": name,
                    "year": year,
                    "keyword": kw,
                    "weight": round(weight, 4),
                })

    result = pd.DataFrame(records)
    print(f"[Topic] Teacher trends: {len(result)} rows for "
          f"{result['teacher_url'].nunique() if not result.empty else 0} teachers")
    return result


# ---------------------------------------------------------------------------
# 4. Per-department topic trends
# ---------------------------------------------------------------------------

def extract_department_topic_trends(
    teacher_trends_df: pd.DataFrame,
    teacher_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate teacher-level trends to department level.

    Returns DataFrame: [department, year, keyword, weight]
    """
    if teacher_trends_df.empty:
        return pd.DataFrame(columns=["department", "year", "keyword", "weight"])

    # Map teacher_url -> department
    url_to_dept = dict(zip(teacher_df["url"], teacher_df["department"]))
    merged = teacher_trends_df.copy()
    merged["department"] = merged["teacher_url"].map(url_to_dept)
    merged = merged.dropna(subset=["department"])

    records = []
    for dept, dept_group in merged.groupby("department"):
        # Aggregate keyword weights across all teachers in the department
        kw_year = dept_group.groupby(["keyword", "year"])["weight"].sum().reset_index()

        # Select top-N keywords by total weight
        kw_totals = kw_year.groupby("keyword")["weight"].sum()
        top_kws = kw_totals.nlargest(TREND_TOP_KEYWORDS_PER_DEPARTMENT).index.tolist()

        for _, row in kw_year[kw_year["keyword"].isin(top_kws)].iterrows():
            records.append({
                "department": dept,
                "year": int(row["year"]),
                "keyword": row["keyword"],
                "weight": round(row["weight"], 4),
            })

    result = pd.DataFrame(records)
    print(f"[Topic] Department trends: {len(result)} rows for "
          f"{result['department'].nunique() if not result.empty else 0} departments")
    return result


# ---------------------------------------------------------------------------
# 5. Save results
# ---------------------------------------------------------------------------

def save_topic_results(
    teacher_trends_df: pd.DataFrame,
    dept_trends_df: pd.DataFrame,
    topic_model,
    embeddings: np.ndarray,
    corpus_df: pd.DataFrame,
):
    """Persist all topic-analysis artefacts to data/ folder."""
    teacher_trends_df.to_pickle("data/teacher_topic_trends.pkl")
    dept_trends_df.to_pickle("data/department_topic_trends.pkl")

    # Save topic info table
    topic_info = topic_model.get_topic_info()
    topic_info.to_pickle("data/topic_info.pkl")

    # Save embeddings + minimal corpus mapping (for network analysis reuse)
    emb_df = corpus_df[["source_link", "teacher_urls"]].copy()
    emb_df["embedding"] = list(embeddings)
    emb_df.to_pickle("data/doc_embeddings.pkl")

    print("[Topic] Saved: teacher_topic_trends.pkl, department_topic_trends.pkl, "
          "topic_info.pkl, doc_embeddings.pkl")
