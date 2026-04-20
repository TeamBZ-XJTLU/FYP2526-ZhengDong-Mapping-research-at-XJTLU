"""
Network Analysis Module — Collaboration graph, centrality metrics, link prediction.

Functions
---------
build_collaboration_graph        Build co-authorship graph from shared works.
compute_network_metrics          Centrality, community detection, activity score.
compute_research_activity_scores Weighted publication/project/activity counts.
compute_potential_collaborations Predict missing links via embedding + topology.
save_network_results             Persist graph + metrics + predictions to data/.
"""

import pickle
import numpy as np
import pandas as pd
import networkx as nx
from itertools import combinations
from collections import defaultdict

from utils.project_config import (
    COLLAB_COMMUNITY_RESOLUTION,
    COLLAB_POTENTIAL_TOP_K,
    COLLAB_SIMILARITY_ALPHA,
    COLLAB_SIMILARITY_BETA,
    ACTIVITY_WEIGHT_PUBLICATION,
    ACTIVITY_WEIGHT_PROJECT,
    ACTIVITY_WEIGHT_ACTIVITY,
)


# ---------------------------------------------------------------------------
# 1. Build collaboration graph
# ---------------------------------------------------------------------------

def _extract_author_urls(authors_val) -> list[str]:
    """Safely extract author URL list from the raw 'authors' field."""
    if not isinstance(authors_val, list):
        return []
    urls = []
    for a in authors_val:
        if isinstance(a, dict):
            url = a.get("url", "")
            if url:
                urls.append(url)
    return urls


def build_collaboration_graph(datas: dict, datalinks: dict) -> nx.Graph:
    """
    Construct an undirected co-authorship graph.

    Nodes = author URLs.  Edge weight = # shared works.
    Edge attr 'works' stores list of {title, link, source_type, year}.
    Node attrs populated from teacher_df where available.
    """
    G = nx.Graph()

    teacher_df = datas.get("teacher_df")
    teacher_urls_set = set()
    if teacher_df is not None:
        teacher_urls_set = set(teacher_df["url"].dropna().unique())
        # Pre-populate nodes with teacher metadata
        for _, t in teacher_df.iterrows():
            url = t.get("url")
            if not url or pd.isna(url):
                continue
            G.add_node(url, **{
                "name": t.get("name", "Unknown"),
                "department": t.get("department", ""),
                "role": t.get("role", ""),
                "citation": t.get("citation", 0),
                "h_index": t.get("h_index", 0),
                "is_external": False,
            })

    source_configs = [
        ("publication", "publication_df"),
        ("project",     "project_df"),
        ("activity",    "activity_df"),
    ]

    for source_type, df_key in source_configs:
        df = datas.get(df_key)
        if df is None:
            continue

        for _, row in df.iterrows():
            author_urls = _extract_author_urls(row.get("authors"))
            if len(author_urls) < 2:
                continue

            title = row.get("title", "")
            link = row.get("link", "")
            year = row.get("year")
            year = int(year) if year is not None and not (isinstance(year, float) and pd.isna(year)) else None

            work_info = {"title": title, "link": link, "source_type": source_type, "year": year}

            # Ensure all author nodes exist
            for url in author_urls:
                if url not in G:
                    # External author — try to get name from authors list
                    name = ""
                    for a in row.get("authors", []):
                        if isinstance(a, dict) and a.get("url") == url:
                            name = a.get("name", "")
                            break
                    G.add_node(url, name=name, department="", role="",
                               citation=0, h_index=0, is_external=True)

            # Add edges for every pair of co-authors
            for u, v in combinations(author_urls, 2):
                if G.has_edge(u, v):
                    G[u][v]["weight"] += 1
                    G[u][v]["works"].append(work_info)
                else:
                    G.add_edge(u, v, weight=1, works=[work_info])

    n_internal = sum(1 for _, d in G.nodes(data=True) if not d.get("is_external", True))
    n_external = sum(1 for _, d in G.nodes(data=True) if d.get("is_external", True))
    print(f"[Network] Graph: {G.number_of_nodes()} nodes "
          f"({n_internal} internal + {n_external} external), "
          f"{G.number_of_edges()} edges")
    return G


# ---------------------------------------------------------------------------
# 2. Research activity scores
# ---------------------------------------------------------------------------

def compute_research_activity_scores(datas: dict, datalinks: dict) -> pd.Series:
    """
    Weighted count of publications, projects, activities per teacher.
    Returns Series indexed by teacher_url.
    """
    counts: dict[str, float] = defaultdict(float)

    weight_map = {
        "author_publication_relation_df": ACTIVITY_WEIGHT_PUBLICATION,
        "author_project_relation_df":     ACTIVITY_WEIGHT_PROJECT,
        "author_activity_relation_df":    ACTIVITY_WEIGHT_ACTIVITY,
    }

    for rel_key, weight in weight_map.items():
        rel = datalinks.get(rel_key)
        if rel is None:
            continue
        for _, r in rel.iterrows():
            url = r.get("teacher_url", "")
            if url:
                counts[url] += weight

    return pd.Series(counts, name="research_activity_score")


# ---------------------------------------------------------------------------
# 3. Network metrics & community detection
# ---------------------------------------------------------------------------

def compute_network_metrics(G: nx.Graph, datas: dict, datalinks: dict) -> pd.DataFrame:
    """
    Compute centrality metrics and community labels for every node.

    Returns DataFrame indexed by teacher_url.
    """
    import community as community_louvain   # python-louvain

    # Centrality measures
    degree_c = nx.degree_centrality(G)
    betweenness_c = nx.betweenness_centrality(G, weight="weight")

    try:
        eigenvector_c = nx.eigenvector_centrality_numpy(G, weight="weight")
    except Exception:
        eigenvector_c = {n: 0.0 for n in G.nodes()}

    # Community detection (Louvain)
    partition = community_louvain.best_partition(G, resolution=COLLAB_COMMUNITY_RESOLUTION, random_state=42)

    # Activity scores
    activity_scores = compute_research_activity_scores(datas, datalinks)

    records = []
    for node, attrs in G.nodes(data=True):
        records.append({
            "teacher_url": node,
            "name": attrs.get("name", ""),
            "department": attrs.get("department", ""),
            "role": attrs.get("role", ""),
            "is_external": attrs.get("is_external", True),
            "citation": attrs.get("citation", 0),
            "h_index": attrs.get("h_index", 0),
            "degree_centrality": round(degree_c.get(node, 0), 6),
            "betweenness_centrality": round(betweenness_c.get(node, 0), 6),
            "eigenvector_centrality": round(eigenvector_c.get(node, 0), 6),
            "community_id": partition.get(node, -1),
            "research_activity_score": activity_scores.get(node, 0),
        })

    metrics_df = pd.DataFrame(records)
    n_communities = metrics_df["community_id"].nunique()
    print(f"[Network] Metrics computed for {len(metrics_df)} nodes, "
          f"{n_communities} communities detected")
    return metrics_df


# ---------------------------------------------------------------------------
# 4. Potential collaboration prediction
# ---------------------------------------------------------------------------

def compute_potential_collaborations(
    G: nx.Graph,
    teacher_df: pd.DataFrame,
    teacher_trends_df: pd.DataFrame,
    emb_path: str = "data/doc_embeddings.pkl",
) -> pd.DataFrame:
    """
    Predict potential collaborations between non-connected internal teachers.

    Scoring = α * cosine_similarity(avg_embedding) + β * normalized_adamic_adar
    """
    from sklearn.metrics.pairwise import cosine_similarity

    # Load embeddings
    try:
        emb_df = pd.read_pickle(emb_path)
    except FileNotFoundError:
        print("[Network] Warning: doc_embeddings.pkl not found – skipping potential collaboration.")
        return pd.DataFrame()

    # Internal teacher URLs only
    internal_urls = [n for n, d in G.nodes(data=True) if not d.get("is_external", True)]
    if len(internal_urls) < 2:
        return pd.DataFrame()

    # Compute average embedding per teacher
    teacher_avg_emb: dict[str, np.ndarray] = {}
    for url in internal_urls:
        mask = emb_df["teacher_urls"].apply(lambda urls: url in urls if isinstance(urls, list) else False)
        docs = emb_df[mask]
        if docs.empty:
            continue
        embs = np.stack(docs["embedding"].values)
        teacher_avg_emb[url] = embs.mean(axis=0)

    eligible_urls = [u for u in internal_urls if u in teacher_avg_emb]
    if len(eligible_urls) < 2:
        return pd.DataFrame()

    # Build embedding matrix for fast cosine computation
    url_list = sorted(eligible_urls)
    emb_matrix = np.stack([teacher_avg_emb[u] for u in url_list])
    sim_matrix = cosine_similarity(emb_matrix)

    # Adamic-Adar for all non-connected pairs
    aa_dict: dict[tuple, float] = {}
    for u, v, score in nx.adamic_adar_index(G, [(u, v) for u, v in combinations(url_list, 2) if not G.has_edge(u, v)]):
        aa_dict[(u, v)] = score
        aa_dict[(v, u)] = score

    # Normalize AA scores
    max_aa = max(aa_dict.values()) if aa_dict else 1.0
    if max_aa == 0:
        max_aa = 1.0

    # Teacher keyword lookup for potential topics
    teacher_kws: dict[str, set] = {}
    if not teacher_trends_df.empty:
        for url, grp in teacher_trends_df.groupby("teacher_url"):
            teacher_kws[url] = set(grp["keyword"].unique())

    # Score all non-connected internal pairs
    url_idx = {url: i for i, url in enumerate(url_list)}
    records: dict[str, list] = defaultdict(list)  # teacher_url -> list of (partner, score, topics)

    for i, url_a in enumerate(url_list):
        for j, url_b in enumerate(url_list):
            if i >= j:
                continue
            if G.has_edge(url_a, url_b):
                continue

            cos_sim = float(sim_matrix[i, j])
            aa_score = aa_dict.get((url_a, url_b), 0.0) / max_aa
            combined = COLLAB_SIMILARITY_ALPHA * cos_sim + COLLAB_SIMILARITY_BETA * aa_score

            # Potential topics from overlapping keywords
            kws_a = teacher_kws.get(url_a, set())
            kws_b = teacher_kws.get(url_b, set())
            overlap = sorted(kws_a & kws_b)
            if not overlap:
                # Use union of top-3 from each
                overlap = sorted((kws_a | kws_b))[:6]

            records[url_a].append((url_b, cos_sim, aa_score, combined, overlap))
            records[url_b].append((url_a, cos_sim, aa_score, combined, overlap))

    # Keep top-K per teacher
    result_rows = []
    seen = set()
    for teacher_url, candidates in records.items():
        candidates.sort(key=lambda x: x[3], reverse=True)
        for partner_url, cos_sim, aa, combined, topics in candidates[:COLLAB_POTENTIAL_TOP_K]:
            pair = tuple(sorted([teacher_url, partner_url]))
            if pair in seen:
                continue
            seen.add(pair)
            result_rows.append({
                "teacher_url_a": pair[0],
                "teacher_url_b": pair[1],
                "topic_similarity": round(cos_sim, 4),
                "adamic_adar": round(aa, 4),
                "combined_score": round(combined, 4),
                "potential_topics": topics,
            })

    result = pd.DataFrame(result_rows)
    print(f"[Network] Potential collaborations: {len(result)} predicted links")
    return result


# ---------------------------------------------------------------------------
# 5. Save results
# ---------------------------------------------------------------------------

def save_network_results(
    G: nx.Graph,
    metrics_df: pd.DataFrame,
    potential_df: pd.DataFrame,
):
    """Persist network analysis artefacts."""
    with open("data/collaboration_graph.pkl", "wb") as f:
        pickle.dump(G, f)

    metrics_df.to_pickle("data/network_metrics.pkl")
    potential_df.to_pickle("data/potential_collaborations.pkl")

    print("[Network] Saved: collaboration_graph.pkl, network_metrics.pkl, "
          "potential_collaborations.pkl")
