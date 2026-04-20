"""
Offline Data Mining Script
==========================
Run this once to generate all pre-computed pkl artefacts used by the web dashboard.

Usage:
    conda activate meizhongfyp
    python deep_data_analyze.py

Generated files (in data/):
    teacher_topic_trends.pkl    – per-teacher keyword weight time series
    department_topic_trends.pkl – per-department keyword weight time series
    topic_info.pkl              – BERTopic topic info table
    doc_embeddings.pkl          – sentence embeddings for all docs
    collaboration_graph.pkl     – networkx co-authorship graph
    network_metrics.pkl         – centrality, community, activity scores
    potential_collaborations.pkl – predicted future collaboration links
"""

import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from data.data_parser import datas, datalinks

from utils.topic_analysis import (
    build_corpus,
    train_bertopic_model,
    extract_teacher_topic_trends,
    extract_department_topic_trends,
    save_topic_results,
)
from utils.network_analysis import (
    build_collaboration_graph,
    compute_network_metrics,
    compute_potential_collaborations,
    save_network_results,
)


def main():
    t0 = time.time()

    # ------------------------------------------------------------------ #
    # Phase 1: Topic Analysis (BERTopic)
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("PHASE 1: Topic Analysis")
    print("=" * 60)

    corpus_df = build_corpus(datas, datalinks)
    if corpus_df.empty:
        print("[ERROR] Corpus is empty – cannot proceed with topic analysis.")
        return

    corpus_texts = corpus_df["text"].tolist()
    topic_model, topics, probs, embeddings = train_bertopic_model(corpus_texts)

    teacher_df = datas["teacher_df"]
    teacher_trends_df = extract_teacher_topic_trends(topic_model, corpus_df, topics, teacher_df)
    dept_trends_df = extract_department_topic_trends(teacher_trends_df, teacher_df)

    save_topic_results(teacher_trends_df, dept_trends_df, topic_model, embeddings, corpus_df)

    t1 = time.time()
    print(f"\n[Phase 1 completed in {t1 - t0:.1f}s]")

    # ------------------------------------------------------------------ #
    # Phase 2: Network Analysis
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("PHASE 2: Network Analysis")
    print("=" * 60)

    G = build_collaboration_graph(datas, datalinks)
    metrics_df = compute_network_metrics(G, datas, datalinks)
    potential_df = compute_potential_collaborations(G, teacher_df, teacher_trends_df)

    save_network_results(G, metrics_df, potential_df)

    t2 = time.time()
    print(f"\n[Phase 2 completed in {t2 - t1:.1f}s]")

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Corpus documents         : {len(corpus_df)}")
    print(f"  BERTopic topics          : {len(set(topics)) - (1 if -1 in topics else 0)}")
    print(f"  Teachers with trends     : {teacher_trends_df['teacher_url'].nunique() if not teacher_trends_df.empty else 0}")
    print(f"  Departments with trends  : {dept_trends_df['department'].nunique() if not dept_trends_df.empty else 0}")
    print(f"  Graph nodes              : {G.number_of_nodes()}")
    print(f"  Graph edges              : {G.number_of_edges()}")
    print(f"  Predicted collab links   : {len(potential_df)}")
    print(f"  Total elapsed time       : {t2 - t0:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
