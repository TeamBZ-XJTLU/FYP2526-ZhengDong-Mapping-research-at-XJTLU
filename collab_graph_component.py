"""
Reusable Collaboration Graph AIO (All-In-One) Component
========================================================
Renders an ego-centric co-authorship network graph with click-based
info panels and evidence tables.  Designed to be embedded in any page.

Usage:
    from utils.collab_graph_component import CollaborationGraphAIO

    section = CollaborationGraphAIO(teacher_url="https://...", aio_id="my-graph")
"""

import json
import math
import uuid

from dash import html, dcc, Input, Output, State, callback, no_update, MATCH
import dash_cytoscape as cyto
import pandas as pd

from data.data_parser import (
    datas,
    teacher_topic_trends,
    network_metrics,
    collaboration_graph,
)
from utils.project_config import (
    COLLAB_MAX_DISPLAY_NEIGHBORS,
    CYTOSCAPE_LAYOUT_NAME,
    CYTOSCAPE_CENTER_NODE_SIZE,
    CYTOSCAPE_NEIGHBOR_NODE_SIZE,
    CYTOSCAPE_EXTERNAL_NODE_SIZE,
    CYTOSCAPE_EDGE_WIDTH_RANGE,
    CHART_FONT_FAMILY,
    ACTIVITY_WEIGHT_PUBLICATION,
    ACTIVITY_WEIGHT_PROJECT,
    ACTIVITY_WEIGHT_ACTIVITY,
)

# -----------------------------------------------------------------------
# Module-level lookup dicts  (teacher url <-> DataFrame index)
# -----------------------------------------------------------------------
_teacher_df = datas.get("teacher_df")
_url_to_idx: dict[str, int] = {}
_idx_to_url: dict[int, str] = {}
if _teacher_df is not None:
    for i, row in _teacher_df.iterrows():
        url = row.get("url")
        if url and not pd.isna(url):
            _url_to_idx[url] = i
            _idx_to_url[i] = url


# -----------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------
def _safe_str(val, default=""):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return default
    return str(val)


def _get_teacher_topics(teacher_url: str) -> list[str]:
    if teacher_topic_trends is None or teacher_topic_trends.empty:
        return []
    t = teacher_topic_trends[teacher_topic_trends["teacher_url"] == teacher_url]
    if t.empty:
        return []
    kw_weights = t.groupby("keyword")["weight"].sum().nlargest(5)
    return kw_weights.index.tolist()


def _build_works_summary(works, type_page=None):
    """Build a compact summary list from raw works for edge data."""
    if type_page is None:
        type_page = {"publication": "publish_info", "project": "project_info", "activity": "activity_info"}
    summary = []
    for wk in works[:8]:
        wk_href = ""
        src_type = wk.get("source_type", "")
        if src_type and wk.get("link"):
            page = type_page.get(src_type, "")
            if page:
                df_key = f"{src_type}_df"
                df = datas.get(df_key)
                if df is not None:
                    match = df[df["link"] == wk["link"]]
                    if not match.empty:
                        wk_href = f"/{page}?id={match.index[0]}"
        summary.append({
            "title": wk.get("title", "")[:60],
            "type": src_type,
            "year": wk.get("year", ""),
            "href": wk_href,
        })
    return summary


# -----------------------------------------------------------------------
# Ego-network element & stylesheet builders
# -----------------------------------------------------------------------
def build_ego_network_elements(center_url: str, expanded_url: str = None):
    """
    Build ego network elements, optionally expanding a neighbor node.

    When *expanded_url* is a direct neighbor of *center_url*, that node's own
    neighbors (and edges) are merged into the element set.  Overlapping nodes
    are NOT duplicated.
    """
    G = collaboration_graph
    if G is None or center_url not in G:
        return [], []

    # --- Center's neighbors ---
    center_neighbors = list(G.neighbors(center_url))
    if len(center_neighbors) > COLLAB_MAX_DISPLAY_NEIGHBORS:
        center_neighbors.sort(key=lambda n: G[center_url][n].get("weight", 0), reverse=True)
        center_neighbors = center_neighbors[:COLLAB_MAX_DISPLAY_NEIGHBORS]

    center_neighbor_set = set(center_neighbors)

    # --- Expanded node's neighbors (only if it's a valid direct neighbor) ---
    expanded_all_neighbors = []
    expanded_new_set = set()  # neighbors unique to expanded node (not in center's set)
    if (expanded_url and expanded_url in G
            and expanded_url != center_url
            and expanded_url in center_neighbor_set):
        exp_nbrs = [n for n in G.neighbors(expanded_url) if n != center_url]
        if len(exp_nbrs) > COLLAB_MAX_DISPLAY_NEIGHBORS:
            exp_nbrs.sort(key=lambda n: G[expanded_url][n].get("weight", 0), reverse=True)
            exp_nbrs = exp_nbrs[:COLLAB_MAX_DISPLAY_NEIGHBORS]
        expanded_all_neighbors = exp_nbrs
        expanded_new_set = set(exp_nbrs) - center_neighbor_set
    else:
        expanded_url = None  # invalid expansion

    # All nodes to display
    node_set = set([center_url]) | center_neighbor_set | expanded_new_set

    elements = []

    # --- Normalization (center edges) ---
    center_weights = [G[center_url][n].get("weight", 1) for n in center_neighbors]
    center_max_w = max(center_weights) if center_weights else 1

    neighbor_strength = {}
    for n in center_neighbors:
        w = G[center_url][n].get("weight", 1)
        neighbor_strength[n] = w / max(center_max_w, 1)

    # --- Add nodes ---
    type_page = {"publication": "publish_info", "project": "project_info", "activity": "activity_info"}

    for url in node_set:
        attrs = G.nodes[url]
        is_center = (url == center_url)
        is_expanded = (url == expanded_url)
        is_expanded_only = (url in expanded_new_set)
        is_external = attrs.get("is_external", True)
        idx = _url_to_idx.get(url)
        topics = _get_teacher_topics(url)
        community = 0
        if network_metrics is not None:
            row = network_metrics[network_metrics["teacher_url"] == url]
            if not row.empty:
                community = int(row.iloc[0].get("community_id", 0))

        strength = neighbor_strength.get(url, 0) if not is_center else 0

        # Determine CSS class
        if is_center:
            css_class = "center"
        elif is_expanded:
            css_class = "expanded"
        elif is_expanded_only:
            css_class = "expanded-neighbor"
        elif is_external:
            css_class = "external"
        else:
            css_class = "internal"

        elements.append({
            "data": {
                "id": url,
                "label": _safe_str(attrs.get("name", "?"))[:20],
                "department": _safe_str(attrs.get("department", "")),
                "role": _safe_str(attrs.get("role", "")),
                "topics": ", ".join(topics) if topics else "N/A",
                "is_center": is_center,
                "is_expanded": is_expanded,
                "is_external": is_external,
                "community": community,
                "strength": round(strength, 3),
                "href": f"/collaboration_network?id={idx}" if idx is not None else "",
            },
            "classes": css_class,
        })

    # --- Add edges from center to its neighbors ---
    min_w, max_w_px = CYTOSCAPE_EDGE_WIDTH_RANGE

    for n in center_neighbors:
        if not G.has_edge(center_url, n):
            continue
        edge_data = G[center_url][n]
        w = edge_data.get("weight", 1)
        works_summary = _build_works_summary(edge_data.get("works", []), type_page)
        norm_w = min_w + (w / max(center_max_w, 1)) * (max_w_px - min_w)

        elements.append({
            "data": {
                "source": center_url,
                "target": n,
                "weight": w,
                "norm_width": round(norm_w, 1),
                "works_json": json.dumps(works_summary),
                "label": f"{w} work{'s' if w > 1 else ''}",
            }
        })

    # --- Add edges from expanded node to its displayed neighbors ---
    if expanded_url and expanded_all_neighbors:
        exp_weights = [G[expanded_url][n].get("weight", 1)
                       for n in expanded_all_neighbors if G.has_edge(expanded_url, n)]
        exp_max_w = max(exp_weights) if exp_weights else 1

        for n in expanded_all_neighbors:
            if not G.has_edge(expanded_url, n):
                continue
            if n not in node_set:
                continue
            edge_data = G[expanded_url][n]
            w = edge_data.get("weight", 1)
            works_summary = _build_works_summary(edge_data.get("works", []), type_page)
            norm_w = min_w + (w / max(exp_max_w, 1)) * (max_w_px - min_w)

            elements.append({
                "data": {
                    "source": expanded_url,
                    "target": n,
                    "weight": w,
                    "norm_width": round(norm_w, 1),
                    "works_json": json.dumps(works_summary),
                    "label": f"{w} work{'s' if w > 1 else ''}",
                },
                "classes": "expanded-edge",
            })

    stylesheet = build_base_stylesheet()
    return elements, stylesheet


def build_base_stylesheet(is_potential: bool = False):
    """Public stylesheet builder."""
    styles = [
        {
            "selector": "node.internal",
            "style": {
                "label": "data(label)",
                "width": CYTOSCAPE_NEIGHBOR_NODE_SIZE,
                "height": CYTOSCAPE_NEIGHBOR_NODE_SIZE,
                "background-color": "#7B8FB2",
                "color": "#334155",
                "font-size": "10px",
                "font-family": CHART_FONT_FAMILY,
                "text-valign": "bottom",
                "text-margin-y": "6px",
                "border-width": 2,
                "border-color": "#B0BDD4",
            },
        },
        {
            "selector": "node.internal[strength]",
            "style": {
                "background-color": "mapData(strength, 0, 1, #7B8FB2, #C75B5B)",
                "border-color": "mapData(strength, 0, 1, #B0BDD4, #E0A0A0)",
            },
        },
        {
            "selector": "node.center",
            "style": {
                "label": "data(label)",
                "width": CYTOSCAPE_CENTER_NODE_SIZE,
                "height": CYTOSCAPE_CENTER_NODE_SIZE,
                "background-color": "#4f46e5",
                "color": "#1e293b",
                "font-size": "12px",
                "font-weight": "bold",
                "font-family": CHART_FONT_FAMILY,
                "text-valign": "bottom",
                "text-margin-y": "8px",
                "border-width": 3,
                "border-color": "#818cf8",
            },
        },
        {
            "selector": "node.external",
            "style": {
                "label": "data(label)",
                "width": CYTOSCAPE_EXTERNAL_NODE_SIZE,
                "height": CYTOSCAPE_EXTERNAL_NODE_SIZE,
                "background-color": "#94a3b8",
                "color": "#64748b",
                "font-size": "8px",
                "font-family": CHART_FONT_FAMILY,
                "shape": "diamond",
                "text-valign": "bottom",
                "text-margin-y": "4px",
            },
        },
        {
            "selector": "edge",
            "style": {
                "width": "data(norm_width)",
                "line-color": "#cbd5e1",
                "curve-style": "bezier",
                "opacity": 0.7,
            },
        },
        {
            "selector": "node:selected",
            "style": {
                "border-color": "#10b981",
                "border-width": 4,
            },
        },
        {
            "selector": "edge:selected",
            "style": {
                "line-color": "#10b981",
                "opacity": 1,
            },
        },
    ]

    if is_potential:
        styles.append({
            "selector": "edge.potential",
            "style": {
                "line-style": "dashed",
                "line-color": "#10b981",
                "line-dash-pattern": [6, 3],
                "opacity": 0.6,
            },
        })
        styles.append({
            "selector": "edge.potential:selected",
            "style": {
                "line-color": "#059669",
                "opacity": 1,
            },
        })

    # --- Expanded-node styles (always included, harmless if no matches) ---
    styles.extend([
        {
            "selector": "node.expanded",
            "style": {
                "label": "data(label)",
                "width": CYTOSCAPE_CENTER_NODE_SIZE * 0.8,
                "height": CYTOSCAPE_CENTER_NODE_SIZE * 0.8,
                "background-color": "#10b981",
                "color": "#1e293b",
                "font-size": "11px",
                "font-weight": "bold",
                "font-family": CHART_FONT_FAMILY,
                "text-valign": "bottom",
                "text-margin-y": "7px",
                "border-width": 3,
                "border-color": "#34d399",
            },
        },
        {
            "selector": "node.expanded-neighbor",
            "style": {
                "label": "data(label)",
                "width": CYTOSCAPE_EXTERNAL_NODE_SIZE,
                "height": CYTOSCAPE_EXTERNAL_NODE_SIZE,
                "background-color": "#6ee7b7",
                "color": "#475569",
                "font-size": "8px",
                "font-family": CHART_FONT_FAMILY,
                "text-valign": "bottom",
                "text-margin-y": "4px",
                "border-width": 1,
                "border-color": "#a7f3d0",
            },
        },
        {
            "selector": "edge.expanded-edge",
            "style": {
                "width": "data(norm_width)",
                "line-color": "#6ee7b7",
                "line-style": "dashed",
                "line-dash-pattern": [4, 2],
                "curve-style": "bezier",
                "opacity": 0.6,
            },
        },
        {
            "selector": "edge.expanded-edge:selected",
            "style": {
                "line-color": "#10b981",
                "opacity": 1,
            },
        },
    ])

    return styles


# -----------------------------------------------------------------------
# Evidence table builders
# -----------------------------------------------------------------------
def _build_shared_works_table(center_url: str, neighbor_url: str):
    """Build an evidence panel showing how each shared work contributes to relationship strength."""
    G = collaboration_graph
    if G is None or not G.has_edge(center_url, neighbor_url):
        return None

    edge_data = G[center_url][neighbor_url]
    works = edge_data.get("works", [])
    if not works:
        return None

    type_page = {"publication": "publish_info", "project": "project_info", "activity": "activity_info"}
    type_weight_map = {
        "publication": ACTIVITY_WEIGHT_PUBLICATION,
        "project": ACTIVITY_WEIGHT_PROJECT,
        "activity": ACTIVITY_WEIGHT_ACTIVITY,
    }
    type_color_map = {
        "publication": ("bg-indigo-100 text-indigo-700", "bg-indigo-500"),
        "project": ("bg-emerald-100 text-emerald-700", "bg-emerald-500"),
        "activity": ("bg-amber-100 text-amber-700", "bg-amber-500"),
    }

    # --- Collect topic keywords for both teachers ---
    center_topics = set(_get_teacher_topics(center_url))
    neighbor_topics = set(_get_teacher_topics(neighbor_url))
    topic_overlap = center_topics & neighbor_topics

    # --- Compute neighborhood max weight for normalization context ---
    neighbors = list(G.neighbors(center_url))
    neighbor_weights = [G[center_url][n].get("weight", 1) for n in neighbors]
    max_w = max(neighbor_weights) if neighbor_weights else 1

    # --- Enrich works ---
    rows = []
    for wk in works:
        src_type = wk.get("source_type", "")
        link = wk.get("link", "")
        title = wk.get("title", "Untitled")
        year = wk.get("year", "")
        weight = type_weight_map.get(src_type, 1)

        journal = ""
        wk_href = ""
        coauthors = []
        if src_type and link:
            df_key = f"{src_type}_df"
            df = datas.get(df_key)
            if df is not None:
                match = df[df["link"] == link]
                if not match.empty:
                    r = match.iloc[0]
                    page = type_page.get(src_type, "")
                    if page:
                        wk_href = f"/{page}?id={match.index[0]}"
                    if src_type == "publication":
                        journal = _safe_str(r.get("journal", ""))
                    elif src_type in ("project", "activity"):
                        journal = _safe_str(r.get("project_type", ""))
                    # Count co-authors on this work
                    authors_raw = r.get("authors", [])
                    if isinstance(authors_raw, list):
                        coauthors = [a.get("url", "") for a in authors_raw if isinstance(a, dict) and a.get("url")]

        rows.append({
            "title": title,
            "type": src_type,
            "type_label": src_type.capitalize() if src_type else "Unknown",
            "year": str(year) if year else "—",
            "journal": journal or "—",
            "href": wk_href,
            "weight": weight,
            "coauthor_count": len(coauthors),
        })

    # Sort by weight desc, then year desc
    rows.sort(key=lambda r: (-r["weight"], r["year"]), reverse=False)

    # --- Compute score summary ---
    total_weighted_score = sum(r["weight"] for r in rows)
    raw_edge_weight = edge_data.get("weight", len(works))
    normalized_strength = raw_edge_weight / max(max_w, 1)

    type_breakdown = {}
    for r in rows:
        t = r["type_label"]
        if t not in type_breakdown:
            type_breakdown[t] = {"count": 0, "subtotal": 0, "unit_weight": r["weight"]}
        type_breakdown[t]["count"] += 1
        type_breakdown[t]["subtotal"] += r["weight"]

    # --- Summary section ---
    summary_pills = []
    for tname, info in type_breakdown.items():
        tag_cls, _ = type_color_map.get(tname.lower(), ("bg-slate-100 text-slate-700", "bg-slate-500"))
        summary_pills.append(
            html.Span(
                f"{info['count']} {tname}{'s' if info['count'] > 1 else ''} × {info['unit_weight']}pt = {info['subtotal']}pt",
                className=f"inline-block text-xs font-medium px-2 py-0.5 rounded-full {tag_cls}",
            )
        )

    # Methodology explanation
    methodology = html.Div(className="bg-slate-50 border border-slate-200 rounded-lg p-3 mb-3", children=[
        html.Div(className="flex items-start gap-2", children=[
            html.Span("📊", className="text-base"),
            html.Div(children=[
                html.P("Relationship Strength Assessment", className="text-xs font-bold text-slate-700 uppercase tracking-wide mb-1"),
                html.P([
                    "Each co-authored work is weighted by type: ",
                    html.Span(f"Publication = {ACTIVITY_WEIGHT_PUBLICATION}pt", className="font-semibold text-indigo-600"),
                    ", ",
                    html.Span(f"Project = {ACTIVITY_WEIGHT_PROJECT}pt", className="font-semibold text-emerald-600"),
                    ", ",
                    html.Span(f"Activity = {ACTIVITY_WEIGHT_ACTIVITY}pt", className="font-semibold text-amber-600"),
                    ". The weighted total reflects how strong and multifaceted this collaboration is.",
                ], className="text-xs text-slate-500 leading-relaxed"),
            ]),
        ]),
    ])

    # Score cards
    score_cards = html.Div(className="grid grid-cols-3 gap-2 mb-3", children=[
        html.Div(className="bg-white border border-slate-200 rounded-lg p-2 text-center", children=[
            html.Div(f"{len(rows)}", className="text-lg font-bold text-slate-800"),
            html.Div("Shared Works", className="text-[10px] text-slate-400 uppercase tracking-wide"),
        ]),
        html.Div(className="bg-white border border-slate-200 rounded-lg p-2 text-center", children=[
            html.Div(f"{total_weighted_score} pt", className="text-lg font-bold text-indigo-600"),
            html.Div("Weighted Score", className="text-[10px] text-slate-400 uppercase tracking-wide"),
        ]),
        html.Div(className="bg-white border border-slate-200 rounded-lg p-2 text-center", children=[
            html.Div(f"{normalized_strength:.0%}", className="text-lg font-bold text-emerald-600"),
            html.Div(f"vs Strongest ({max_w} works)", className="text-[10px] text-slate-400 uppercase tracking-wide"),
        ]),
    ])

    # Type breakdown bar
    bar_segments = []
    for tname, info in type_breakdown.items():
        pct = info["subtotal"] / max(total_weighted_score, 1) * 100
        _, bar_cls = type_color_map.get(tname.lower(), ("", "bg-slate-400"))
        bar_segments.append(
            html.Div(
                title=f"{tname}: {info['subtotal']}pt ({pct:.0f}%)",
                className=f"h-full {bar_cls} rounded-sm",
                style={"width": f"{pct}%", "minWidth": "4px"},
            )
        )

    breakdown_bar = html.Div(className="mb-3", children=[
        html.Div(className="flex items-center gap-2 mb-1", children=[
            html.Span("Score Composition", className="text-[10px] font-bold text-slate-400 uppercase tracking-wide"),
            html.Div(className="flex gap-1.5 ml-auto", children=summary_pills),
        ]),
        html.Div(className="flex gap-0.5 h-2 bg-slate-100 rounded overflow-hidden", children=bar_segments),
    ])

    # Topic overlap section (only if both have topics)
    topic_section = None
    if center_topics or neighbor_topics:
        topic_children = []
        if topic_overlap:
            topic_children.append(
                html.Div(className="flex flex-wrap gap-1 mb-1", children=[
                    html.Span(kw, className="text-[10px] px-1.5 py-0.5 rounded bg-green-100 text-green-700 font-medium")
                    for kw in sorted(topic_overlap)
                ] + [html.Span(f"({len(topic_overlap)} shared)", className="text-[10px] text-slate-400 self-center")])
            )
        only_center = center_topics - neighbor_topics
        only_neighbor = neighbor_topics - center_topics
        if only_center:
            center_name = _safe_str(G.nodes[center_url].get("name")) or "Teacher A"
            topic_children.append(
                html.Div(className="flex flex-wrap gap-1", children=[
                    html.Span(f"{center_name} only:", className="text-[10px] text-slate-400 self-center"),
                ] + [html.Span(kw, className="text-[10px] px-1.5 py-0.5 rounded bg-slate-100 text-slate-500")
                     for kw in sorted(only_center)])
            )
        if only_neighbor:
            neighbor_name = _safe_str(G.nodes[neighbor_url].get("name")) or "Teacher B"
            topic_children.append(
                html.Div(className="flex flex-wrap gap-1", children=[
                    html.Span(f"{neighbor_name} only:", className="text-[10px] text-slate-400 self-center"),
                ] + [html.Span(kw, className="text-[10px] px-1.5 py-0.5 rounded bg-slate-100 text-slate-500")
                     for kw in sorted(only_neighbor)])
            )
        if topic_children:
            topic_section = html.Div(className="mb-3", children=[
                html.Span("Research Topic Overlap", className="text-[10px] font-bold text-slate-400 uppercase tracking-wide block mb-1"),
            ] + topic_children)

    # --- Per-work contribution table ---
    header = html.Thead(
        html.Tr([
            html.Th("#", className="text-center text-xs font-bold text-slate-500 uppercase px-2 py-2 w-8"),
            html.Th("Title", className="text-left text-xs font-bold text-slate-500 uppercase px-3 py-2"),
            html.Th("Type", className="text-center text-xs font-bold text-slate-500 uppercase px-2 py-2"),
            html.Th("Year", className="text-center text-xs font-bold text-slate-500 uppercase px-2 py-2 w-14"),
            html.Th("Weight", className="text-center text-xs font-bold text-slate-500 uppercase px-2 py-2 w-16"),
            html.Th("Contribution", className="text-center text-xs font-bold text-slate-500 uppercase px-2 py-2 w-32"),
        ]),
        className="bg-slate-50",
    )

    body_rows = []
    for i, r in enumerate(rows, 1):
        title_el = html.A(r["title"], href=r["href"],
                          className="text-indigo-600 hover:underline") if r["href"] else html.Span(r["title"])
        tag_cls, bar_cls = type_color_map.get(r["type"], ("bg-slate-100 text-slate-700", "bg-slate-400"))
        pct = r["weight"] / max(total_weighted_score, 1) * 100

        contribution_bar = html.Div(className="flex items-center gap-1.5", children=[
            html.Div(
                className=f"h-1.5 {bar_cls} rounded-full",
                style={"width": f"{pct}%", "minWidth": "4px"},
            ),
            html.Span(f"{pct:.0f}%", className="text-[10px] text-slate-400 whitespace-nowrap"),
        ])

        body_rows.append(html.Tr([
            html.Td(str(i), className="px-2 py-2 text-xs text-slate-400 text-center"),
            html.Td(html.Div([
                title_el,
                html.Div(r["journal"], className="text-[10px] text-slate-400 truncate") if r["journal"] != "—" else None,
            ]), className="px-3 py-2 text-sm text-slate-700 max-w-xs"),
            html.Td(
                html.Span(r["type_label"], className=f"text-[10px] font-medium px-1.5 py-0.5 rounded-full {tag_cls}"),
                className="px-2 py-2 text-center",
            ),
            html.Td(r["year"], className="px-2 py-2 text-xs text-slate-500 text-center"),
            html.Td(
                html.Span(f"{r['weight']} pt", className="text-xs font-semibold text-slate-700"),
                className="px-2 py-2 text-center",
            ),
            html.Td(
                html.Div(className="w-full", children=[contribution_bar]),
                className="px-2 py-2",
            ),
        ], className="border-t border-slate-100 hover:bg-slate-50/50"))

    body = html.Tbody(body_rows)

    # --- Assemble full panel ---
    children = [
        html.Div(className="flex items-center gap-2 mb-2", children=[
            html.H4("Collaboration Evidence", className="text-sm font-bold text-slate-700 uppercase tracking-wide"),
            html.Span(f"{len(rows)} works · {total_weighted_score} pt weighted score",
                      className="text-xs text-slate-400"),
        ]),
        methodology,
        score_cards,
        breakdown_bar,
    ]
    if topic_section:
        children.append(topic_section)

    children.append(
        html.Div(className="overflow-x-auto rounded-lg border border-slate-200", children=[
            html.Table([header, body], className="w-full text-left"),
        ])
    )

    return html.Div(className="mt-4", children=children)


def _build_center_summary_matrix(center_url: str):
    """Build a summary matrix of all collaborators for the center teacher."""
    G = collaboration_graph
    if G is None or center_url not in G:
        return None

    neighbors = list(G.neighbors(center_url))
    if not neighbors:
        return None

    # Sort by edge weight descending
    neighbors.sort(key=lambda n: G[center_url][n].get("weight", 0), reverse=True)
    neighbors = neighbors[:30]  # cap at 30 rows

    weights = [G[center_url][n].get("weight", 1) for n in neighbors]
    max_w = max(weights) if weights else 1

    header = html.Thead(
        html.Tr([
            html.Th("Collaborator", className="text-left text-xs font-bold text-slate-500 uppercase px-3 py-2"),
            html.Th("Department", className="text-left text-xs font-bold text-slate-500 uppercase px-3 py-2"),
            html.Th("Works", className="text-center text-xs font-bold text-slate-500 uppercase px-3 py-2"),
            html.Th("Weighted Score", className="text-center text-xs font-bold text-slate-500 uppercase px-3 py-2"),
            html.Th("Shared Topics", className="text-left text-xs font-bold text-slate-500 uppercase px-3 py-2"),
            html.Th("Strength", className="text-center text-xs font-bold text-slate-500 uppercase px-3 py-2"),
        ]),
        className="bg-slate-50",
    )

    type_weight_map = {
        "publication": ACTIVITY_WEIGHT_PUBLICATION,
        "project": ACTIVITY_WEIGHT_PROJECT,
        "activity": ACTIVITY_WEIGHT_ACTIVITY,
    }

    body_rows = []
    center_topics = set(_get_teacher_topics(center_url))
    for n in neighbors:
        attrs = G.nodes[n]
        n_name = _safe_str(attrs.get("name", "?"))
        n_dept = _safe_str(attrs.get("department", ""))
        idx = _url_to_idx.get(n)
        w = G[center_url][n].get("weight", 0)
        strength = w / max(max_w, 1)

        # Compute weighted score from individual works
        edge_works = G[center_url][n].get("works", [])
        weighted_score = sum(type_weight_map.get(wk.get("source_type", ""), 1) for wk in edge_works)

        n_topics = set(_get_teacher_topics(n))
        shared = sorted(center_topics & n_topics)[:3]

        name_el = html.A(n_name, href=f"/collaboration_network?id={idx}",
                         className="text-indigo-600 hover:underline font-medium") if idx is not None else \
                  html.Span(n_name, className="font-medium")

        # Bar indicator for strength
        bar = html.Div(className="flex items-center gap-1", children=[
            html.Div(className="w-16 bg-slate-200 rounded-full h-1.5", children=[
                html.Div(className="bg-indigo-500 h-1.5 rounded-full",
                         style={"width": f"{strength * 100:.0f}%"}),
            ]),
            html.Span(f"{strength:.0%}", className="text-xs text-slate-400 w-8"),
        ])

        body_rows.append(html.Tr([
            html.Td(name_el, className="px-3 py-2 text-sm"),
            html.Td(n_dept, className="px-3 py-2 text-xs text-slate-500 max-w-[140px] truncate"),
            html.Td(str(w), className="px-3 py-2 text-sm text-center font-semibold text-slate-700"),
            html.Td(
                html.Span(f"{weighted_score} pt", className="text-xs font-semibold text-indigo-600"),
                className="px-3 py-2 text-center",
            ),
            html.Td(html.Div([html.Span(t, className="tag is-light is-info is-small mr-1") for t in shared] if shared else "—"),
                    className="px-3 py-2"),
            html.Td(bar, className="px-3 py-2"),
        ], className="border-t border-slate-100 hover:bg-slate-50/50"))

    body = html.Tbody(body_rows)

    return html.Div(className="mt-4", children=[
        html.H4("All Collaborators Overview",
                 className="text-sm font-bold text-slate-700 uppercase tracking-wide mb-2"),
        html.Div(className="overflow-x-auto rounded-lg border border-slate-200", children=[
            html.Table([header, body], className="w-full text-left"),
        ]),
    ])


# -----------------------------------------------------------------------
# Info-panel renderers
# -----------------------------------------------------------------------
def _render_node_info_panel(data, center_url: str):
    """Render node details + evidence table."""
    name = data.get("label", "")
    href = data.get("href", "")
    tapped_url = data.get("id", "")
    is_center = data.get("is_center", False)

    name_el = html.A(name, href=href,
                      className="font-bold text-indigo-600 hover:underline text-base") if href else \
              html.Span(name, className="font-bold text-slate-800 text-base")

    info_panel = html.Div(
        className="bg-white rounded-lg border border-emerald-200 shadow-sm p-4",
        children=[
            name_el,
            html.P(data.get("role", ""), className="text-sm text-indigo-600 mt-1") if data.get("role") else None,
            html.P(data.get("department", ""), className="text-sm text-slate-500") if data.get("department") else None,
            html.Div(className="mt-2 pt-2 border-t border-slate-100", children=[
                html.Span("Research Topics: ", className="text-xs text-slate-400 font-bold uppercase"),
                html.Span(data.get("topics", "N/A"), className="text-sm text-slate-600"),
            ]) if data.get("topics") else None,
        ],
    )

    # Evidence: center node → summary matrix; neighbor → shared works table
    evidence = None
    if is_center:
        evidence = _build_center_summary_matrix(center_url)
    elif center_url and tapped_url:
        evidence = _build_shared_works_table(center_url, tapped_url)

    return html.Div([info_panel, evidence] if evidence else [info_panel])


def _render_edge_info_panel(data, center_url: str):
    """Render edge details + evidence table."""
    works_json = data.get("works_json", "[]")
    try:
        works = json.loads(works_json)
    except (json.JSONDecodeError, TypeError):
        works = []

    items = []
    for w in works[:8]:
        title = w.get("title", "Untitled")
        href = w.get("href", "")
        meta = f" ({w.get('type', '')}, {w.get('year', '')})"
        if href:
            items.append(html.Li(
                className="text-sm text-slate-600 mb-1",
                children=[
                    html.A(title, href=href, className="font-medium text-indigo-600 hover:underline"),
                    html.Span(meta, className="text-slate-400"),
                ],
            ))
        else:
            items.append(html.Li(
                className="text-sm text-slate-600 mb-1",
                children=[
                    html.Span(title, className="font-medium"),
                    html.Span(meta, className="text-slate-400"),
                ],
            ))

    info_panel = html.Div(
        className="bg-white rounded-lg border border-emerald-200 shadow-sm p-4",
        children=[
            html.H4(f"{data.get('weight', 0)} Shared Work(s)",
                     className="font-bold text-slate-800 text-base mb-2"),
            html.Ul(items, className="list-disc pl-4") if items else \
                html.P("No details available.", className="text-sm text-slate-400"),
        ],
    )

    # Build evidence table from the edge endpoints
    source_url = data.get("source", "")
    target_url = data.get("target", "")
    evidence = _build_shared_works_table(source_url, target_url)

    return html.Div([info_panel, evidence] if evidence else [info_panel])


# -----------------------------------------------------------------------
# AIO Component
# -----------------------------------------------------------------------
class CollaborationGraphAIO(html.Div):
    """
    Reusable ego-network graph with click-based info panels and evidence tables.

    Parameters
    ----------
    teacher_url : str
        The center teacher's URL in the collaboration graph.
    aio_id : str | None
        Unique ID for this instance.  Auto-generated if None.
    height : str
        CSS height of the Cytoscape canvas (default "520px").
    """

    class ids:
        graph = lambda aio_id: {
            "component": "CollabGraphAIO", "sub": "graph", "aio_id": aio_id}
        info_panel = lambda aio_id: {
            "component": "CollabGraphAIO", "sub": "info_panel", "aio_id": aio_id}
        store = lambda aio_id: {
            "component": "CollabGraphAIO", "sub": "store", "aio_id": aio_id}

    ids = ids

    def __init__(self, teacher_url: str, aio_id=None, height: str = "520px"):
        if aio_id is None:
            aio_id = str(uuid.uuid4())

        elements, stylesheet = build_ego_network_elements(teacher_url)

        if not elements:
            # No data — show placeholder
            inner = html.Div(className="mt-12", children=[
                html.H2("Collaboration Network",
                         className="text-xl font-bold text-slate-800 mb-4 border-l-4 border-indigo-500 pl-4"),
                html.P("No collaboration data available for this faculty member.",
                       className="text-slate-400 italic text-sm"),
            ])
            super().__init__(inner)
            return

        teacher_name = ""
        G = collaboration_graph
        if G and teacher_url in G:
            teacher_name = _safe_str(G.nodes[teacher_url].get("name", ""), "")

        inner = html.Div(className="mt-12", children=[
            html.H2(f"Collaboration Network{' — ' + teacher_name if teacher_name else ''}",
                     className="text-xl font-bold text-slate-800 mb-4 border-l-4 border-indigo-500 pl-4"),
            html.P("Click a node or edge to view details and evidence below.",
                   className="text-slate-400 text-sm mb-4"),
            # Store center_url for callbacks
            dcc.Store(id=self.ids.store(aio_id), data={
                "center_url": teacher_url,
                "expanded_url": None,
            }),
            html.Div(className="bg-white rounded-xl border border-slate-200 shadow-sm p-2", children=[
                cyto.Cytoscape(
                    id=self.ids.graph(aio_id),
                    elements=elements,
                    stylesheet=stylesheet,
                    layout={"name": CYTOSCAPE_LAYOUT_NAME, "animate": True},
                    style={"width": "100%", "height": height},
                    minZoom=0.3,
                    maxZoom=2.5,
                    userZoomingEnabled=False,
                    className="cyto-ctrl-zoom",
                ),
            ]),
            html.Div(id=self.ids.info_panel(aio_id), className="mt-4"),
        ])

        super().__init__(inner)


# -----------------------------------------------------------------------
# Pattern-matching callbacks (auto-registered on import)
# -----------------------------------------------------------------------
@callback(
    Output(CollaborationGraphAIO.ids.info_panel(MATCH), "children"),
    Output(CollaborationGraphAIO.ids.graph(MATCH), "elements"),
    Output(CollaborationGraphAIO.ids.graph(MATCH), "layout"),
    Output(CollaborationGraphAIO.ids.store(MATCH), "data"),
    Input(CollaborationGraphAIO.ids.graph(MATCH), "tapNodeData"),
    State(CollaborationGraphAIO.ids.store(MATCH), "data"),
    prevent_initial_call=True,
)
def _aio_tap_node(data, store_data):
    if not data:
        return no_update, no_update, no_update, no_update

    center_url = store_data.get("center_url", "") if store_data else ""
    current_expanded = store_data.get("expanded_url") if store_data else None
    tapped_url = data.get("id", "")
    is_center = data.get("is_center", False)

    # Always build the info panel (unchanged logic)
    info = _render_node_info_panel(data, center_url)

    G = collaboration_graph
    # Check if tapped node is a direct neighbor of center in the full graph
    is_direct_neighbor = (
        not is_center
        and G is not None
        and center_url in G
        and tapped_url in G
        and G.has_edge(center_url, tapped_url)
    )

    if is_center:
        # Clicking center node: collapse any expansion
        if current_expanded:
            new_elements, _ = build_ego_network_elements(center_url)
            new_store = {**store_data, "expanded_url": None}
            return info, new_elements, {"name": CYTOSCAPE_LAYOUT_NAME, "animate": True}, new_store
        return info, no_update, no_update, no_update

    if is_direct_neighbor:
        if current_expanded == tapped_url:
            # Toggle off: clicking the already-expanded node collapses it
            new_elements, _ = build_ego_network_elements(center_url)
            new_store = {**store_data, "expanded_url": None}
            return info, new_elements, {"name": CYTOSCAPE_LAYOUT_NAME, "animate": True}, new_store
        else:
            # Expand this node (replaces any previous expansion)
            new_elements, _ = build_ego_network_elements(center_url, expanded_url=tapped_url)
            new_store = {**store_data, "expanded_url": tapped_url}
            return info, new_elements, {"name": CYTOSCAPE_LAYOUT_NAME, "animate": True}, new_store

    # Non-direct neighbor (e.g. expanded node's own neighbor) — just show info
    return info, no_update, no_update, no_update


@callback(
    Output(CollaborationGraphAIO.ids.info_panel(MATCH), "children", allow_duplicate=True),
    Input(CollaborationGraphAIO.ids.graph(MATCH), "tapEdgeData"),
    State(CollaborationGraphAIO.ids.store(MATCH), "data"),
    prevent_initial_call=True,
)
def _aio_tap_edge(data, store_data):
    if not data:
        return no_update
    center_url = store_data.get("center_url", "") if store_data else ""
    return _render_edge_info_panel(data, center_url)
