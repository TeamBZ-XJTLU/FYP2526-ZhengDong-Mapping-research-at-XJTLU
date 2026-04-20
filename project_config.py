# ============================================================
# Centralized Configuration for Data Mining & Visualization
# ============================================================

# --- BERTopic Model Configuration ---
BERTOPIC_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BERTOPIC_MIN_TOPIC_SIZE = 10
BERTOPIC_NR_TOPICS = "auto"          # "auto" or an integer
BERTOPIC_TOP_N_WORDS = 10
BERTOPIC_NGRAM_RANGE = (1, 3)        # unigrams + bigrams + trigrams for coherent phrases
BERTOPIC_MIN_ABSTRACT_LENGTH = 30    # skip abstracts shorter than this

# --- Topic Trend Configuration ---
TREND_TOP_KEYWORDS_PER_TEACHER = 8
TREND_MIN_DOCS_FOR_TEACHER = 3       # teachers with fewer docs get a warning
TREND_TOP_KEYWORDS_PER_DEPARTMENT = 10

# --- Collaboration Network Configuration ---
COLLAB_TOP_ACTIVE_TEACHERS = 24
COLLAB_MAX_DISPLAY_NEIGHBORS = 15
COLLAB_COMMUNITY_RESOLUTION = 1.0    # Louvain resolution parameter
COLLAB_POTENTIAL_TOP_K = 10          # top-K potential collaborators per teacher
COLLAB_SIMILARITY_ALPHA = 0.6        # weight for topic similarity
COLLAB_SIMILARITY_BETA = 0.4         # weight for Adamic-Adar score

# --- Research Activity Score Weights ---
ACTIVITY_WEIGHT_PUBLICATION = 3
ACTIVITY_WEIGHT_PROJECT = 2
ACTIVITY_WEIGHT_ACTIVITY = 1

# --- Visualization Configuration ---
CHART_FONT_FAMILY = "Inter, sans-serif"
CHART_FONT_SIZE = 10
CHART_TITLE_FONT_SIZE = 14
CHART_COLOR_PALETTE = [
    "#6366f1",  # indigo-500
    "#10b981",  # emerald-500
    "#f59e0b",  # amber-500
    "#ef4444",  # red-500
    "#8b5cf6",  # violet-500
    "#06b6d4",  # cyan-500
    "#ec4899",  # pink-500
    "#14b8a6",  # teal-500
    "#f97316",  # orange-500
    "#64748b",  # slate-500
    "#a855f7",  # purple-500
    "#22d3ee",  # cyan-400
]
CHART_MARGIN = dict(t=50, b=20, l=30, r=30)

# --- Cytoscape Network Graph Styling ---
CYTOSCAPE_LAYOUT_NAME = "cose"
CYTOSCAPE_CENTER_NODE_SIZE = 60
CYTOSCAPE_NEIGHBOR_NODE_SIZE = 35
CYTOSCAPE_EXTERNAL_NODE_SIZE = 25
CYTOSCAPE_EDGE_WIDTH_RANGE = (1, 6)   # min/max edge width based on weight

# Community colors for Cytoscape nodes
COMMUNITY_COLORS = [
    "#6366f1", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6",
    "#06b6d4", "#ec4899", "#14b8a6", "#f97316", "#64748b",
    "#a855f7", "#22d3ee", "#84cc16", "#fb923c", "#818cf8",
]
