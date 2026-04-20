import dash
from dash import html, dcc, Input, Output, callback, no_update
import pandas as pd
import plotly.express as px
import random
from layouts.layout import main_layout
from urllib.parse import parse_qs
from data.data_parser import datas, datalinks, teacher_topic_trends
from pages.teacher_info.prompt import get_teacher_ai_prompt
from utils.ui_components import AISummaryAIO
from utils.collab_graph_component import CollaborationGraphAIO
from data.data_parser import collaboration_graph
from utils.project_config import CHART_FONT_FAMILY, CHART_FONT_SIZE, CHART_TITLE_FONT_SIZE, CHART_MARGIN, CHART_COLOR_PALETTE

# 注册页面
dash.register_page(
    __name__, 
    path="/teacher_info", 
    name="Faculty Profile"
)

# 页面基础布局
layout = main_layout([
    dcc.Location(id='teacher-url-manager', refresh=False),
    
    html.Div(
        id="loading-teacher",
        children=html.Div(id='teacher-detail-content', className="p-4 md:p-8")
    )
])

# 将随机挑选一个valid行的id的逻辑封装成一个函数，供回调使用
def select_valid_id(df):
    if df is not None and len(df) > 0:
        valid_indices = df[df['personal_profile'].apply(lambda x: isinstance(x, str) and len(x) > 50)].index
        if len(valid_indices) > 0:
            return random.choice(valid_indices)
    return random.randint(0, len(df) - 1)



# 回调 1: 处理自动跳转逻辑 (如果没有提供 ID，随机选择一位老师)
@callback(
    Output('teacher-url-manager', 'search'),
    Input('teacher-url-manager', 'search')
)
def redirect_if_no_teacher_id(search):
    if not search or 'id=' not in search:
        df = datas.get('teacher_df')
        if df is not None and len(df) > 0:
            random_idx = select_valid_id(df)
            return f"?id={random_idx}"
    return no_update

# 回调 2: 渲染教师详细内容
@callback(
    Output('teacher-detail-content', 'children'),
    Input('teacher-url-manager', 'search')
)
def render_teacher_content(search):
    if not search:
        return html.Div("Loading Profile...", className="text-slate-400 italic")

    # 1. 解析 ID 参数
    params = parse_qs(search.lstrip('?'))
    idx_str = params.get('id', [None])[0]
    
    try:
        idx = int(idx_str)
        df = datas['teacher_df']
        if idx < 0 or idx >= len(df):
            raise IndexError
        row = df.iloc[idx]
    except (ValueError, TypeError, IndexError):
        return html.Div(
            className="notification is-danger is-light",
            children="Invalid Faculty ID. Please return to the overview page."
        )

    # 2. 数据处理辅助函数
    def get_val(key, default=None):
        val = row.get(key)
        if val is None: return default
        if isinstance(val, float) and pd.isna(val): return default
        if isinstance(val, (str, list, dict)) and len(val) == 0: return default
        return val

    # 3. 构建页面组件
    
    # 头像与基本信息
    name = get_val('name', 'Unknown Name')
    role = get_val('role', 'Faculty Member')
    dept = get_val('department', 'XJTLU')
    img_url = get_val('image_url', 'https://via.placeholder.com/150?text=No+Image')
    
    header_section = html.Div(className="flex flex-col md:flex-row gap-8 items-start mb-12", children=[
        html.Img(src=img_url, 
                 key=f"teacher-img-{idx}",
                 className="w-40 h-40 rounded-2xl object-cover shadow-lg border-4 border-white"),
        html.Div(className="flex-1", children=[
            html.H1(name, className="text-4xl font-black text-slate-900 mb-2"),
            html.P(role, className="text-xl text-indigo-600 font-medium mb-1"),
            html.P(dept, className="text-slate-500 mb-6 flex items-center gap-2"),
            html.Div(className="flex flex-wrap gap-4", children=[
                html.A(f"✉ {get_val('email', 'N/A')}", href=f"mailto:{get_val('email', '')}", className="button is-small is-light is-rounded") if get_val('email') else None,
                html.Span(f"📞 {get_val('phone')}", className="tag is-light is-medium py-4") if get_val('phone') else None,
                html.A("Pure Profile ↗", href=get_val('url', '#'), target="_blank", className="button is-small is-link is-outlined is-rounded")
            ])
        ])
    ])

    # 学术统计卡片
    stats_grid = html.Div(className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-12", children=[
        html.Div(className="p-6 bg-white border border-slate-100 rounded-2xl shadow-sm text-center", children=[
            html.Label("Citations", className="text-[10px] uppercase font-bold text-slate-400 tracking-widest block mb-2"),
            html.Div(str(int(get_val('citation', 0))), className="text-3xl font-black text-slate-800")
        ]),
        html.Div(className="p-6 bg-white border border-slate-100 rounded-2xl shadow-sm text-center", children=[
            html.Label("H-Index", className="text-[10px] uppercase font-bold text-slate-400 tracking-widest block mb-2"),
            html.Div(str(int(get_val('h_index', 0))), className="text-3xl font-black text-indigo-600")
        ]),
        html.Div(className="p-6 bg-white border border-slate-100 rounded-2xl shadow-sm text-center col-span-2", children=[
            html.Label("Research Areas", className="text-[10px] uppercase font-bold text-slate-400 tracking-widest block mb-2"),
            html.Div(className="flex flex-wrap justify-center gap-2", children=[
                html.Span(area, className="tag is-info is-light") for area in get_val('research_areas', [])
            ] if get_val('research_areas') else [html.Span("General Research", className="text-slate-400")])
        ]),
    ])

    # 个人简介
    profile_text = get_val('personal_profile')
    profile_section = html.Div(className="mb-12", children=[
        html.H2("Biography", className="text-xl font-bold text-slate-800 mb-4 border-l-4 border-indigo-500 pl-4"),
        html.P(profile_text, className="text-slate-600 leading-relaxed")
    ]) if profile_text else None

    # 列表类信息辅助函数 (用于渲染 Education, Experience, Teaching 等)
    def render_list_section(title, items, icon="•"):
        if not items: return None
        return html.Div(className="mb-8", children=[
            html.H3(title, className="text-sm font-bold text-slate-400 uppercase tracking-wider mb-4"),
            html.Ul(className="space-y-3", children=[
                html.Li(className="flex items-start gap-3 text-slate-700", children=[
                    html.Span(icon, className="text-indigo-500 font-bold"),
                    html.Span(item)
                ]) for item in items
            ])
        ])

    # 详细履历网格
    details_grid = html.Div(className="grid grid-cols-1 md:grid-cols-2 gap-12 mb-12", children=[
        html.Div([
            render_list_section("Education", get_val('education'), "🎓"),
            render_list_section("Experience", get_val('experience'), "💼"),
        ]),
        html.Div([
            render_list_section("Teaching", get_val('teaching'), "📖"),
            render_list_section("Awards & Honours", get_val('awards_and_honours'), "🏆"),
        ])
    ])

    # 研究关键词 (Fingerprint) - 根据权重调整透明度
    fingerprint_raw = get_val('fingerprint', [])
    fingerprint_section = None
    if fingerprint_raw:
        tags = []
        for item in fingerprint_raw[:30]: # 最多显示30个
            word, score = item[0], item[1]
            # 根据分数计算样式：分数越高，颜色越深
            opacity = max(0.3, min(1, score))
            tags.append(html.Span(word, className="tag is-medium is-rounded m-1", style={
                "backgroundColor": f"rgba(99, 102, 241, {opacity})",
                "color": "white" if opacity > 0.6 else "#4338ca",
                "fontWeight": "bold" if score > 0.8 else "normal"
            }))
        
        fingerprint_section = html.Div(className="mb-12 p-8 bg-slate-50 rounded-3xl", children=[
            html.H2("Research Fingerprint", className="text-xl font-bold text-slate-800 mb-6 text-center"),
            html.Div(className="flex flex-wrap justify-center", children=tags)
        ])

    # Extracted Fingerprint (BERTopic-mined topics)
    extracted_fp_section = None
    current_url_fp = get_val('url')
    if teacher_topic_trends is not None and current_url_fp:
        t_fp = teacher_topic_trends[teacher_topic_trends['teacher_url'] == current_url_fp]
        if not t_fp.empty:
            kw_weights = t_fp.groupby('keyword')['weight'].sum().nlargest(20)
            if not kw_weights.empty:
                max_w = kw_weights.max()
                fp_tags = []
                for kw, w in kw_weights.items():
                    opacity = max(0.3, min(1.0, w / max_w))
                    fp_tags.append(html.Span(kw, className="tag is-medium is-rounded m-1", style={
                        "backgroundColor": f"rgba(16, 185, 129, {opacity})",
                        "color": "white" if opacity > 0.5 else "#065f46",
                        "fontWeight": "bold" if opacity > 0.7 else "normal",
                    }))
                extracted_fp_section = html.Div(className="mb-12 p-8 bg-emerald-50/50 rounded-3xl", children=[
                    html.H2("Extracted Fingerprint", className="text-xl font-bold text-slate-800 mb-2 text-center"),
                    html.P("Research topics extracted via BERTopic analysis", className="text-xs text-slate-400 text-center mb-4"),
                    html.Div(className="flex flex-wrap justify-center", children=fp_tags),
                ])

    # 相似专家
    similar = get_val('similar_profiles', [])
    similar_section = html.Div(className="mt-12 pt-8 border-t border-slate-100", children=[
        html.H3("Similar Faculty Profiles", className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-4"),
        html.Div(className="flex flex-wrap gap-2", children=[
            html.Span(p, className="tag is-light is-rounded") for p in similar
        ])
    ]) if similar else None

    # 相关项目推荐部分 (利用 author_publication_relation_df 查找该教师相关的 publications)
    rel_df = datalinks.get('author_publication_relation_df')
    pub_df = datas.get('publication_df')
    related_section = None
    if rel_df is not None and pub_df is not None:
        current_url = get_val('url')
        if current_url:
            related_links = rel_df[rel_df['teacher_url'] == current_url]['source_link'].unique()
            if len(related_links) > 0:
                related_idx = pub_df.index[pub_df['link'].isin(related_links)].tolist()
                if related_idx:
                    display_idx = random.sample(related_idx, min(4, len(related_idx)))
                    cards = []
                    for ridx in display_idx:
                        r_row = pub_df.iloc[ridx]
                        r_title = r_row.get('title', 'Unknown Title')
                        if r_title and isinstance(r_title, str):
                            cards.append(
                                html.A(
                                    href=f"/publish_info?id={ridx}",
                                    target="_blank",
                                    className="block p-4 bg-white border border-slate-200 rounded-lg shadow-sm hover:shadow-md hover:border-indigo-300 transition-all",
                                    children=[
                                        html.H4(r_title, className="font-bold text-slate-800 text-sm truncate mb-1"),
                                        html.P(r_row.get('journal', r_row.get('publisher', 'Publication')), className="text-xs text-slate-500 truncate")
                                    ]
                                )
                            )
                    
                    if cards:
                        related_section = html.Div(className="mt-12", children=[
                            html.H2("Related Publications", className="text-xl font-bold text-slate-800 mb-4 flex items-center border-l-4 border-indigo-500 pl-4"),
                            html.Div(className="grid grid-cols-1 md:grid-cols-2 gap-4", children=cards)
                        ])

    # AI 智能分析组件
    system_prompt, user_content = get_teacher_ai_prompt(row.to_dict())
    ai_section = AISummaryAIO(system_prompt, user_content, aio_id="teacher-info", estimated_seconds=30)

    # --- Collaboration Network (reusable AIO component) ---
    collab_section = None
    current_url = get_val('url')
    if current_url and collaboration_graph is not None and current_url in collaboration_graph:
        collab_section = CollaborationGraphAIO(
            teacher_url=current_url,
            aio_id=f"teacher-collab-{idx}",
            height="420px",
        )

    # --- Tutor Research Trend (BERTopic-mined topic evolution) ---
    trend_section = None
    if teacher_topic_trends is not None and current_url:
        t_trends = teacher_topic_trends[teacher_topic_trends['teacher_url'] == current_url]
        if not t_trends.empty:
            t_trends = t_trends.sort_values('year')
            fig_trend = px.line(
                t_trends, x='year', y='weight', color='keyword',
                markers=True,
                color_discrete_sequence=CHART_COLOR_PALETTE,
            )
            fig_trend.update_layout(
                margin=CHART_MARGIN,
                font=dict(family=CHART_FONT_FAMILY, size=CHART_FONT_SIZE),
                title_font=dict(size=CHART_TITLE_FONT_SIZE),
                legend_title_text='Keyword',
                xaxis_title='Year',
                yaxis_title='Topic Weight',
                xaxis=dict(dtick=1),
                plot_bgcolor='white',
                paper_bgcolor='white',
            )
            trend_section = html.Div(className='mt-12', children=[
                html.H2('Tutor Research Trend',
                         className='text-xl font-bold text-slate-800 mb-4 border-l-4 border-indigo-500 pl-4'),
                html.Div(
                    className='bg-white p-6 rounded-xl border border-slate-200 shadow-sm',
                    children=dcc.Graph(figure=fig_trend, config={'displayModeBar': False})
                )
            ])
        else:
            trend_section = html.Div(className='mt-12', children=[
                html.H2('Tutor Research Trend',
                         className='text-xl font-bold text-slate-800 mb-4 border-l-4 border-indigo-500 pl-4'),
                html.P('Insufficient publication data for topic trend analysis.',
                       className='text-slate-400 italic text-sm')
            ])

    # 底部返回
    footer = html.Div(className="mt-16 flex justify-between items-center", children=[
        html.A("← Back to Dashboard", href="/", className="button is-text text-slate-500 no-underline hover:text-indigo-600 px-0"),
        html.Button("View Random Faculty ⟳", id="btn-random-teacher", className="button is-light is-rounded")
    ])

    return html.Div(className="animate-in fade-in slide-in-from-bottom-4 duration-500", children=[
        # 面包屑
        html.Nav(className="breadcrumb mb-8 text-xs", children=[
            html.Ul([
                html.Li(html.A("Research Dashboard", href="/")),
                html.Li(className="is-active", children=html.A("Faculty Profile", href="#"))
            ])
        ]),

        header_section,
        stats_grid,
        profile_section,
        details_grid,
        fingerprint_section,
        extracted_fp_section,
        similar_section,
        
        # 相关发表物
        related_section,

        # AI 智能分析
        ai_section,

        # Collaboration Network
        collab_section,

        # Research topic evolution
        trend_section,
        
        footer
    ])

# 回调 3: 随机教师跳转
@callback(
    Output('teacher-url-manager', 'search', allow_duplicate=True),
    Input('btn-random-teacher', 'n_clicks'),
    prevent_initial_call=True
)
def handle_random_teacher_click(n_clicks):
    if n_clicks:
        df = datas.get('teacher_df')
        random_idx = select_valid_id(df)
        return f"?id={random_idx}"
    return no_update