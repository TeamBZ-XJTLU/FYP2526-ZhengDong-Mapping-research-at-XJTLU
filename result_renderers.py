"""
Renderers for different types of search results
"""
from dash import html
import re


def highlight_keyword(text, keyword, context_words=15):
    """
    Highlight keyword in text and extract context around it
    
    Args:
        text: The text to search in
        keyword: The keyword to highlight
        context_words: Number of words to show before and after the keyword
    
    Returns:
        List of Dash components with highlighted keyword
    """
    if not text or not keyword:
        return []
    
    text_str = str(text)
    keyword_lower = keyword.lower()
    
    # Find the position of the keyword (case-insensitive)
    match = re.search(re.escape(keyword_lower), text_str.lower())
    if not match:
        return []
    
    start_pos = match.start()
    end_pos = match.end()
    
    # Extract context: find word boundaries
    # Go backwards to find context_words words before
    words_before = []
    temp_pos = start_pos - 1
    while temp_pos >= 0 and len(words_before) < context_words:
        if text_str[temp_pos].isspace():
            # Found a word boundary, extract the word
            word_start = temp_pos + 1
            word_end = start_pos
            for i in range(temp_pos + 1, start_pos):
                if not text_str[i].isspace():
                    word_start = i
                    break
            if word_start < start_pos:
                words_before.insert(0, text_str[word_start:start_pos])
                start_pos = word_start
        temp_pos -= 1
    
    # Go forwards to find context_words words after
    words_after = []
    temp_pos = end_pos
    while temp_pos < len(text_str) and len(words_after) < context_words:
        if text_str[temp_pos].isspace():
            # Found a word boundary
            word_end = temp_pos
            for i in range(end_pos, temp_pos):
                if not text_str[i].isspace():
                    word_end = temp_pos
                    break
            if end_pos < word_end:
                words_after.append(text_str[end_pos:word_end])
                end_pos = word_end
        temp_pos += 1
    
    # If text is too long, extract context
    if len(text_str) > 200:
        # Find better boundaries
        context_start = max(0, start_pos - 100)
        context_end = min(len(text_str), end_pos + 100)
        
        # Adjust to word boundaries
        while context_start > 0 and not text_str[context_start].isspace():
            context_start -= 1
        while context_end < len(text_str) and not text_str[context_end].isspace():
            context_end += 1
        
        before_text = text_str[context_start:match.start()].strip()
        keyword_text = text_str[match.start():match.end()]
        after_text = text_str[match.end():context_end].strip()
        
        components = []
        if context_start > 0:
            components.append(html.Span("... ", className="text-slate-400"))
        components.append(html.Span(before_text, className="text-slate-600"))
        components.append(html.Mark(keyword_text, className="bg-yellow-200 px-1 font-semibold"))
        components.append(html.Span(after_text, className="text-slate-600"))
        if context_end < len(text_str):
            components.append(html.Span(" ...", className="text-slate-400"))
        
        return components
    else:
        # Short text, just highlight the keyword
        before_text = text_str[:match.start()]
        keyword_text = text_str[match.start():match.end()]
        after_text = text_str[match.end():]
        
        return [
            html.Span(before_text, className="text-slate-600"),
            html.Mark(keyword_text, className="bg-yellow-200 px-1 font-semibold"),
            html.Span(after_text, className="text-slate-600")
        ]


def render_matched_field_preview(field_name, field_value, keyword):
    """
    Render a preview of the matched field with highlighted keyword
    
    Args:
        field_name: Name of the field that matched
        field_value: Value of the field
        keyword: The search keyword
    
    Returns:
        Dash HTML component
    """
    if field_value is None:
        return html.Div()
    
    # Convert lists to string
    if isinstance(field_value, list):
        field_value = ', '.join([str(v) for v in field_value])
    
    field_value_str = str(field_value)
    
    # Special style for BERTopic mined keywords
    if field_name == "mined_keyword":
        highlighted = highlight_keyword(field_value_str, keyword)
        if not highlighted:
            # If keyword doesn't literally appear, just show the mined keywords
            highlighted = [html.Span(field_value_str, className="text-emerald-700")]
        return html.Div(className="mt-3 p-3 bg-emerald-50 rounded-lg border border-emerald-200", children=[
            html.P(className="text-xs text-emerald-600 uppercase font-semibold mb-1", children=[
                "🔬 BERTopic Mined Keyword"
            ]),
            html.P(className="text-sm", children=highlighted)
        ])

    # Create highlighted preview
    highlighted = highlight_keyword(field_value_str, keyword)
    
    if not highlighted:
        return html.Div()
    
    return html.Div(className="mt-3 p-3 bg-slate-50 rounded-lg border border-slate-200", children=[
        html.P(className="text-xs text-slate-500 uppercase font-semibold mb-1", children=[
            f"Found in: {field_name.replace('_', ' ').title()}"
        ]),
        html.P(className="text-sm", children=highlighted)
    ])


def render_teacher_result(result, keyword, teacher_data):
    """Render teacher search result card with detailed information"""
    
    # Type tag
    tag = html.Span("👤 Faculty", className="tag is-info text-sm")
    
    # Basic info
    name = teacher_data.get('name', 'Unknown')
    role = teacher_data.get('role', '')
    department = teacher_data.get('department', '')
    email = teacher_data.get('email', '')
    citation = teacher_data.get('citation', 0)
    h_index = teacher_data.get('h_index', 0)
    
    # Metrics
    metrics = []
    if citation and citation > 0:
        metrics.append(
            html.Div(className="flex items-center gap-1", children=[
                html.Span("📊", className="text-sm"),
                html.Span(f"{int(citation)} citations", className="text-slate-600 text-xs")
            ])
        )
    if h_index and h_index > 0:
        metrics.append(
            html.Div(className="flex items-center gap-1", children=[
                html.Span("📈", className="text-sm"),
                html.Span(f"h-index: {int(h_index)}", className="text-slate-600 text-xs")
            ])
        )
    
    # Find the first matched field to show preview
    matched_field = result.get('matched_fields', [])[0] if result.get('matched_fields') else None
    field_preview = html.Div()
    if matched_field:
        field_value = teacher_data.get(matched_field)
        field_preview = render_matched_field_preview(matched_field, field_value, keyword)
    
    return html.A(
        href=result['link'],
        target="_blank",
        className="block bg-white rounded-xl shadow hover:shadow-xl transition-all duration-300 p-6 border border-slate-100 hover:border-indigo-300",
        children=[
            # Header
            html.Div(className="flex items-start justify-between mb-3", children=[
                tag,
                html.Div(className="flex gap-2", children=metrics)
            ]),
            
            # Name
            html.H3(name, className="text-xl font-bold text-slate-900 mb-2"),
            
            # Role
            html.P(role, className="text-indigo-600 text-sm mb-1") if role else html.Div(),
            
            # Department
            html.P(department, className="text-slate-500 text-sm mb-2") if department else html.Div(),
            
            # Email
            html.P(email, className="text-slate-400 text-xs") if email else html.Div(),
            
            # Matched field preview
            field_preview
        ]
    )


def render_activity_result(result, keyword, activity_data):
    """Render activity search result card with detailed information"""
    
    tag = html.Span("📅 Activity", className="tag is-success text-sm")
    
    # Basic info
    title = activity_data.get('title', 'Untitled Activity')
    project_type = activity_data.get('project_type', '')
    event_type = activity_data.get('event_type', '')
    period = activity_data.get('period', '')
    location = activity_data.get('location', '')
    year = activity_data.get('year', '')
    
    # Authors
    authors_raw = activity_data.get('authors', [])
    authors_str = ''
    if isinstance(authors_raw, list) and len(authors_raw) > 0:
        authors_str = ', '.join([a.get('name', '') for a in authors_raw[:3]])
        if len(authors_raw) > 3:
            authors_str += f' +{len(authors_raw) - 3} more'
    
    # Metadata tags
    metadata_tags = []
    if event_type:
        metadata_tags.append(html.Span(event_type, className="tag is-light is-small"))
    if year:
        metadata_tags.append(html.Span(str(year), className="tag is-light is-small"))
    
    # Find matched field preview
    matched_field = result.get('matched_fields', [])[0] if result.get('matched_fields') else None
    field_preview = html.Div()
    if matched_field:
        field_value = activity_data.get(matched_field)
        field_preview = render_matched_field_preview(matched_field, field_value, keyword)
    
    return html.A(
        href=result['link'],
        target="_blank",
        className="block bg-white rounded-xl shadow hover:shadow-xl transition-all duration-300 p-6 border border-slate-100 hover:border-green-300",
        children=[
            # Header
            html.Div(className="flex items-start justify-between mb-3", children=[
                tag,
                html.Div(className="flex flex-wrap gap-1", children=metadata_tags)
            ]),
            
            # Title
            html.H3(
                title, 
                className="text-xl font-bold text-slate-900 mb-2",
                style={'display': '-webkit-box', 'WebkitLineClamp': '2', 'WebkitBoxOrient': 'vertical', 'overflow': 'hidden'}
            ),
            
            # Authors
            html.P(authors_str, className="text-indigo-600 text-sm mb-2") if authors_str else html.Div(),
            
            # Type and period
            html.Div(className="flex flex-wrap gap-2 text-sm text-slate-500 mb-1", children=[
                html.Span(f"📌 {project_type}") if project_type else html.Div(),
                html.Span(f"📅 {period}") if period else html.Div(),
            ]),
            
            # Location
            html.P(f"📍 {location}", className="text-slate-400 text-xs") if location else html.Div(),
            
            # Matched field preview
            field_preview
        ]
    )


def render_publication_result(result, keyword, publication_data):
    """Render publication search result card with detailed information"""
    
    tag = html.Span("📄 Publication", className="tag is-warning text-sm")
    
    # Basic info
    title = publication_data.get('title', 'Untitled Publication')
    journal = publication_data.get('journal', '')
    conference = publication_data.get('conference', '')
    year = publication_data.get('year', '')
    volume = publication_data.get('volume', '')
    issue_number = publication_data.get('issue_number', '')
    pages = publication_data.get('pages', '')
    
    # Authors
    authors_raw = publication_data.get('authors', [])
    authors_str = ''
    if isinstance(authors_raw, list) and len(authors_raw) > 0:
        authors_str = ', '.join([a.get('name', '') for a in authors_raw[:3]])
        if len(authors_raw) > 3:
            authors_str += ' et al.'
    
    # Venue (journal or conference)
    venue = journal if journal else conference
    
    # Publication details
    pub_details = []
    if year:
        pub_details.append(str(year))
    if volume:
        pub_details.append(f"Vol. {volume}")
    if issue_number:
        pub_details.append(f"Issue {issue_number}")
    if pages:
        pub_details.append(f"pp. {pages}")
    
    pub_details_str = ' | '.join(pub_details)
    
    # Find matched field preview
    matched_field = result.get('matched_fields', [])[0] if result.get('matched_fields') else None
    field_preview = html.Div()
    if matched_field:
        field_value = publication_data.get(matched_field)
        field_preview = render_matched_field_preview(matched_field, field_value, keyword)
    
    return html.A(
        href=result['link'],
        target="_blank",
        className="block bg-white rounded-xl shadow hover:shadow-xl transition-all duration-300 p-6 border border-slate-100 hover:border-yellow-300",
        children=[
            # Header
            html.Div(className="flex items-start justify-between mb-3", children=[
                tag,
                html.Span(str(year), className="tag is-light") if year else html.Div()
            ]),
            
            # Title
            html.H3(
                title, 
                className="text-xl font-bold text-slate-900 mb-2",
                style={'display': '-webkit-box', 'WebkitLineClamp': '2', 'WebkitBoxOrient': 'vertical', 'overflow': 'hidden'}
            ),
            
            # Authors
            html.P(authors_str, className="text-indigo-600 text-sm mb-2") if authors_str else html.Div(),
            
            # Venue
            html.P(venue, className="text-slate-600 text-sm font-medium mb-1") if venue else html.Div(),
            
            # Publication details
            html.P(pub_details_str, className="text-slate-400 text-xs") if pub_details_str else html.Div(),
            
            # Matched field preview
            field_preview
        ]
    )


def render_project_result(result, keyword, project_data):
    """Render project search result card with detailed information"""
    
    tag = html.Span("🔬 Project", className="tag is-danger text-sm")
    
    # Basic info
    title = project_data.get('title', 'Untitled Project')
    project_type = project_data.get('project_type', '')
    status = project_data.get('status', '')
    fund = project_data.get('fund')
    year = project_data.get('year', '')
    desc_date_0 = project_data.get('desc_date_0', '')
    
    # Authors
    authors_raw = project_data.get('authors', [])
    authors_str = ''
    if isinstance(authors_raw, list) and len(authors_raw) > 0:
        authors_str = ', '.join([a.get('name', '') for a in authors_raw[:3]])
        if len(authors_raw) > 3:
            authors_str += f' +{len(authors_raw) - 3} more'
    
    # Status tag
    status_tag = html.Div()
    if status:
        status_class = "tag is-success is-light" if status == "Finished" else "tag is-info is-light"
        status_tag = html.Span(status, className=status_class + " text-xs")
    
    # Project details
    project_details = []
    if fund and fund > 0:
        project_details.append(f"💰 ¥{int(fund):,}")
    if desc_date_0:
        project_details.append(f"📅 {desc_date_0}")
    
    # Find matched field preview
    matched_field = result.get('matched_fields', [])[0] if result.get('matched_fields') else None
    field_preview = html.Div()
    if matched_field:
        field_value = project_data.get(matched_field)
        field_preview = render_matched_field_preview(matched_field, field_value, keyword)
    
    return html.A(
        href=result['link'],
        target="_blank",
        className="block bg-white rounded-xl shadow hover:shadow-xl transition-all duration-300 p-6 border border-slate-100 hover:border-red-300",
        children=[
            # Header
            html.Div(className="flex items-start justify-between mb-3", children=[
                tag,
                status_tag
            ]),
            
            # Title
            html.H3(
                title, 
                className="text-xl font-bold text-slate-900 mb-2",
                style={'display': '-webkit-box', 'WebkitLineClamp': '2', 'WebkitBoxOrient': 'vertical', 'overflow': 'hidden'}
            ),
            
            # Authors
            html.P(authors_str, className="text-indigo-600 text-sm mb-2") if authors_str else html.Div(),
            
            # Project type
            html.P(f"📂 {project_type}", className="text-slate-600 text-sm mb-1") if project_type else html.Div(),
            
            # Project details (fund and date)
            html.Div(className="flex flex-wrap gap-3 text-sm text-slate-500", children=[
                html.Span(detail) for detail in project_details
            ]) if project_details else html.Div(),
            
            # Matched field preview
            field_preview
        ]
    )
