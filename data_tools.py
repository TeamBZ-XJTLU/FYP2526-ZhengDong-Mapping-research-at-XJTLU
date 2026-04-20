import random
import pandas as pd
from data.data_parser import datas, datalinks

def get_associated_info(item_type, row_dict):
    """
    Extract associated information (teachers and their other activities/projects/publications)
    to enrich the prompt context for a more comprehensive AI summary.
    item_type can be 'activity', 'project', 'publication', or 'teacher'.
    """
    associated_info_list = []
    source_link = row_dict.get('link') or row_dict.get('url')
    if not source_link:
        return ""
        
    teacher_urls = []
    
    if item_type == 'teacher':
        teacher_urls = [source_link]
    else:
        rel_df_name = f'author_{item_type}_relation_df'
        if rel_df_name in datalinks:
            rel_df = datalinks[rel_df_name]
            matching_rows = rel_df[rel_df['source_link'] == source_link]
            teacher_urls = matching_rows['teacher_url'].dropna().unique().tolist()

    if not teacher_urls:
        return ""
        
    teacher_df = datas.get('teacher_df')
    if teacher_df is None or teacher_df.empty:
        return ""
        
    # Limit to top 3 teachers to avoid exceeding prompt length limits
    for t_url in teacher_urls[:3]:
        t_rows = teacher_df[teacher_df['url'] == t_url]
        if t_rows.empty:
            continue
        t_row = t_rows.iloc[0]
        
        t_name = t_row.get('name', 'Unknown Researcher')
        t_title = t_row.get('role', '')
        t_areas = t_row.get('research_areas', [])
        
        info = f"\n### Associated Researcher: {t_name}"
        if t_title and str(t_title) != 'nan':
            info += f" ({t_title})"
        if isinstance(t_areas, list) and t_areas:
            info += f"\n- Research Areas: {', '.join(t_areas)}"
            
        other_works = []
        
        # 1. Other Publications
        if 'author_publication_relation_df' in datalinks:
            pub_rel = datalinks['author_publication_relation_df']
            t_pubs_links = pub_rel[pub_rel['teacher_url'] == t_url]['source_link'].tolist()
            if t_pubs_links:
                pub_df = datas.get('publication_df')
                if pub_df is not None and not pub_df.empty:
                    other_pubs = pub_df[pub_df['link'].isin(t_pubs_links) & (pub_df['link'] != source_link)]
                    titles = other_pubs['title'].dropna().tolist()[:3]
                    if titles:
                        other_works.append(f"- Recent Publications: {'; '.join(titles)}")
                        
        # 2. Other Projects
        if 'author_project_relation_df' in datalinks:
            proj_rel = datalinks['author_project_relation_df']
            t_proj_links = proj_rel[proj_rel['teacher_url'] == t_url]['source_link'].tolist()
            if t_proj_links:
                proj_df = datas.get('project_df')
                if proj_df is not None and not proj_df.empty:
                    other_projs = proj_df[proj_df['link'].isin(t_proj_links) & (proj_df['link'] != source_link)]
                    titles = other_projs['title'].dropna().tolist()[:3]
                    if titles:
                        other_works.append(f"- Recent Projects: {'; '.join(titles)}")
                        
        # 3. Other Activities
        if 'author_activity_relation_df' in datalinks:
            act_rel = datalinks['author_activity_relation_df']
            t_act_links = act_rel[act_rel['teacher_url'] == t_url]['source_link'].tolist()
            if t_act_links:
                act_df = datas.get('activity_df')
                if act_df is not None and not act_df.empty:
                    other_acts = act_df[act_df['link'].isin(t_act_links) & (act_df['link'] != source_link)]
                    titles = other_acts['title'].dropna().tolist()[:3]
                    if titles:
                        other_works.append(f"- Recent Activities: {'; '.join(titles)}")
                        
        if other_works:
            info += "\n" + "\n".join(other_works)
            
        associated_info_list.append(info)
        
    if associated_info_list:
        return "\n\n**Additional Context (Associated Researchers & their other works)**:\n" + "\n".join(associated_info_list)
    return ""


# ── AI-driven Query Execution Engine ────────────────────────────────────────

def _apply_condition(field_value, condition: dict) -> bool:
    """Evaluate a single filter condition against a field value."""
    op = condition.get('op', 'icontains')
    value = condition.get('value')

    if field_value is None:
        return False
    if isinstance(field_value, float) and pd.isna(field_value):
        return False
    if value is None:
        return False

    if op == 'icontains':
        return str(value).lower() in str(field_value).lower()

    elif op == 'icontains_list':
        search = str(value).lower()
        if isinstance(field_value, list):
            return any(search in str(item).lower() for item in field_value)
        return search in str(field_value).lower()

    elif op == 'eq_icase':
        return str(field_value).strip().lower() == str(value).strip().lower()

    elif op == 'gt':
        try:
            return float(field_value) > float(value)
        except (ValueError, TypeError):
            return False

    elif op == 'gte':
        try:
            return float(field_value) >= float(value)
        except (ValueError, TypeError):
            return False

    elif op == 'lt':
        try:
            return float(field_value) < float(value)
        except (ValueError, TypeError):
            return False

    elif op == 'lte':
        try:
            return float(field_value) <= float(value)
        except (ValueError, TypeError):
            return False

    return False


def _apply_table_filter(df: pd.DataFrame, conditions: list, logic: str = 'AND') -> pd.DataFrame:
    """Apply a list of filter conditions to a DataFrame using AND / OR logic."""
    if not conditions or df.empty:
        return df

    masks = []
    for condition in conditions:
        field = condition.get('field')
        if field not in df.columns:
            continue
        mask = df[field].apply(lambda v, c=condition: _apply_condition(v, c))
        masks.append(mask)

    if not masks:
        return df

    combined = masks[0]
    for m in masks[1:]:
        combined = (combined | m) if logic == 'OR' else (combined & m)

    return df[combined]


def execute_ai_query(query_json: dict, k: int = 30) -> dict:
    """
    Execute the structured query plan returned by the Step 1 LLM.

    Expected query_json schema::

        {
          "intent": "...",
          "teacher_filter": {
            "enabled": bool,
            "conditions": [{"field": str, "op": str, "value": any}, ...],
            "logic": "AND" | "OR",
            "fetch_all_teacher_content": bool
          },
          "table_queries": [
            {
              "table": "publication_df" | "project_df" | "activity_df",
              "conditions": [...],
              "filter_logic": "AND" | "OR",
              "teacher_join": bool
            }
          ]
        }

    Returns: {table_name: [row_dicts], ...}
    """
    results: dict = {}

    teacher_filter = query_json.get('teacher_filter', {})
    table_queries = query_json.get('table_queries', [])

    matched_teacher_urls: list = []

    # ── A: Apply teacher filter ──────────────────────────────────────────────
    if teacher_filter.get('enabled'):
        teacher_df = datas.get('teacher_df')
        if teacher_df is not None and not teacher_df.empty:
            conditions = teacher_filter.get('conditions', [])
            logic = teacher_filter.get('logic', 'AND')
            filtered_teachers = _apply_table_filter(teacher_df, conditions, logic)
            matched_teacher_urls = filtered_teachers['url'].dropna().unique().tolist()

            if not filtered_teachers.empty:
                rows = filtered_teachers.to_dict('records')
                results['teacher_df'] = random.sample(rows, min(k, len(rows)))

    # ── B: Fetch all content for matched teachers ────────────────────────────
    if teacher_filter.get('fetch_all_teacher_content') and matched_teacher_urls:
        _content_rel_map = [
            ('project_df',     'author_project_relation_df'),
            ('publication_df', 'author_publication_relation_df'),
            ('activity_df',    'author_activity_relation_df'),
        ]
        for table_name, rel_name in _content_rel_map:
            rel_df = datalinks.get(rel_name)
            content_df = datas.get(table_name)
            if rel_df is None or content_df is None or content_df.empty:
                continue
            matched_links = rel_df[
                rel_df['teacher_url'].isin(matched_teacher_urls)
            ]['source_link'].unique()
            matched_rows = content_df[content_df['link'].isin(matched_links)]
            if not matched_rows.empty:
                rows = matched_rows.to_dict('records')
                results[table_name] = random.sample(rows, min(k, len(rows)))

    # ── C: Process each explicit table_query ────────────────────────────────
    _rel_map = {
        'project_df':     'author_project_relation_df',
        'publication_df': 'author_publication_relation_df',
        'activity_df':    'author_activity_relation_df',
    }

    for tq in table_queries:
        table_name = tq.get('table')
        content_df = datas.get(table_name)
        if content_df is None or content_df.empty:
            continue

        conditions = tq.get('conditions', [])
        filter_logic = tq.get('filter_logic', 'AND')
        teacher_join = tq.get('teacher_join', False)

        filtered_df = _apply_table_filter(content_df, conditions, filter_logic) if conditions else content_df.copy()

        # Restrict to records authored by matched teachers if requested
        if teacher_join and matched_teacher_urls:
            rel_name = _rel_map.get(table_name)
            if rel_name and rel_name in datalinks:
                rel_df = datalinks[rel_name]
                allowed_links = rel_df[
                    rel_df['teacher_url'].isin(matched_teacher_urls)
                ]['source_link'].unique()
                filtered_df = filtered_df[filtered_df['link'].isin(allowed_links)]

        if filtered_df.empty:
            continue

        new_rows = filtered_df.to_dict('records')

        # Merge with existing results for this table (union, deduplicate by link)
        if table_name in results:
            existing_links = {r.get('link') for r in results[table_name]}
            deduped = [r for r in new_rows if r.get('link') not in existing_links]
            merged = results[table_name] + deduped
            results[table_name] = random.sample(merged, min(k, len(merged)))
        else:
            results[table_name] = random.sample(new_rows, min(k, len(new_rows)))

    return results

