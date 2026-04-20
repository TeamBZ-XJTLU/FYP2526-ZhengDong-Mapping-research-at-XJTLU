import uuid
from dash import html, dcc, Input, Output, State, callback, no_update, MATCH
from utils.ai_parser import get_ai_response

def get_ai_progress_ui(estimated_seconds: int = 20):
    """
    Returns a fake progress bar component for AI generation waiting state.
    """
    return html.Div(
        className="flex flex-col items-center justify-center py-6", 
        children=[
            html.I(className="fas fa-magic text-2xl mb-4 text-indigo-500 animate-bounce"),
            html.Div(f"Generating AI Analysis... ", className="text-sm text-indigo-600 font-bold mb-4 animate-pulse tracking-wide"),
            html.Div(className="w-full max-w-lg bg-slate-200/80 rounded-full h-2.5 mb-2 overflow-hidden", children=[
                html.Div(
                    className="bg-gradient-to-r from-indigo-500 to-purple-500 h-2.5 rounded-full", 
                    style={"animation": f"ai-progress {estimated_seconds}s ease-out forwards"}
                )
            ])
        ]
    )

class AISummaryAIO(html.Div):
    """
    A reusable All-in-One component for the AI generation section.
    Encapsulates the placeholder UI, the progress animation, and the backend data-fetching callbacks.
    """
    class ids:
        trigger = lambda aio_id: {
            'component': 'AISummaryAIO',
            'subcomponent': 'trigger',
            'aio_id': aio_id
        }
        container = lambda aio_id: {
            'component': 'AISummaryAIO',
            'subcomponent': 'container',
            'aio_id': aio_id
        }
        store = lambda aio_id: {
            'component': 'AISummaryAIO',
            'subcomponent': 'store',
            'aio_id': aio_id
        }
        fetch_trigger = lambda aio_id: {
            'component': 'AISummaryAIO',
            'subcomponent': 'fetch_trigger',
            'aio_id': aio_id
        }

    ids = ids

    def __init__(self, system_prompt: str, user_content: str, aio_id=None, estimated_seconds=30):
        if aio_id is None:
            aio_id = str(uuid.uuid4())
            
        layout = html.Div(className="mt-16 pt-8 border-t border-slate-100", children=[
            html.Div(
                className="flex items-center gap-2 mb-6",
                children=[
                    html.Span(className="icon is-small text-indigo-500", children=html.I(className="fas fa-magic")),
                    html.H2("AI Smart Analysis", className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-indigo-500 to-purple-600 m-0")
                ]
            ),
            dcc.Store(id=self.ids.store(aio_id), data={"system_prompt": system_prompt, "user_content": user_content, "estimated_seconds": estimated_seconds}),
            dcc.Store(id=self.ids.fetch_trigger(aio_id), data=0),
            html.Div(
                id=self.ids.container(aio_id), 
                className="p-6 bg-gradient-to-br from-indigo-50/80 to-purple-50/50 rounded-2xl border border-indigo-100 shadow-sm transition-all", 
                children=html.Div(
                    id=self.ids.trigger(aio_id),
                    n_clicks=0,
                    className="cursor-pointer flex flex-col items-center justify-center py-6 text-black hover:text-indigo-600 transition-colors",
                    children=[
                        html.I(className="fas fa-sparkles text-3xl mb-3"),
                        html.Strong("Click here to generate AI analysis...", className="italic text-sm tracking-wide text-black")
                    ]
                )
            )
        ])
        
        super().__init__(layout.children, className=layout.className)


@callback(
    Output(AISummaryAIO.ids.container(MATCH), 'children', allow_duplicate=True),
    Output(AISummaryAIO.ids.fetch_trigger(MATCH), 'data'),
    Input(AISummaryAIO.ids.trigger(MATCH), 'n_clicks'),
    State(AISummaryAIO.ids.store(MATCH), 'data'),
    prevent_initial_call=True
)
def show_aio_progress(n_clicks, store_data):
    if not n_clicks:
        return no_update, no_update
        
    estimated_seconds = store_data.get("estimated_seconds", 30)
    return get_ai_progress_ui(estimated_seconds=estimated_seconds), 1


@callback(
    Output(AISummaryAIO.ids.container(MATCH), 'children'),
    Input(AISummaryAIO.ids.fetch_trigger(MATCH), 'data'),
    State(AISummaryAIO.ids.store(MATCH), 'data'),
    prevent_initial_call=True
)
def fetch_aio_summary(trigger, store_data):
    if not trigger:
        return no_update
        
    system_prompt = store_data.get("system_prompt")
    user_content = store_data.get("user_content")
    
    success, final_resp = get_ai_response(system_prompt, user_content)
    
    if not success:
        return html.Div(
            className="p-5 mt-2 bg-red-50/50 border border-red-200 text-red-600 rounded-xl flex items-start gap-4",
            children=[
                html.Span(className="icon mt-1", children=html.I(className="fas fa-exclamation-circle text-red-500")),
                html.Div([
                    html.Strong("AI Analysis Currently Unavailable", className="block mb-1 text-red-700"),
                    html.P(final_resp, className="text-sm font-mono opacity-90 m-0 break-words")
                ])
            ]
        )

    return dcc.Markdown(
        final_resp,
        className="ai-markdown-content max-w-none text-slate-700 leading-relaxed font-sans mt-2"
    )
