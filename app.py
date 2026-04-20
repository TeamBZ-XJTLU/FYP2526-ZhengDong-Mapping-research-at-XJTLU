import dash
from dash import Dash, html, page_container, dcc, Input, Output
from layouts.layout import sidebar_layout, main_layout
from dash_tailwindcss_plugin import setup_tailwindcss_plugin


app = Dash(
    __name__,
    use_pages=True,
    suppress_callback_exceptions=True,
    external_scripts=[
    ],
    external_stylesheets=[
    ]
)
app.title = "Academic Output Dashboard"

# 根据不同的 URL 路径动态渲染不同的页面内容
app.layout = html.Div([
    sidebar_layout(),
    dash.page_container 
])




if __name__ == "__main__":
    app.run(debug=True)
