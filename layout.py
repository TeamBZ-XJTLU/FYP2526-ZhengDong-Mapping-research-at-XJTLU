import dash
from dash import html, dcc
from data.data_parser import datas,datalinks,sorted_departments
import pandas as pd
import os

def sidebar_layout():
    departments = sorted_departments
    
    return html.Div(
        className="fixed top-0 left-0 h-screen w-64 bg-slate-900 text-slate-100 flex flex-col shadow-xl",
        children=[
            # Header
            html.Div(
                className="p-6 border-b border-slate-800",
                children=[
                    html.H2("Academic Analysis", className="text-xl font-bold text-indigo-400"),
                    html.P("XJTLU Dashboard", className="text-[10px] text-slate-500 uppercase tracking-widest")
                ]
            ),
            
            # 页面导航
            html.Div(className="px-2 py-4 text-xs font-semibold text-slate-500 uppercase", children="Main Pages"),
            html.Nav(
                className="px-4 space-y-1",
                children=[
                    dcc.Link(
                        page['name'], 
                        href=page['relative_path'], 
                        className="flex items-center p-2 rounded-md hover:bg-slate-800 transition-colors no-underline text-slate-300 text-sm"
                    ) for page in dash.page_registry.values()
                ]
            ),

            # 部门筛选列表 (可滚动区域)
            html.Div(className="px-2 py-4 mt-4 border-t border-slate-800 text-xs font-semibold text-slate-500 uppercase", children="Departments"),
            html.Div(
                className="flex-1 overflow-y-auto px-4 py-2 custom-scrollbar", # custom-scrollbar 可以在 assets/style.css 定义
                children=[
                    dcc.Link(
                        dept,
                        href=f"/?dept={dept}", # 通过查询参数传递部门
                        className="block p-2 text-xs text-slate-400 hover:text-indigo-400 hover:bg-slate-800/50 rounded transition-all no-underline"
                    ) for dept in departments
                ]
            ),

            html.Div(className="p-4 bg-slate-950 text-[10px] text-slate-600", children="© 2026 FYP Project")
        ]
    )

def main_layout(content):
    return html.Div(
        className="ml-64 min-h-screen bg-slate-50",
        children=[
            html.Header(
                className="h-14 bg-white border-b border-slate-200 flex items-center justify-between px-8 shadow-sm",
                children=[
                    html.Span("Research Insights", className="text-sm font-semibold text-slate-600"),
                    html.Div(className="flex space-x-4", children=[
                        html.Span("Live Data", className="tag is-info is-light text-[10px]") # 使用 Bulma Tag
                    ])
                ]
            ),
            html.Main(className="p-8", children=content)
        ]
    )