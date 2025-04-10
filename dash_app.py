import dash
from dash import dcc, html, dash_table, Input, Output, State, callback_context
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pathlib import Path
import os
import json
import plotly.express as px # Add plotly express
from collections import defaultdict # Import defaultdict

# --- Configuration ---
INPUT_DIR = Path("input")
CALIBRATION_FILE = Path("calibration.csv") # Define calibration file path
X_COLUMN = "RT [min]"
Y_COLUMN = "Area"
AMINO_ACID_LABELS = [
    "Asp", "Glu", "Cys", "Asn", "Ser", "Gln", "His", "IS", 
    "Gly", "Thr", "Arg", "Ala", "Tyr", "Val", "Met", "Trp", 
    "Phe", "Ile", "Leu", "Lys"
] # Standard amino acid order provided by user

# Colors
STD_COLOR = 'red'
OTHER_COLOR = 'grey'
SELECTED_COLOR = '#FF9800'  # Orange for selected/removed points
MANUAL_ASSIGN_COLOR = 'green' # Color for manually assigned points
UNASSIGNED_COLOR = '#FF9800' # Orange for points needing assignment

# --- Find CSV files ---
def find_csv_files(directory: Path) -> list[Path]:
    """Finds all CSV files in the specified directory."""
    return list(directory.glob("*.csv"))

def create_figure(selected_files_paths, unassigned_points=None, manual_assignments_log=None):
    """Create Plotly figure from selected CSV files, styling unassigned and manually assigned points."""
    if unassigned_points is None:
        unassigned_points = {}
    if manual_assignments_log is None:
        manual_assignments_log = {}

    all_data = []
    combined_df_list = []

    for file_path in selected_files_paths:
        try:
            df = pd.read_csv(file_path)
            if X_COLUMN not in df.columns or Y_COLUMN not in df.columns:
                print(f"File '{file_path.name}' is missing required columns. Skipping.")
                continue
            df['File'] = file_path.name
            df['point_id'] = [f"{file_path.name}_{i}" for i in range(len(df))]
            all_data.append(df)
            combined_df_list.append(df)
        except Exception as e:
            print(f"Error reading file '{file_path.name}': {e}")

    if not all_data:
        return go.Figure(layout=dict(title="No valid data could be loaded")), pd.DataFrame()

    fig = go.Figure()
    all_data.sort(key=lambda df: df['File'].iloc[0])

    std_files_data = []
    other_files_data = []
    for df in all_data:
        filename = df['File'].iloc[0]
        if 'std' in filename.lower():
            std_files_data.append(df)
        else:
            other_files_data.append(df)

    manually_assigned_pids = set(manual_assignments_log.keys())

    # Plot other files
    for df in other_files_data:
        file_label = df['File'].iloc[0]

        # --- Categorize points ---
        is_unassigned = df['point_id'].apply(lambda x: x in unassigned_points)
        is_manual = df['point_id'].apply(lambda x: x in manually_assigned_pids)
        is_regular = ~(is_unassigned | is_manual)

        df_regular = df[is_regular]
        df_unassigned = df[is_unassigned]
        df_manual = df[is_manual]

        # --- Plot regular points (gray) ---
        if not df_regular.empty:
            # Stem lines
            x_stems, y_stems = [], []
            for _, row in df_regular.iterrows():
                x, y = row[X_COLUMN], row[Y_COLUMN]
                x_stems.extend([x, x, None])
                y_stems.extend([0, y, None])
            fig.add_trace(go.Scatter(x=x_stems, y=y_stems, mode='lines', line=dict(color=OTHER_COLOR, width=0.5), name=file_label, legendgroup=file_label, showlegend=False))
            # Markers
            fig.add_trace(go.Scatter(
                x=df_regular[X_COLUMN], y=df_regular[Y_COLUMN], mode='markers',
                marker=dict(color=OTHER_COLOR, size=4),
                name=file_label, legendgroup=file_label, showlegend=True,
                customdata=df_regular['point_id'],
                hovertemplate=f"{X_COLUMN}: %{{x}}<br>{Y_COLUMN}: %{{y}}<br>ID: %{{customdata}}<extra>{file_label}</extra>"
            ))

        # --- Plot unassigned points (orange 'x') ---
        if not df_unassigned.empty:
            # Stem lines
            x_stems_un, y_stems_un = [], []
            for _, row in df_unassigned.iterrows():
                x, y = row[X_COLUMN], row[Y_COLUMN]
                x_stems_un.extend([x, x, None])
                y_stems_un.extend([0, y, None])
            fig.add_trace(go.Scatter(x=x_stems_un, y=y_stems_un, mode='lines', line=dict(color=UNASSIGNED_COLOR, width=0.5, dash='dot'), name=f"{file_label} (Unassigned)", legendgroup=f"{file_label}_unassigned", showlegend=False))
            # Markers
            fig.add_trace(go.Scatter(
                x=df_unassigned[X_COLUMN], y=df_unassigned[Y_COLUMN], mode='markers',
                marker=dict(color=UNASSIGNED_COLOR, size=5, symbol='x'),
                name=f"{file_label} (Unassigned)", legendgroup=f"{file_label}_unassigned", showlegend=True,
                customdata=df_unassigned['point_id'],
                hovertemplate=f"{X_COLUMN}: %{{x}}<br>{Y_COLUMN}: %{{y}}<br>ID: %{{customdata}}<extra>{file_label} (Unassigned - Click to Assign)</extra>"
            ))

        # --- Plot manually assigned points (green circle) ---
        if not df_manual.empty:
             # Stem lines
            x_stems_man, y_stems_man = [], []
            for _, row in df_manual.iterrows():
                x, y = row[X_COLUMN], row[Y_COLUMN]
                x_stems_man.extend([x, x, None])
                y_stems_man.extend([0, y, None])
            fig.add_trace(go.Scatter(x=x_stems_man, y=y_stems_man, mode='lines', line=dict(color=MANUAL_ASSIGN_COLOR, width=0.5), name=f"{file_label} (Manual)", legendgroup=f"{file_label}_manual", showlegend=False))
            # Markers
            manual_hover_texts = []
            for pid in df_manual['point_id']:
                assigned_aa = manual_assignments_log.get(pid, {}).get('aa', 'Unknown') # Get AA from log
                manual_hover_texts.append(f"Manually Assigned: {assigned_aa}")

            fig.add_trace(go.Scatter(
                x=df_manual[X_COLUMN], y=df_manual[Y_COLUMN], mode='markers',
                marker=dict(color=MANUAL_ASSIGN_COLOR, size=6, symbol='circle'),
                name=f"{file_label} (Manual)", legendgroup=f"{file_label}_manual", showlegend=True,
                customdata=df_manual['point_id'],
                hovertext=manual_hover_texts,
                hovertemplate=f"{X_COLUMN}: %{{x}}<br>{Y_COLUMN}: %{{y}}<br>ID: %{{customdata}}<br>%{{hovertext}}<extra>{file_label}</extra>"
            ))

    # NEW: Sort standard files by minimum retention time (using all points, not filtered)
    std_files_with_min_rt = []
    for df in std_files_data:
        if not df.empty:
            min_rt = df[X_COLUMN].min()
            std_files_with_min_rt.append((df, min_rt))
    std_files_with_min_rt.sort(key=lambda x: x[1])
    sorted_std_files = [item[0] for item in std_files_with_min_rt]

    # Plot Amino Acid Labels based on sorted standard files
    if sorted_std_files and len(AMINO_ACID_LABELS) > 0:
        # --- AA Label logic (remains largely the same) ---
        all_retention_times = []
        for std_df in sorted_std_files:
            all_retention_times.extend(std_df[X_COLUMN].tolist())
        all_retention_times.sort()

        min_rt_diff = 0.1
        distinct_rts = []
        for rt in all_retention_times:
            if not distinct_rts or rt - distinct_rts[-1] >= min_rt_diff:
                distinct_rts.append(rt)

        num_labels = len(AMINO_ACID_LABELS)
        num_positions = len(distinct_rts)

        for idx in range(min(num_labels, num_positions)):
            rt_pos = distinct_rts[idx]
            aa_label = AMINO_ACID_LABELS[idx]
            fig.add_trace(go.Scatter(
                x=[rt_pos], y=[0], mode='text', text=[aa_label],
                textposition='bottom center',
                textfont=dict(family='Arial', size=10, color='black'),
                showlegend=False, hoverinfo='skip'
            ))

        if num_labels > num_positions and num_positions > 0:
            last_rt = distinct_rts[-1]
            rt_step = 1.0
            for idx in range(num_positions, num_labels):
                label_pos = last_rt + rt_step * (idx - num_positions + 1)
                aa_label = AMINO_ACID_LABELS[idx]
                fig.add_trace(go.Scatter(
                    x=[label_pos], y=[0], mode='text', text=[aa_label],
                    textposition='bottom center',
                    textfont=dict(family='Arial', size=10, color='black'),
                    showlegend=False, hoverinfo='skip'
                ))
    else:
         # Fallback label placement if no std files
        base_rt = 5
        rt_step = 2
        for idx, aa_label in enumerate(AMINO_ACID_LABELS):
            label_pos = base_rt + (idx * rt_step)
            fig.add_trace(go.Scatter(
                 x=[label_pos], y=[0], mode='text', text=[aa_label],
                 textposition='bottom center',
                 textfont=dict(family='Arial', size=10, color='black'),
                 showlegend=False, hoverinfo='skip'
             ))

    # Plot standard files (red lines)
    for df in sorted_std_files:
        file_label = df['File'].iloc[0] + " (Std)"

        # Standard files are not manually assigned or unassigned, just plot them
        # Stem lines
        x_stems_std, y_stems_std = [], []
        for _, row in df.iterrows():
            x, y = row[X_COLUMN], row[Y_COLUMN]
            x_stems_std.extend([x, x, None])
            y_stems_std.extend([0, y, None])
        fig.add_trace(go.Scatter(x=x_stems_std, y=y_stems_std, mode='lines', line=dict(color=STD_COLOR, width=1), name=file_label, legendgroup=file_label, showlegend=False))
        # Markers
        fig.add_trace(go.Scatter(
            x=df[X_COLUMN], y=df[Y_COLUMN], mode='markers',
            marker=dict(color=STD_COLOR, size=6),
            name=file_label, legendgroup=file_label, showlegend=True,
            customdata=df['point_id'],
            hovertemplate=f"{X_COLUMN}: %{{x}}<br>{Y_COLUMN}: %{{y}}<br>ID: %{{customdata}}<extra>{file_label}</extra>"
        ))

    # Customize the Plotly layout
    fig.update_layout(
        xaxis_title="Retention Time (min)",
        yaxis_title="Area",
        yaxis_range=[0, None],
        legend_title="Files",
        yaxis_gridcolor='lightgrey', yaxis_gridwidth=1,
        xaxis_gridcolor='lightgrey', xaxis_gridwidth=1,
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=50, b=40),
        xaxis=dict(type='linear'),
        yaxis=dict(constrain='domain'),
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"),
        hovermode="closest",
        showlegend=False
    )

    combined_df = pd.concat(combined_df_list) if combined_df_list else pd.DataFrame()
    return fig, combined_df

def generate_assignment_table(final_assignments, manual_log, sample_files, std_rt_map):
    """Generates the DataTable component for AA assignments."""
    print("Generating Assignment Table...") # Placeholder print
    # TODO: Implement table generation logic here
    # Combine auto and manual, handle overrides
    # Create DataFrame based on final_assignments, std_rt_map, manual_log info
    # Return DataTable component

    # Placeholder Implementation:
    aa_df = pd.DataFrame(index=AMINO_ACID_LABELS)
    std_rt_column = [f"{std_rt_map.get(aa, '--'):.3f}" if std_rt_map.get(aa) is not None else '--' for aa in AMINO_ACID_LABELS]
    aa_df['Standard RT'] = std_rt_column

    for sample_file in sample_files:
        rt_column = []
        file_assignments = final_assignments.get(sample_file, {})
        for aa in AMINO_ACID_LABELS:
            data = file_assignments.get(aa)
            if data:
                rt_val = data.get('rt')
                assign_type = data.get('type', 'unknown')
                if rt_val is not None:
                    rt_str = f"{rt_val:.3f}"
                    if assign_type == 'manual':
                        rt_str += " (M)"
                else:
                    rt_str = "(M?)" if assign_type == 'manual' else "(Auto?)"
                rt_column.append(rt_str)
            else:
                rt_column.append("--")
        aa_df[f"{sample_file} (RT)"] = rt_column

    aa_df.reset_index(inplace=True)
    aa_df.rename(columns={'index': 'Amino Acid'}, inplace=True)

    table = dash_table.DataTable(
         id='aa-assignment-table',
         columns=[
             {'name': 'Amino Acid', 'id': 'Amino Acid'},
             *[{ 'name': col, 'id': col } for col in aa_df.columns if col != 'Amino Acid']
         ],
         data=aa_df.to_dict('records'),
         style_table={'overflowX': 'auto'},
         style_cell={'textAlign': 'center', 'padding': '8px', 'width': 'auto', 'whiteSpace': 'normal', 'height': 'auto'},
         style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
         style_data_conditional=[
             {'if': {'column_id': 'Amino Acid'}, 'textAlign': 'left', 'fontWeight': 'bold', 'backgroundColor': 'rgb(240, 240, 240)'},
             {'if': {'column_id': 'Standard RT'}, 'backgroundColor': 'rgb(245, 245, 245)'}
         ],
         export_format="csv",
     )
    # Create explanatory text and download button container
    result_container = html.Div([
        html.H3("Amino Acid Assignments", style={'marginBottom': '15px'}),
        html.P([
            "Assignments based on RT proximity (≤ 0.5 min) to standards. ",
            "(M) indicates a manual assignment."
            # Add count of unassigned if needed? Need unassigned store data here.
        ]),
        table,
        html.Div([
            html.Button(
                "Export Assignments Table",
                id="export-aa-assignments", # Keep this ID
                style={'backgroundColor': '#2196F3', 'color': 'white', 'padding': '8px 12px', 'border': 'none', 'cursor': 'pointer', 'borderRadius': '4px', 'marginTop': '15px'}
            ),
            html.Button( # NEW Confirm Button
                "Confirm Assignments & Calculate Normalized Area",
                id="confirm-assignments-button",
                n_clicks=0,
                style={'backgroundColor': '#4CAF50', 'color': 'white', 'padding': '8px 12px', 'border': 'none', 'cursor': 'pointer', 'borderRadius': '4px', 'marginTop': '15px', 'marginLeft': '10px'}
            ),
            html.Div(id="export-aa-status", style={'marginTop': '10px'}) # Keep this ID
        ])
    ])
    return result_container

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True,
                external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css']) # Added basic CSS
server = app.server


# Create app layout
app.layout = html.Div(
    style={'backgroundColor': 'white', 'color': 'black', 'fontFamily': 'Arial, sans-serif', 'padding': '20px'},
    children=[
        # --- Stores ---
        dcc.Store(id='last-zoom-state', data=None),
        dcc.Store(id='amino-acid-assignments', data={}), # Holds final combined auto/manual {file: {aa: {data...}}}
        dcc.Store(id='std-rt-map-store', data={}), # Holds {aa: rt} map from standards
        dcc.Store(id='unassigned-points-store', data={}), # Holds {point_id: True} for unassigned sample points
        dcc.Store(id='manual-assign-point-id-store', data=None), # Holds point_id being manually assigned
        dcc.Store(id='manual-assignments-log-store', data={}), # Holds {point_id: {'aa': aa, 'rt': rt, 'area': area}}
        dcc.Store(id='combined-data-for-assignment-store', data={}), # Holds {point_id: {'File': f, X_COL: rt, Y_COL: area}}
        # RENAMED store: Holds results list of dicts [{Sample, AA, NormArea, Conc}, ...]
        dcc.Store(id='results-data-store', data=None),

        # --- Tabs ---
        dcc.Tabs(id='tabs', value='tab-plot', children=[
            dcc.Tab(label='Plot & Results', value='tab-plot', children=[ # Renamed tab
                html.Div([
                    # --- Plot & Results Area ---
                    html.Div(
                        id='plot-container',
                        style={'width': '100%', 'padding': '10px', 'boxSizing': 'border-box'}, # Make plot container full width
                        children=[
                            dcc.Graph(
                                id='chromatogram-plot',
                                style={'height': 'calc(100vh - 150px)'}, # Adjust height slightly?
                                config={
                                    'scrollZoom': True,
                                    'displayModeBar': 'hover',
                                    'modeBarButtonsToRemove': ['select2d', 'lasso2d', 'zoomIn', 'zoomOut', 'autoScale', 'hoverClosestCartesian', 'hoverCompareCartesian'], # Removed select/lasso
                                    'toImageButtonOptions': {'format': 'png', 'filename': 'chromatogram_plot', 'height': 800, 'width': 1200, 'scale': 2}
                                }
                            ),
                            # Manual Assignment Controls
                            html.Div(
                                id='manual-assign-controls',
                                style={'display': 'none', 'padding': '15px', 'border': '1px solid #eee', 'borderRadius': '5px', 'marginTop': '10px'}, # Slightly styled
                                children=[
                                    html.P(id='manual-assign-prompt', style={'fontWeight': 'bold', 'marginBottom': '10px'}),
                                    dcc.Dropdown(
                                        id='manual-assign-dropdown',
                                        options=[{'label': aa, 'value': aa} for aa in AMINO_ACID_LABELS] + [{'label': 'Unassign Point', 'value': 'UNASSIGN'}], # Add unassign option
                                        placeholder="Select Amino Acid or Unassign...",
                                        style={'width': '250px', 'marginBottom': '10px'}
                                    ),
                                    html.Div(id='manual-assign-validation-message', style={'color': 'red', 'marginTop': '5px', 'fontSize': 'small'}),
                                    html.Button('Assign', id='manual-assign-button', n_clicks=0, style={'marginRight': '10px', 'backgroundColor': '#4CAF50', 'color': 'white', 'border': 'none', 'padding': '5px 10px', 'borderRadius': '3px'}),
                                    html.Button('Cancel', id='manual-assign-cancel-button', n_clicks=0, style={'backgroundColor': '#f44336', 'color': 'white', 'border': 'none', 'padding': '5px 10px', 'borderRadius': '3px'})
                                ]
                            ),
                            # Assignment Table Results (after Auto-Assign)
                            html.Div(
                                id='aa-assignment-results',
                                style={'marginTop': '20px', 'display': 'none'} # Initially hidden
                            ),
                            # Normalized Area & Concentration Results (after Confirm)
                            html.Div(id='results-tables-container', style={'marginTop': '20px'}), # UPDATED div for results tables + export
                        ]
                    )
                ], style={'width': '100%'}), # Removed flex style
            ]),
            dcc.Tab(label='Raw Data', value='tab-data', children=[ # Renamed tab
                html.Div([
                    html.H3("Raw Data Table", style={'marginBottom': '20px'}),
                    html.Div([
                        html.P("Showing combined data from all files."), # Updated text
                        html.Button('Export Raw Data', id='export-button', n_clicks=0, style={'backgroundColor': '#2196F3', 'color': 'white', 'padding': '8px 12px', 'border': 'none', 'cursor': 'pointer', 'borderRadius': '4px', 'marginBottom': '20px'}),
                        html.Div(id='export-status'),
                    ]),
                    html.Div(id='data-table-container')
                ], style={'padding': '20px'})
            ])
        ]),
        html.Div(style={'clear': 'both'})
    ]
)


# --- MODIFIED Callback to update plot and data table ---
@app.callback(
    [Output('chromatogram-plot', 'figure'),
     Output('data-table-container', 'children'),
     Output('combined-data-for-assignment-store', 'data')], # ADDED output for combined data
    [Input('tabs', 'value'), # Trigger on tab change (includes initial load)
     Input('unassigned-points-store', 'data'), # ADDED
     Input('manual-assignments-log-store', 'data')], # ADDED
    [State('last-zoom-state', 'data')]
)
def update_plot_and_data(active_tab,
                         unassigned_points, manual_assignments, # ADDED inputs
                         last_zoom_state):
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'No ID'
    print(f"update_plot_and_data triggered by: {triggered_id}") # Debug print

    # Initialize inputs if None
    unassigned_points = unassigned_points or {}
    manual_assignments = manual_assignments or {}

    # Always use all found CSV files
    all_csv_files = find_csv_files(INPUT_DIR)
    selected_files_paths = sorted(all_csv_files)

    if not selected_files_paths:
        empty_fig = go.Figure(layout=dict(title="No files found in input directory", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'))
        return empty_fig, html.Div("No data to display"), {}

    # Create figure using new arguments
    fig, combined_df = create_figure(selected_files_paths, unassigned_points, manual_assignments)

    # --- Prepare combined data for assignment store ---
    combined_data_for_store = {}
    # Prepare combined data for assignment store
    if not combined_df.empty and 'point_id' in combined_df.columns:
        try:
            # Select only necessary columns and convert to dict {point_id: {col: val}}
            relevant_cols = ['point_id', 'File', X_COLUMN, Y_COLUMN]
            # Check if all columns exist before proceeding
            if all(col in combined_df.columns for col in relevant_cols):
                combined_data_for_store = combined_df[relevant_cols].set_index('point_id').to_dict('index')
            else:
                 print("Warning: Not all expected columns found in combined_df for assignment store.")
        except Exception as e:
            print(f"Error preparing data for assignment store: {e}")
            combined_data_for_store = {} # Ensure it's an empty dict on error

    # --- Apply previous zoom state ---
    assignment_change_triggered = triggered_id in ['unassigned-points-store', 'manual-assignments-log-store']
    # Apply zoom unless triggered by assignment changes (which shouldn't reset zoom)
    apply_zoom = last_zoom_state and isinstance(last_zoom_state, dict) # Apply if present

    if apply_zoom:
        if 'xaxis.range[0]' in last_zoom_state and 'xaxis.range[1]' in last_zoom_state:
            fig.update_layout(xaxis_range=[last_zoom_state['xaxis.range[0]'], last_zoom_state['xaxis.range[1]']])
        if 'yaxis.range[0]' in last_zoom_state and 'yaxis.range[1]' in last_zoom_state:
             # Apply y-axis constraint (>=0) and min span from stored state logic
             y_min = max(0, last_zoom_state['yaxis.range[0]'])
             y_max = last_zoom_state['yaxis.range[1]']
             min_y_span = 10
             if y_max > y_min and (y_max - y_min < min_y_span):
                 mid = (y_min + y_max) / 2
                 y_min_adj = max(0, mid - min_y_span / 2)
                 y_max_adj = y_min_adj + min_y_span
                 y_min = y_min_adj
                 y_max = y_max_adj
             fig.update_layout(yaxis_range=[y_min, y_max])

    # --- Create data table ---
    data_table_output = dash.no_update
    if active_tab == 'tab-data':
        if not combined_df.empty:
            table_data = combined_df.to_dict('records')
            columns = [
                {'name': 'File', 'id': 'File', 'type': 'text'},
                {'name': X_COLUMN, 'id': X_COLUMN, 'type': 'numeric', 'format': {'specifier': '.4f'}},
                {'name': Y_COLUMN, 'id': Y_COLUMN, 'type': 'numeric', 'format': {'specifier': '.2f'}},
            ]
            for col in combined_df.columns:
                if col not in ['File', X_COLUMN, Y_COLUMN, 'point_id']:
                    columns.append({'name': col, 'id': col})

            data_table_output = dash_table.DataTable(
                id='data-table',
                columns=columns,
                data=table_data,
                page_size=20,
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'padding': '8px', 'minWidth': '100px', 'maxWidth': '300px', 'whiteSpace': 'normal', 'height': 'auto'},
                style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                filter_action="native",
                sort_action="native",
                sort_mode="multi",
                export_format="csv",
            )
        else:
             data_table_output = html.Div("No data to display")
    elif triggered_id == 'update-button' and active_tab != 'tab-data': # Note: update-button no longer exists
        data_table_output = html.Div("Switch to Data tab to view tabular data")

    # Removed status message as there's no place to display it now
    return fig, data_table_output, combined_data_for_store

# --- Callback to Store Zoom/Pan State ---
@app.callback(
    Output('last-zoom-state', 'data'),
    Input('chromatogram-plot', 'relayoutData'),
    State('last-zoom-state', 'data'),
    prevent_initial_call=True
)
def handle_zoom_pan(relayout_data, current_zoom_state):
    if not relayout_data:
        return dash.no_update

    if relayout_data.get('xaxis.autorange') or relayout_data.get('yaxis.autorange'):
        return None # Reset stored zoom on autorange

    # Check if specific zoom/pan ranges occurred using full keys
    x_range_keys = ['xaxis.range[0]', 'xaxis.range[1]']
    y_range_keys = ['yaxis.range[0]', 'yaxis.range[1]']
    x_range_present = all(k in relayout_data for k in x_range_keys)
    y_range_present = all(k in relayout_data for k in y_range_keys)
    x_range_present_short = 'xaxis.range' in relayout_data and len(relayout_data['xaxis.range']) == 2
    y_range_present_short = 'yaxis.range' in relayout_data and len(relayout_data['yaxis.range']) == 2


    new_zoom_state = current_zoom_state.copy() if current_zoom_state else {}
    updated = False

    if x_range_present:
        new_zoom_state['xaxis.range[0]'] = relayout_data['xaxis.range[0]']
        new_zoom_state['xaxis.range[1]'] = relayout_data['xaxis.range[1]']
        updated = True
    elif x_range_present_short:
        new_zoom_state['xaxis.range[0]'] = relayout_data['xaxis.range'][0]
        new_zoom_state['xaxis.range[1]'] = relayout_data['xaxis.range'][1]
        updated = True


    if y_range_present:
        # Store raw y range, apply constraints in update_plot_and_data
        new_zoom_state['yaxis.range[0]'] = relayout_data['yaxis.range[0]']
        new_zoom_state['yaxis.range[1]'] = relayout_data['yaxis.range[1]']
        updated = True
    elif y_range_present_short:
        new_zoom_state['yaxis.range[0]'] = relayout_data['yaxis.range'][0]
        new_zoom_state['yaxis.range[1]'] = relayout_data['yaxis.range'][1]
        updated = True

    # Only return if we have a complete state or if something updated
    if updated and all(k in new_zoom_state for k in x_range_keys + y_range_keys):
        return new_zoom_state
    elif updated:
         # Store partial updates
         return new_zoom_state
    else:
         # print(f"No relevant zoom/pan keys in relayoutData: {relayout_data.keys()}")
         return dash.no_update


# --- MODIFIED Callback to export RAW data ---
@app.callback(
    Output('export-status', 'children'),
    Input('export-button', 'n_clicks'),
    # No state needed, find files directly
    prevent_initial_call=True
)
def export_raw_data(n_clicks): # Renamed function, removed selected_files state
    ctx = callback_context
    if not ctx.triggered:
        return ""
    all_csv_files = find_csv_files(INPUT_DIR)

    selected_files_paths = all_csv_files # Use all found files
    if not selected_files_paths:
        return html.Div("No files found for export", style={'color': 'red'})

    try:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Create a general 'exports' directory if it doesn't exist
        base_export_dir = Path("exports")
        base_export_dir.mkdir(exist_ok=True)
        export_dir = base_export_dir / f"raw_data_{timestamp}"
        export_dir.mkdir(exist_ok=True)

        exported_count = 0
        errors = []
        for file_path in selected_files_paths:
            try:
                df = pd.read_csv(file_path)
                # Maybe add point_id column before export for traceability?
                if 'point_id' not in df.columns:
                     df['point_id'] = [f"{file_path.name}_{i}" for i in range(len(df))]

                output_path = export_dir / f"raw_{file_path.name}"
                df.to_csv(output_path, index=False) # Export original df
                exported_count += 1
            except Exception as read_e:
                 error_msg = f"Error processing file {file_path.name} for raw export: {read_e}"
                 print(error_msg)
                 errors.append(error_msg)

        if exported_count > 0 and not errors:
            return html.Div(f"Successfully exported {exported_count} raw file(s) to '{export_dir}'", style={'color': 'green'})
        elif exported_count > 0 and errors:
             return html.Div([f"Exported {exported_count} file(s) with errors to '{export_dir}'. Errors:", html.Ul([html.Li(e) for e in errors])], style={'color': 'orange'})
        else:
             return html.Div(["Export failed. Could not process any files.", html.Ul([html.Li(e) for e in errors])], style={'color': 'red'})

    except Exception as e:
        return html.Div(f"Error creating export directory or during raw export: {str(e)}", style={'color': 'red'})

# --- REVISED Callback for auto-assignment AND table updates ---
@app.callback(
    [Output('aa-assignment-results', 'children'),
     Output('aa-assignment-results', 'style'),
     Output('amino-acid-assignments', 'data', allow_duplicate=True), # Output final combined assignments
     Output('std-rt-map-store', 'data'), # Output Std RT map
     Output('unassigned-points-store', 'data', allow_duplicate=True), # Output unassigned points {pid: True}
     Output('results-data-store', 'data', allow_duplicate=True)], # MODIFIED output store, clear results
    [Input('manual-assignments-log-store', 'data'), # TRIGGER on manual changes
     Input('chromatogram-plot', 'figure')], # TRIGGER after plot update
    # No State needed for files
    prevent_initial_call=True # Important to prevent running before figure exists
)
def auto_assign_amino_acids(manual_assignments_log, figure):
    # Use ctx to determine trigger
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    print(f"auto_assign_amino_acids triggered by: {triggered_id}") # Debug print

    # Clear results store whenever assignments are recalculated
    clear_results_store = None


    # Get all files every time
    all_csv_files = find_csv_files(INPUT_DIR)
    selected_files_paths = sorted(all_csv_files)
    manual_assignments_log = manual_assignments_log or {} # Get the latest log
    manually_assigned_pids = set(manual_assignments_log.keys())

    if not selected_files_paths:
        # Clear results if no files selected
        return html.Div("No files selected for assignment", style={'color': 'red'}), {'display': 'block'}, {}, {}, {}, clear_results_store

    try:
        data_by_file = {}
        std_files = []
        sample_files = []
        all_initial_sample_point_ids = set()

        for file_path in selected_files_paths:
            try:
                df = pd.read_csv(file_path)
                if X_COLUMN not in df.columns or Y_COLUMN not in df.columns:
                    continue
                df['point_id'] = [f"{file_path.name}_{i}" for i in range(len(df))]

                # Store the full data temporarily
                data_by_file[file_path.name] = df

                if 'std' in file_path.name.lower():
                    std_files.append(file_path.name)
                else:
                    sample_files.append(file_path.name)
                    # Store all sample point IDs initially
                    all_initial_sample_point_ids.update(df['point_id'].tolist())

            except Exception as e:
                print(f"Error processing file {file_path.name}: {e}")

        if not std_files:
             # Handle case with no standards - only manual assignments possible
             final_assignments_no_std = defaultdict(dict)
             for pid, manual_data in manual_assignments_log.items():
                 try:
                     filename = "_".join(pid.split('_')[:-1])
                     if filename in sample_files: # Only include if file is selected
                        aa = manual_data['aa']
                        final_assignments_no_std[filename][aa] = {
                            'rt': manual_data.get('rt'),
                            'area': manual_data.get('area'),
                            'distance': None,
                            'point_id': pid,
                            'type': 'manual'
                         }
                 except Exception as e:
                     print(f"Error merging manual assignment (no std) for {pid}: {e}")

             table_html_no_std = generate_assignment_table(final_assignments_no_std, manual_assignments_log, sample_files, {})
             # All non-manually assigned points are unassigned
             unassigned_no_std = {pid: True for pid in all_initial_sample_point_ids if pid not in manually_assigned_pids}
             return table_html_no_std, {'display': 'block', 'marginTop': '20px'}, final_assignments_no_std, {}, unassigned_no_std, clear_results_store


        if not data_by_file or not sample_files:
             # Clear results if no sample data
             return html.Div("No valid sample data found."), {'display': 'block'}, {}, {}, {}, clear_results_store

        # --- Calculate Standard RT Map ---
        std_rt_map = {}
        std_files_with_min_rt = []
        for std_file in std_files:
            if std_file in data_by_file:
                df = data_by_file[std_file] # Use the full standard data for RT map
                if not df.empty:
                    min_rt = df[X_COLUMN].min()
                    std_files_with_min_rt.append((std_file, min_rt))
        std_files_with_min_rt.sort(key=lambda x: x[1])
        sorted_std_files = [item[0] for item in std_files_with_min_rt]

        all_std_points = []
        for std_file in sorted_std_files:
            if std_file in data_by_file:
                df = data_by_file[std_file]
                # Filter out points that happen to have IDs matching manual assignments? Should not occur.
                df_std_filtered = df[~df['point_id'].isin(manually_assigned_pids)]
                std_points = df_std_filtered[[X_COLUMN, Y_COLUMN]].sort_values(by=X_COLUMN).values.tolist()
                all_std_points.extend(std_points)
        all_std_points.sort(key=lambda x: x[0])

        min_rt_diff = 0.1
        distinct_std_points = []
        for point in all_std_points:
            if not distinct_std_points or point[0] - distinct_std_points[-1][0] >= min_rt_diff:
                distinct_std_points.append(point)

        num_aa_labels = len(AMINO_ACID_LABELS)
        num_distinct_points = len(distinct_std_points)
        for idx in range(min(num_aa_labels, num_distinct_points)):
            rt_value = distinct_std_points[idx][0]
            aa_label = AMINO_ACID_LABELS[idx]
            std_rt_map[aa_label] = rt_value # Store as {AA: RT}

        # --- Perform Auto-Assignment ---
        auto_assignments_by_file = defaultdict(dict)
        assigned_point_ids_auto = set()
        std_aa_to_rt = std_rt_map # Use the map directly {AA: RT}

        for sample_file in sample_files:
            if sample_file not in data_by_file:
                continue
            sample_df = data_by_file[sample_file]
            # IMPORTANT: Filter out points already manually assigned BEFORE auto-assignment
            sample_df_filtered = sample_df[~sample_df['point_id'].isin(manually_assigned_pids)]
            if sample_df_filtered.empty:
                continue

            # Group points by AA they are closest to (within tolerance)
            possible_assignments = defaultdict(list)
            # Use the filtered df for auto-assignment
            for _, row in sample_df_filtered.iterrows():
                sample_rt = row[X_COLUMN]
                sample_area = row[Y_COLUMN]
                point_id = row['point_id']

                best_match_aa = None
                min_distance = float('inf')

                for aa_label, current_std_rt in std_aa_to_rt.items():
                    distance = abs(sample_rt - current_std_rt)
                    if distance <= 0.5:
                       if distance < min_distance:
                           min_distance = distance
                           best_match_aa = aa_label
                           # Store candidate info associated with this best match
                           candidate_info = {
                                'rt': sample_rt,
                                'area': sample_area,
                                'distance': distance,
                                'point_id': point_id
                           }
                           # Do not break here, find the absolute closest within tolerance

                # After checking all AAs, if a best match was found, add it
                if best_match_aa:
                     possible_assignments[best_match_aa].append(candidate_info)


            # Resolve conflicts: For each AA, if multiple points match, choose the one with highest Area.
            for aa_label, candidates in possible_assignments.items():
                if candidates:
                    best_candidate = max(candidates, key=lambda x: x['area']) # Max area wins
                    auto_assignments_by_file[sample_file][aa_label] = {
                         **best_candidate,
                         'type': 'auto' # Mark as auto
                     }
                    assigned_point_ids_auto.add(best_candidate['point_id'])

        # --- Determine Unassigned Points ---
        # Unassigned = (All sample points) - (Manually assigned) - (Auto assigned)
        unassigned_point_ids = all_initial_sample_point_ids - manually_assigned_pids - assigned_point_ids_auto
        unassigned_points_store_data = {pid: True for pid in unassigned_point_ids}

        # --- Prepare Final Assignments (Combine Auto + Manual) ---
        final_assignments = defaultdict(dict)
        for file, assignments in auto_assignments_by_file.items():
             for aa, data in assignments.items():
                  final_assignments[file][aa] = data.copy() # Copy auto assignments

        # Merge/overwrite with manual assignments
        for pid, manual_data in manual_assignments_log.items():
            try:
                filename_parts = pid.split('_')
                if len(filename_parts) < 2: continue # Skip malformed IDs
                filename = "_".join(filename_parts[:-1])
                aa = manual_data.get('aa')

                if not aa: continue # Skip if manual entry has no AA

                if filename in sample_files: # Only process if file is currently selected
                     # Check if this AA was auto-assigned to a *different* point and remove the auto one
                     # If manual assigns the *same* AA as auto, the manual assignment overwrites below
                     if aa in final_assignments[filename] and \
                        final_assignments[filename][aa]['type'] == 'auto' and \
                        final_assignments[filename][aa]['point_id'] != pid:
                         old_auto_pid = final_assignments[filename][aa]['point_id']
                         print(f"Manual assignment {pid} ({aa}) overrides different auto-assigned point {old_auto_pid} for {aa} in {filename}")
                         del final_assignments[filename][aa] # Remove the old auto-assignment for this AA
                         # Need to mark the old auto-assigned point as unassigned now
                         if old_auto_pid not in manually_assigned_pids: # Check it wasn't manually assigned elsewhere
                            unassigned_points_store_data[old_auto_pid] = True

                     # Always add/overwrite with the manual assignment for this AA
                     final_assignments[filename][aa] = {
                         'rt': manual_data.get('rt'),
                         'area': manual_data.get('area'),
                         'distance': None, # Manual assignments don't have distance from std
                         'point_id': pid,
                         'type': 'manual'
                     }

            except Exception as e:
                print(f"Error merging manual assignment for {pid}: {e}")

        # --- Generate Table ---
        # Generate table using the *final* combined assignments and the *original* manual log
        table_html = generate_assignment_table(final_assignments, manual_assignments_log, sample_files, std_rt_map)

        return table_html, {'display': 'block', 'marginTop': '20px'}, final_assignments, std_rt_map, unassigned_points_store_data, clear_results_store

    except Exception as e:
        import traceback
        print(f"Error in auto assignment/table update: {str(e)}")
        print(traceback.format_exc())
        # Return error message, keep display block, no updates to data stores, clear results store
        return html.Div(f"Error updating assignments: {str(e)}"), {'display': 'block'}, dash.no_update, dash.no_update, dash.no_update, clear_results_store

# --- REVISED Callback to export assignments TABLE (CSV) ---
@app.callback(
    Output('export-aa-status', 'children'),
    Input('export-aa-assignments', 'n_clicks'),
    [State('amino-acid-assignments', 'data'), # Combined final assignments
     State('std-rt-map-store', 'data')], # Need std map
     # Removed State('stored-selected-files', 'data')
    prevent_initial_call=True
)
def export_aa_assignments_csv(n_clicks, final_assignments, std_rt_map):
    if not final_assignments:
        return html.Div("No assignment data available to export", style={'color': 'red'})
    # Get all files directly
    all_csv_files = find_csv_files(INPUT_DIR)
    final_assignments = final_assignments or {}
    std_rt_map = std_rt_map or {}
    selected_files_paths = all_csv_files # Use all files
    sample_files = sorted([f.name for f in selected_files_paths if 'std' not in f.name.lower()])

    try:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Use general exports directory
        base_export_dir = Path("exports")
        base_export_dir.mkdir(exist_ok=True)
        export_dir = base_export_dir / "aa_assignments_csv"
        export_dir.mkdir(exist_ok=True)

        # --- Reconstruct DataFrame from final_assignments ---
        aa_df = pd.DataFrame(index=AMINO_ACID_LABELS)

        # Add Standard RT column
        std_rt_column = [f"{std_rt_map.get(aa, '--'):.3f}" if std_rt_map.get(aa) is not None else '--' for aa in AMINO_ACID_LABELS]
        aa_df['Standard RT'] = std_rt_column

        for sample_file in sample_files:
            rt_column_export = []
            file_assignments = final_assignments.get(sample_file, {})

            for aa in AMINO_ACID_LABELS:
                assignment_data = file_assignments.get(aa)
                if assignment_data:
                    rt_val = assignment_data.get('rt')
                    assign_type = assignment_data.get('type')
                    if rt_val is not None:
                        rt_str = f"{rt_val:.3f}"
                        if assign_type == 'manual':
                            rt_str += " (M)" # Indicate manual
                    else:
                        rt_str = "(M - No RT)" if assign_type == 'manual' else "(Auto - No RT)"
                    rt_column_export.append(rt_str)
                else:
                    rt_column_export.append("") # Use empty string for missing

            aa_df[f"{sample_file} (RT)"] = rt_column_export

        # Reset index
        aa_df.reset_index(inplace=True)
        aa_df.rename(columns={'index': 'Amino Acid'}, inplace=True)

        # Select and order columns
        export_columns = ['Amino Acid', 'Standard RT'] + [f"{sf} (RT)" for sf in sample_files]
        aa_df_export = aa_df[export_columns]

        # Save
        output_path = export_dir / f"aa_assignments_{timestamp}.csv"
        aa_df_export.to_csv(output_path, index=False)

        return html.Div(f"Successfully exported assignments table to '{output_path}'", style={'color': 'green'})

    except Exception as e:
        import traceback
        print(f"Error exporting assignments CSV: {str(e)}")
        print(traceback.format_exc())
        return html.Div(f"Error exporting assignments CSV: {str(e)}", style={'color': 'red'})

# --- NEW Callback to handle plot clicks for manual assignment ---
@app.callback(
    [Output('manual-assign-controls', 'style'),
     Output('manual-assign-prompt', 'children'),
     Output('manual-assign-point-id-store', 'data', allow_duplicate=True),
     Output('manual-assign-dropdown', 'value', allow_duplicate=True)], # Reset dropdown
    [Input('chromatogram-plot', 'clickData')],
    [State('combined-data-for-assignment-store', 'data'),
     State('manual-assignments-log-store', 'data')],
    prevent_initial_call=True
)
def handle_plot_click(clickData, combined_data, manual_log):
    if not clickData or not clickData.get('points'):
        # Hide controls if click is outside points
        return {'display': 'none'}, dash.no_update, None, None

    point_data = clickData['points'][0]
    point_id = point_data.get('customdata')
    combined_data = combined_data or {}
    manual_log = manual_log or {}

    if not point_id or point_id not in combined_data:
        print(f"Clicked point ID {point_id} not found in combined data.")
        # Hide controls if point ID invalid
        return {'display': 'none'}, dash.no_update, None, None

    point_info = combined_data[point_id]
    filename = point_info.get('File')
    rt = point_info.get(X_COLUMN) # Keep as number
    area = point_info.get(Y_COLUMN) # Keep as number

    # Only allow assignment for non-standard files
    if filename and 'std' in filename.lower():
        print(f"Clicked on a standard file point ({point_id}), skipping assignment.")
        # Hide controls if std file point
        return {'display': 'none'}, dash.no_update, None, None

    # Check if already manually assigned to pre-select dropdown
    current_assignment_data = manual_log.get(point_id)
    current_aa = current_assignment_data.get('aa') if current_assignment_data else None

    rt_display = f"{rt:.3f}" if rt is not None else 'N/A'
    area_display = f"{area:,.1f}" if area is not None else 'N/A' # Add comma for thousands

    prompt_text = f"Assign Point: ID={point_id} | File={filename} | RT={rt_display} | Area={area_display}"
    # Show controls, set prompt, store pid, pre-select dropdown (or None if not assigned)
    visible_style = {'display': 'block', 'padding': '15px', 'border': '1px solid #eee', 'borderRadius': '5px', 'marginTop': '10px'}
    return visible_style, prompt_text, point_id, current_aa

# --- REVISED Callback to handle manual assignment confirmation ---
@app.callback(
    [Output('manual-assign-controls', 'style', allow_duplicate=True), # Hide controls
     Output('manual-assign-point-id-store', 'data', allow_duplicate=True), # Clear stored pid
     Output('manual-assignments-log-store', 'data', allow_duplicate=True), # Update log
     Output('unassigned-points-store', 'data', allow_duplicate=True), # Update unassigned
     Output('manual-assign-dropdown', 'value', allow_duplicate=True), # Reset dropdown
     Output('manual-assign-validation-message', 'children')], # ADDED validation message output
    [Input('manual-assign-button', 'n_clicks'),
     Input('manual-assign-cancel-button', 'n_clicks')],
    [State('manual-assign-point-id-store', 'data'),
     State('manual-assign-dropdown', 'value'),
     State('manual-assignments-log-store', 'data'),
     State('unassigned-points-store', 'data'),
     State('combined-data-for-assignment-store', 'data'),
     State('amino-acid-assignments', 'data')], # ADDED state for final assignments check
    prevent_initial_call=True
)
def handle_manual_assignment(assign_click, cancel_click, point_id, selected_aa,
                           manual_log, unassigned_points, combined_data,
                           final_assignments): # ADDED final_assignments
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'No ID'

    hidden_style = {'display': 'none'}
    visible_style = {'display': 'block', 'padding': '15px', 'border': '1px solid #eee', 'borderRadius': '5px', 'marginTop': '10px'} # Reuse from handle_plot_click

    manual_log = manual_log or {}
    unassigned_points = unassigned_points or {}
    combined_data = combined_data or {}
    final_assignments = final_assignments or {} # Initialize if None

    # Default outputs
    output_style = hidden_style
    output_point_id = None
    output_dropdown_value = None
    output_manual_log = dash.no_update
    output_unassigned = dash.no_update
    output_validation_message = "" # Clear message by default

    if triggered_id == 'manual-assign-cancel-button' or not point_id:
        # Hide, clear pid, no log change, no unassigned change, clear dropdown, clear validation
        return output_style, output_point_id, output_manual_log, output_unassigned, output_dropdown_value, output_validation_message

    if triggered_id == 'manual-assign-button':
        if not selected_aa:
            # Keep controls open, keep point_id, keep dropdown value, don't update logs
            print("No AA selected for assignment.")
            # Keep controls open, show validation message
            return (visible_style, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
                    "Please select an Amino Acid or 'Unassign Point'.")

        point_info = combined_data.get(point_id)
        filename = point_info.get('File')
        rt = point_info.get(X_COLUMN)
        area = point_info.get(Y_COLUMN)

        # --- Validation Check: Prevent assigning AA if already assigned to DIFFERENT point in same file ---
        if selected_aa != "UNASSIGN" and filename and final_assignments:
            file_assignments = final_assignments.get(filename, {})
            if selected_aa in file_assignments:
                existing_assignment_data = file_assignments[selected_aa]
                existing_pid = existing_assignment_data.get('point_id')
                # Check if the existing assignment is for a DIFFERENT point
                if existing_pid and existing_pid != point_id:
                    error_message = f"Error: {selected_aa} already assigned to point {existing_pid} in {filename}. Unassign it first or cancel."
                    # Keep controls open, keep selections, show error
                    return (visible_style, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
                            error_message)
        # --- End Validation Check ---

        # Create copies to modify
        new_manual_log = manual_log.copy()
        new_unassigned_points = unassigned_points.copy()

        if selected_aa == "UNASSIGN": # Check for the special unassign value
            # Remove from manual log if exists
            if point_id in new_manual_log:
                print(f"Unassigning point {point_id} (was {new_manual_log[point_id].get('aa')})")
                del new_manual_log[point_id]
            # Add to unassigned points (will be handled by auto-assign callback trigger)
            # Explicitly adding here might be redundant but ensures consistency if callback logic changes
            if point_id not in new_unassigned_points:
                 new_unassigned_points[point_id] = True
            output_manual_log = new_manual_log
            output_unassigned = new_unassigned_points # Pass the potentially updated dict
            print(f"Point {point_id} marked as unassigned.")
        else:
            # Add/update manual log
            new_manual_log[point_id] = {'aa': selected_aa, 'rt': rt, 'area': area}
            # Remove from unassigned points if exists
            if point_id in new_unassigned_points:
                del new_unassigned_points[point_id]
            output_manual_log = new_manual_log
            output_unassigned = new_unassigned_points # Pass the potentially updated dict
            print(f"Manually assigned {selected_aa} to point {point_id}")

        # Hide, clear pid, update log, update unassigned, clear dropdown, clear validation
        return output_style, output_point_id, output_manual_log, output_unassigned, output_dropdown_value, output_validation_message

    # Should not happen unless callback triggered unexpectedly
    return (dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
            "An unexpected error occurred.")


# --- NEW Callback to Calculate Normalized Area & Concentration ---
@app.callback(
    Output('results-data-store', 'data', allow_duplicate=True), # UPDATED store name
    Input('confirm-assignments-button', 'n_clicks'),
    [State('amino-acid-assignments', 'data')],
    # Removed State('stored-selected-files', 'data')
    prevent_initial_call=True
)
def calculate_results(n_clicks, final_assignments): # Renamed function, removed selected_files state
    if n_clicks == 0 or not final_assignments:
        print("Calculation skipped: No confirm click or no assignments.")
        return None # No calculation needed yet or no assignments done

    final_assignments = final_assignments or {}
    # Get all files
    all_csv_files = find_csv_files(INPUT_DIR)
    selected_files_paths = all_csv_files # Use all files
    sample_files = sorted([f.name for f in selected_files_paths if 'std' not in f.name.lower()])

    if not sample_files:
        print("No sample files found for calculation.")
        return None

    # --- Load Calibration Data ---
    calibration_data = {}
    calibration_errors = []
    if CALIBRATION_FILE.exists():
        try:
            df_cal = pd.read_csv(CALIBRATION_FILE)
            # Ensure required columns exist using ACTUAL names from CSV
            required_cols = ['AminoAcid', 'Slope', 'y-intercept']
            if not all(col in df_cal.columns for col in required_cols):
                 # Update error message to show the names it looked for
                 calibration_errors.append(f"Calibration file '{CALIBRATION_FILE}' missing required columns ({required_cols}). Found: {list(df_cal.columns)}")
            else:
                # Rename 'y-intercept' to 'Intercept' and 'AminoAcid' to 'Amino Acid'
                # to match expected dict keys later
                df_cal.rename(columns={'y-intercept': 'Intercept', 'AminoAcid': 'Amino Acid'}, inplace=True)

                # Convert to dictionary {AA: {'Slope': float, 'Intercept': float}}
                # Use the RENAMED columns for indexing and selection
                calibration_data = df_cal.set_index('Amino Acid')[['Slope', 'Intercept']].apply(pd.to_numeric, errors='coerce').to_dict('index')
                # Check for NaN values after conversion
                for aa, values in calibration_data.items():
                    if pd.isna(values['Slope']) or pd.isna(values['Intercept']):
                        calibration_errors.append(f"Invalid non-numeric Slope or Intercept for '{aa}' in calibration file.")
                        # Remove invalid entry? Or handle downstream? Let's keep it for now and handle downstream.

        except Exception as e:
            calibration_errors.append(f"Error reading calibration file '{CALIBRATION_FILE}': {e}")
    else:
        calibration_errors.append(f"Calibration file '{CALIBRATION_FILE}' not found.")

    if calibration_errors:
        print("Calibration data issues:", calibration_errors)
        # Proceed with normalization but concentration will be NaN/missing

    # --- Calculate Normalized Area and Concentration ---
    results_list = []
    norm_errors = []

    for sample_file in sample_files:
        file_assignments = final_assignments.get(sample_file, {})
        is_assignment = file_assignments.get('IS') # Find Internal Standard

        if not is_assignment or is_assignment.get('area') is None or is_assignment.get('area') <= 0:
            msg = f"Missing/invalid IS area for {sample_file}. Cannot normalize/calculate concentration for this file."
            print(f"Warning: {msg}")
            norm_errors.append(msg)
            # Add entries with NaN for this sample? Or skip? Let's skip for now.
            continue # Skip this file

        is_area = is_assignment['area']

        for aa_label, assignment_data in file_assignments.items():
            # Skip IS itself for normalization/concentration
            if aa_label == 'IS':
                continue

            aa_area = assignment_data.get('area')
            normalized_area = None
            concentration = None

            if aa_area is not None:
                normalized_area = aa_area / is_area

                # Calculate concentration if calibration data is valid for this AA
                if aa_label in calibration_data:
                    cal_params = calibration_data[aa_label]
                    slope = cal_params.get('Slope')
                    # Use the renamed 'Intercept' key here
                    intercept = cal_params.get('Intercept')
                    # Ensure slope and intercept are valid numbers
                    if pd.notna(slope) and pd.notna(intercept):
                        concentration = (normalized_area * slope) + intercept
                    else:
                        if f"Invalid calibration for {aa_label}" not in norm_errors: # Avoid duplicate messages
                            norm_errors.append(f"Invalid calibration for {aa_label}")
                # else: # AA not in calibration file
                    # if f"Missing calibration for {aa_label}" not in norm_errors:
                    #     norm_errors.append(f"Missing calibration for {aa_label}")


                results_list.append({
                    'Sample': sample_file,
                    'Amino Acid': aa_label,
                    'Normalized Area': normalized_area,
                    'Concentration (mg/L)': concentration # Will be None if calculation failed
                })
            # else: # Handle case where assigned AA has no area? Should not happen based on current logic
            #     print(f"Warning: Assigned AA '{aa_label}' in {sample_file} has no area data.")


    if norm_errors:
         # Log errors but still return potentially partial results
         print(f"Calculation completed with issues: {', '.join(norm_errors)}")

    if not results_list:
        print("No results could be calculated.")
        return None

    # Return list of dicts for the store
    print(f"Calculated {len(results_list)} results entries.")
    return results_list

# --- NEW Callback to Display Results Tables (Normalized Area & Concentration) ---
@app.callback(
    Output('results-tables-container', 'children'),
    Input('results-data-store', 'data'),
    prevent_initial_call=True
)
def display_results_tables(results_data):
    if not results_data:
        return html.Div(style={'display': 'none'}) # Hide if no data

    try:
        # Convert list of dicts to DataFrame
        df_results = pd.DataFrame(results_data)

        if df_results.empty:
             return html.Div("No results data to display.")

        # --- Create Normalized Area Table ---
        df_pivot_norm = pd.pivot_table(df_results,
                                       values='Normalized Area',
                                       index='Sample',
                                       columns='Amino Acid')
        # Ensure correct AA order, including potentially missing ones
        # Filter AMINO_ACID_LABELS to exclude 'IS' for these tables
        display_aa_labels = [aa for aa in AMINO_ACID_LABELS if aa != 'IS']
        df_pivot_norm = df_pivot_norm.reindex(columns=display_aa_labels)
        df_pivot_norm.reset_index(inplace=True)

        norm_table_cols = [{'name': 'Sample', 'id': 'Sample'}]
        # Use .4f format for normalized area
        norm_table_cols.extend([{'name': aa, 'id': aa, 'type': 'numeric', 'format': {'specifier': '.4f'}} for aa in display_aa_labels])

        # Separate Sample column before formatting
        samples_norm = df_pivot_norm['Sample']
        numeric_data_norm = df_pivot_norm[display_aa_labels]

        # Apply formatting only to numeric columns using .map (replaces deprecated .applymap)
        formatted_numeric_norm = numeric_data_norm.map(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) and pd.notna(x) else '--')

        # Combine Sample column with formatted numeric data
        formatted_data_norm = pd.concat([samples_norm, formatted_numeric_norm], axis=1)

        norm_table = dash_table.DataTable(
             id='normalized-area-table',
             columns=norm_table_cols,
             # Use the correctly formatted combined data
             data=formatted_data_norm.to_dict('records'),
             style_table={'overflowX': 'auto', 'marginTop': '15px'},
             style_cell={'textAlign': 'center', 'padding': '8px', 'width': 'auto', 'whiteSpace': 'normal', 'height': 'auto'},
             style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
             style_data_conditional=[
                 {'if': {'column_id': 'Sample'}, 'textAlign': 'left', 'fontWeight': 'bold', 'backgroundColor': 'rgb(240, 240, 240)'},
             ],
             filter_action="native",
             sort_action="native",
             sort_mode="multi",
             page_size=25,
             export_format="none",
         )

        # --- Create Concentration Table ---
        concentration_present = 'Concentration (mg/L)' in df_results.columns and df_results['Concentration (mg/L)'].notna().any()
        conc_table_div = html.Div()

        if concentration_present:
            df_pivot_conc = pd.pivot_table(df_results,
                                          values='Concentration (mg/L)',
                                          index='Sample',
                                          columns='Amino Acid')
            df_pivot_conc = df_pivot_conc.reindex(columns=display_aa_labels)
            df_pivot_conc.reset_index(inplace=True)

            conc_table_cols = [{'name': 'Sample', 'id': 'Sample'}]
            # Use .3f format for concentration
            conc_table_cols.extend([{'name': aa, 'id': aa, 'type': 'numeric', 'format': {'specifier': '.3f'}} for aa in display_aa_labels])

            # Separate Sample column before formatting
            samples_conc = df_pivot_conc['Sample']
            numeric_data_conc = df_pivot_conc[display_aa_labels]

            # Apply formatting only to numeric columns using .map
            formatted_numeric_conc = numeric_data_conc.map(lambda x: f"{x:.3f}" if isinstance(x, (int, float)) and pd.notna(x) else '--')

            # Combine Sample column with formatted numeric data
            formatted_data_conc = pd.concat([samples_conc, formatted_numeric_conc], axis=1)

            conc_table = dash_table.DataTable(
                 id='concentration-table',
                 columns=conc_table_cols,
                 # Use the correctly formatted combined data
                 data=formatted_data_conc.to_dict('records'), # Format NaNs implicitly handled by map
                 style_table={'overflowX': 'auto', 'marginTop': '15px'},
                 style_cell={'textAlign': 'center', 'padding': '8px', 'width': 'auto', 'whiteSpace': 'normal', 'height': 'auto'},
                 style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                 style_data_conditional=[
                     {'if': {'column_id': 'Sample'}, 'textAlign': 'left', 'fontWeight': 'bold', 'backgroundColor': 'rgb(240, 240, 240)'},
                 ],
                 filter_action="native",
                 sort_action="native",
                 sort_mode="multi",
                 page_size=25,
                 export_format="none",
             )
            conc_table_div = html.Div([
                html.H3("Concentration Results (mg/L)", style={'marginTop': '30px'}),
                conc_table
            ])

        # --- Combine Components ---
        return html.Div([
            html.H3("Normalized Area Results (relative to IS)", style={'marginTop': '30px'}),
            norm_table,
            conc_table_div,
        ])

    except Exception as e:
        import traceback
        print(f"Error generating results tables: {str(e)}")
        print(traceback.format_exc())
        # Return error message in the UI
        return html.Div(f"Error displaying results tables: {str(e)}", style={'color': 'red', 'marginTop': '20px'})

# Run the app
if __name__ == '__main__':
    # Ensure export directories exist on startup
    Path("exports/raw_data").mkdir(parents=True, exist_ok=True)
    Path("exports/aa_assignments_csv").mkdir(parents=True, exist_ok=True)
    Path("exports/results_excel").mkdir(parents=True, exist_ok=True) # Keep for now in case needed later, or remove fully

    # Check for calibration file
    if not CALIBRATION_FILE.exists():
        print(f"--- WARNING ---")
        print(f"Calibration file '{CALIBRATION_FILE}' not found in the current directory.")
        print(f"Concentration calculation (mg/L) will not be performed.")
        print(f"Please create a '{CALIBRATION_FILE}' with columns: Amino Acid, Slope, Intercept") # Note: Corrected column name
        print(f"---------------")

    app.run(debug=True, port=8001) 