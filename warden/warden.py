# %%
from pathlib import Path
import numpy as np
import pandas as pd
from dash import Dash, html, dcc, clientside_callback, Output, Input, Patch, ctx, ALL, State, register_page, no_update
from dash.exceptions import PreventUpdate
import dash_mantine_components as dmc
import dash_bootstrap_components as dbc
import plotly.express as px
from pandas.tseries.offsets import *
import plotly.graph_objects as go
from datetime import datetime, timedelta, date
import time
from urllib.parse import urlparse, parse_qs
from dash_extensions.enrich import (
    DashProxy,
    Input,
    Output,
    Serverside,
    ServersideOutputTransform,
    State,
    dcc,
    html,
    callback
)
from flask_caching import Cache
import re
from . common import loadConfig, readPickle
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)-40s %(message)s', datefmt="%Y-%m-%d %H:%M:%S")


class LoggingTimer:
    def __init__(self, name):
        self.name = name
        self.elapsed = 0.

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, *args):
        end = time.time()
        self.elapsed += end-self.start

    def __del__(self):
        logger.info(self.name+" " + str(self.elapsed))


config = loadConfig("config.yaml")
path_to_pickles = config["baseDir"] / config["dataLocation"]
possible_datasets = []
for repo in config["repositories"]:
    for dataset in repo["datasets"]:
        dataset_name = dataset["name"]
        pickleFile = path_to_pickles/f'{dataset_name}.hkl'
        if pickleFile.exists():
            possible_datasets.append(dataset_name)
        else:
            logger.warn(f"Pickle file {pickleFile} does not exist")
possible_datasets.sort()

annotationsFile = config["baseDir"]/config["annotations"]["filename"]

#################################################################################
#   ____            _        _                _                            _    #   
#  |  _ \  __ _ ___| |__    / \   _ __  _ __ | |    __ _ _   _  ___  _   _| |_  #
#  | | | |/ _` / __| '_ \  / _ \ | '_ \| '_ \| |   / _` | | | |/ _ \| | | | __| #
#  | |_| | (_| \__ \ | | |/ ___ \| |_) | |_) | |__| (_| | |_| | (_) | |_| | |_  #
#  |____/ \__,_|___/_| |_/_/   \_\ .__/| .__/|_____\__,_|\__, |\___/ \__,_|\__| #
#                                |_|   |_|               |___/                  #
#################################################################################

# %%

app = DashProxy(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME], transforms=[ServersideOutputTransform()])

# caching
cache = Cache(app.server, config={
    'CACHE_TYPE': 'SimpleCache'
})
readPickle = cache.memoize(timeout=60*20)(readPickle)

app.layout = dmc.MantineProvider(
        dbc.Container(
        [
            html.Span(
                [
                    dbc.Label(className='fa fa-moon', html_for='switch'),
                    dbc.Switch(id='switch', value=False, className='d-inline-block ms-1', persistence=True),
                    dbc.Label(className='fa fa-sun', html_for='switch'),
                ]
            ),
            html.Br(),

            #These are components that store certain data. They are not visually seen but hold the data for callbacks to use.
            dcc.Store(id='current-stored-depth', data=0),
            dcc.Store(id='previous-path', data= ''),
            dcc.Loading(dcc.Store(id='data-store'), fullscreen=True, type='dot'),
            dcc.Loading(dcc.Store(id='second-data-store'), fullscreen=True, type='dot'),
            dcc.Location(id='url', refresh=False),
            dcc.Store(id='annotation-date-store'),

            #Pops up when a point on the graph is clicked
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("Annotation at [DATE]",id="modal-title"), close_button=True),
                    dbc.ModalBody(
                        [
                            dbc.Input(id="annotation-input", placeholder="Type something...", type="text"),
                            html.P(id="gitsha-modal", children="GitSHA: [GITSHA]"),
                            html.P(id="note-modal", children="Annotation: Null")
                        ]
                    ),
                    dbc.ModalFooter(
                        [
                            # Item on the left
                            html.A(
                                id="commit-anchor",
                                children=f"Go To {config["code"]["name"]} commits",
                                href=f"{config["code"]["url"]}/compare",
                                target='_blank',
                            ),

                            # This div and its contents will be pushed to the right
                            html.Div(
                                [
                                    dbc.Button(
                                        "Delete",
                                        id="annotation-delete",
                                        color="danger",
                                        outline=True,
                                        n_clicks=0,
                                        className="me-2",  # Add a margin to the right of this button
                                    ),
                                    dbc.Button(
                                        "Submit",
                                        id="annotation-submit",
                                        n_clicks=0,
                                    )
                                ],
                            )
                        ],
                        className="d-flex justify-content-between", # This is the key change
                    ),
                ],
                id="annotation-modal",
                centered=True,
                is_open=False,
            ),

            html.H1('WARDEN Performance Dashboard'),

            # Inputs
            html.Div([
                html.H4('Which dataset?'),
                dbc.InputGroup(
                    [
                        dcc.Dropdown(
                            options=possible_datasets,
                            value=possible_datasets[0],
                            id='dataset-selection',
                            clearable=False,
                            style={'width': 550, 'color': 'black'},
                        )
                    ],
                    className='mx-5 mb-5',
                ),
                html.H4('Compared against which other dataset'),
                dbc.InputGroup(
                    [
                        dcc.Dropdown(
                            options=["None"]+possible_datasets,
                            value="None",
                            id='second-dataset-selection',
                            clearable=False,
                            style={'width': 550, 'color': 'black'},
                        )
                    ],
                    className='mx-5 mb-5',
                ),
                html.H4('How many days?'),
                dcc.Loading(dcc.Slider(7, 120,
                                       id='days-slider',
                                       step=None,
                                       marks={
                                           7: '7 days',
                                           30: '30 days',
                                           60: '60 days',
                                           90: '90 days',
                                           120:'120 days'
                                       },
                                       value=30,
                                       persistence=True,
                                       ), type='dot'),
                dbc.Button(["Display ", html.I(className='fa fa-rotate')], id='do-display-btn', className='px-2', style={'marginTop': '2em'}, size="lg"),
            ], style={'marginBottom': '3em'}),

            html.Hr(),

            # Navigation
            html.Div([
                dbc.Row([
                    dbc.Col([
                        dbc.Button([html.I(className='fa fa-angles-up')], id='top-level-btn', className='px-2'),
                        dbc.Button([html.I(className='fa fa-angle-up')], id='previous-level-btn', className='px-2'),
                    ], width=1),
                    dbc.Col([
                        html.H5('Displayed dataset: ', id='current-dataset'),
                        html.H5('Displayed subtest: ', id='current-subtest'),
                    ]),
                ])
            ]),

            # Graphs
            dcc.Loading(html.Div(id='graph-container', children=[], className='mt-4 mx-5'), fullscreen=True, type='dot'),
        ],
        className='mw-70',
    )
)



################################################
#    ____      _ _ _                _          #
#   / ___|__ _| | | |__   __ _  ___| | _____   #
#  | |   / _` | | | '_ \ / _` |/ __| |/ / __|  #
#  | |__| (_| | | | |_) | (_| | (__|   <\__ \  #
#   \____\__,_|_|_|_.__/ \__,_|\___|_|\_\___/  #
#                                              #
# ##############################################                                          


##################################################
# Data and graph callbacks

@app.callback(
    Output(component_id='dataset-selection', component_property='value'),
    Output(component_id='second-dataset-selection', component_property='value'),
    Output(component_id='do-display-btn', component_property='n_clicks'),
    Input('url', 'search')
)
def update_dropdown_from_url(search):
    if not search:
        return no_update
    parsed_queries = parse_qs(urlparse(search).query)
    dataset = parsed_queries.get('dataset', [None])[0]
    if dataset is None or dataset not in possible_datasets:
        return no_update
    second_dataset = parsed_queries.get('second_dataset', ["None"])[0]
    if second_dataset != "None" and second_dataset not in possible_datasets:
        return no_update
    return (dataset, second_dataset, 0)


@callback(
    Output(component_id='data-store', component_property='data'),
    Output(component_id='second-data-store', component_property='data'),
    Output(component_id='current-stored-depth', component_property='data', allow_duplicate=True),
    Output(component_id='previous-path', component_property='data', allow_duplicate=True),
    
    State(component_id='dataset-selection', component_property='value'),
    State(component_id='second-dataset-selection', component_property='value'),
    State(component_id='days-slider', component_property='value'),
    Input(component_id='do-display-btn', component_property='n_clicks'),
    Input(component_id='url', component_property='pathname'),
)
def select_dataset(val, val2, days, _, __):
    with LoggingTimer(f"TIMER FOR select_dataset() for {val}:"):
        pickleFile = path_to_pickles/f'{val}.hkl'
        df = readPickle(pickleFile)
        df = df[df['date'].values >= (df['date'].max() - timedelta(days=days))]
    if val2 != "None":
        with LoggingTimer(f"TIMER FOR select_dataset() for {val2}:"):
            pickleFile = path_to_pickles/f'{val2}.hkl'
            df2 = readPickle(pickleFile)
            df2 = df2[df2['date'].values >= (df2['date'].max() - timedelta(days=days))]
        return (Serverside(df), Serverside(df2), 0, '')
    else:
        return (Serverside(df), Serverside(None), 0, '')


@callback(
    Output(component_id={'type': 'stopped-reporting-text', 'index': ALL}, component_property='children'),
    Input(component_id={'type': 'generated-graph', 'index': ALL}, component_property='figure'),
    State(component_id='data-store', component_property='data')
)
def note_stopped_reporting(figs, df):
    missingDataWarningAfterDays = config["missingDataWarningAfterDays"]
    text_list = []
    df['date'] = pd.to_datetime(df['date'])
    for f in figs:
        f = go.Figure(f)
        dff = df[df['readable_path'] == f['data'][0]['meta']['path']]
        xmax = dff['date'].max()
        if xmax < pd.Timestamp(date.today()) - timedelta(days=missingDataWarningAfterDays):
            f.update_layout(
                paper_bgcolor='lawngreen'
            )
            text_list.append(html.P(f'***Stopped reporting >{missingDataWarningAfterDays} days ago***', className='text-danger'))
        else:
            text_list.append('')
    return text_list


@callback(
    # Output(component_id='dummy-holder', component_property='children'),
    Output(component_id='graph-container', component_property='children', allow_duplicate=True),

    Input(component_id='current-stored-depth', component_property='data'),
    Input(component_id='previous-path', component_property='data'),
    Input(component_id='data-store', component_property='data'),
    Input(component_id='second-data-store', component_property='data'),
    State(component_id='dataset-selection', component_property='value'),
    State(component_id='second-dataset-selection', component_property='value'),
    prevent_initial_call='initial_duplicate'
)
def graph_creator(curr_depth, previous_path, df, second_df, chosen_dset, second_chosen_dset):
    '''
    This graph deals with actually making the graphs.
    It does the following things:
        1. extracts only the graphs that have the correct number of slashes (aka current graph depth) and that contains the previous path (name of the graph we just dived in to)
        2. for each of the rows we just extracted, 
            - determine whether or not it can be dived in to, 
            - truncate it's title(for ease of reading), 
            - generate graph
        3. return 
    '''

    have_second_dataset = (second_chosen_dset != "None")

    with LoggingTimer("TIMER FOR graph_creator():"):
        with LoggingTimer("TIMER FOR graph_creator()::load_annotations"):
            assert annotationsFile.exists()

            def truncate_text(text, max_length=30):
                """Truncates a string and adds '...' if it's too long."""
                if len(text) > max_length:
                    return text[:max_length - 3] + '...'
                return text

            mach_annot_df = pd.read_csv(annotationsFile)
            mach_annot_df['date_only'] = pd.to_datetime(mach_annot_df['date_only'])
            mach_annot_df['truncated_note'] = mach_annot_df['note'].apply(truncate_text)

        with LoggingTimer("TIMER FOR graph_creator()::process_data"):
            dff = df[df['number_of_slashes'] == curr_depth]
            dff = dff[dff['readable_path'].str.startswith(previous_path)]

            mach_annot_df_dataset = mach_annot_df[mach_annot_df['dataset_name'] == chosen_dset]
            dff = pd.merge(dff, mach_annot_df_dataset[['date_only', 'note', 'truncated_note']], on='date_only', how='left')

            if have_second_dataset:
                second_dff = second_df[second_df['number_of_slashes'] == curr_depth]
                second_dff = second_dff[second_dff['readable_path'].str.startswith(previous_path)]

                mach_annot_df_second_dataset = mach_annot_df[mach_annot_df['dataset_name'] == second_chosen_dset]
                second_dff = pd.merge(second_dff, mach_annot_df_second_dataset[['date_only', 'note', 'truncated_note']], on='date_only', how='left')


            def natural_sorting(s):
                return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

            unique_paths = sorted(dff['readable_path'].unique(),
                                  key=lambda x: natural_sorting(x))

        counter = 0
        truncate_title_length = 43
        list_of_graphs = []
        with LoggingTimer("TIMER FOR graph_creator()::ForLoop:"):
            for path in unique_paths:

                dff_path = dff[dff['readable_path'] == path]

                if dff_path['has_children'].any():
                    schrodingers_button = dbc.Button([html.I(className='fa fa-angle-down')], id={'type': 'generated-button', 'index': path})
                else:
                    schrodingers_button = html.Div()

                plot = go.Figure()

                # creating graph
                plot.add_trace(go.Scatter(x=dff_path["date"],
                                          y=dff_path["measurement"],
                                          name=chosen_dset,
                                          mode="lines+markers",
                                          marker={'color': ['red' if not pd.isna(a) else 'cornflowerblue' for a in dff_path['note']],
                                                  'symbol': ['square' if not pd.isna(a) else 'circle' for a in dff_path['note']]},
                                          hovertemplate=("Date: %{x}<br>" +
                                                         "Measurement: %{y}<br>" +
                                                         "GitSHA: %{customdata[0]}<br>" +
                                                         "Note: %{customdata[1]}<extra></extra>"),
                                          meta={"path": path},
                                          showlegend=have_second_dataset,
                                          customdata=dff_path[["gitSHA", "truncated_note", "note", "prevGitSHA"]]))

                if not have_second_dataset:

                    # adds the rolling average line
                    df_for_avg = dff_path.copy()
                    df_for_avg['rolling_average'] = df_for_avg['measurement'].rolling(window=10, min_periods=1).mean()
                    plot.add_trace(go.Scatter(x=df_for_avg['date'],
                                              y=df_for_avg['rolling_average'],
                                              mode='lines',
                                              name='Avg.',
                                              showlegend=False,
                                              line={'color': 'red'},
                                              hoverinfo='skip'))

                else:

                    # add graph for second dataset
                    second_dff_path = second_dff[second_dff["readable_path"] == path]
                    plot.add_trace(go.Scatter(x=second_dff_path["date"],
                                              y=second_dff_path["measurement"],
                                              name=second_chosen_dset,
                                              mode="lines+markers",
                                              marker={'color': ['yellow' if not pd.isna(a) else 'red' for a in second_dff_path['note']],
                                                      'symbol': ['square' if not pd.isna(a) else 'circle' for a in second_dff_path['note']]},
                                              line={"color": 'red'},
                                              hovertemplate=("Date: %{x}<br>" +
                                                             "Measurement: %{y}<br>" +
                                                             "GitSHA: %{customdata[0]}<br>" +
                                                             "Note: %{customdata[1]}<extra></extra>"),
                                              showlegend=True,
                                              customdata=second_dff_path[["gitSHA", "truncated_note", "note", "prevGitSHA"]]))

                # add vertical lines for annotations
                for _, elem in dff_path[['date', 'note']].iterrows():
                    if not pd.isna(elem['note']):
                        plot.add_vline(x=elem['date'], line_width=1, line_dash='dot', line_color='green')

                if have_second_dataset:

                    # add vertical lines for annotations
                    for _, elem in second_dff_path[['date', 'note']].iterrows():
                        if not pd.isna(elem['note']):
                            plot.add_vline(x=elem['date'], line_width=1, line_dash='dot', line_color='purple')

                # determine y range
                MARGIN = 0.1
                ymin = dff_path['measurement'].min()
                ymax = dff_path['measurement'].max()
                if have_second_dataset:
                    ymin2 = second_dff_path['measurement'].min()
                    ymax2 = second_dff_path['measurement'].max()
                    if np.isfinite(ymin2):
                        ymin = min(ymin, ymin2)
                    if np.isfinite(ymax2):
                        ymax = max(ymax, ymax2)

                # Truncate the path for ease of reading
                truncated_path = path
                if len(previous_path) > 0:
                    truncated_path = truncated_path[len(previous_path)+1:]
                truncated_path = truncated_path if len(truncated_path) < truncate_title_length else f'...{truncated_path[-truncate_title_length:]}'

                plot.update_layout(
                    title_font_size=12,
                    hoverlabel={'align': "left"},
                    title_text=truncated_path,
                    xaxis={'autorange': True, 'type': 'date'},
                    yaxis={'range': [ymin* (1-MARGIN), ymax * (1+MARGIN)], 'type': 'linear'}
                )

                # add to our list of graphs
                list_of_graphs.append(
                    dbc.Card(id=f'{path}_card',
                             children=[
                                 html.Div(
                                     children=[
                                         dcc.Graph(
                                             id={'type': 'generated-graph', 'index': f'{path}_{counter}'},
                                             figure=plot,
                                             style={'height': '375px', 'width': '375px'}
                                         ),
                                     ],
                                     className='mx-auto'
                                 ),
                                 html.Div(id= {'type': 'stopped-reporting-text', 'index': f'{path}_stopped_reporting_txt'}, className='mx-5 mb-3'),
                                 schrodingers_button
                             ]
                             )
                )
                counter += 1

    logger.info(f"CONTAINED {counter} GRAPHS")

    return [ 
        # html.Button(id={'index': 'DUMMY', 'type': 'generated-button'}),
        dmc.SimpleGrid(
            cols=3,
            spacing='s',
            verticalSpacing='xs',
            children=list_of_graphs
        )
    ]


##################################################
# Annotations callbacks

@callback(
    Output(component_id='annotation-modal', component_property='is_open', allow_duplicate=True),
    Output(component_id='current-stored-depth', component_property='data', allow_duplicate=True),

    Input(component_id='annotation-submit', component_property='n_clicks'),
    State(component_id='annotation-input', component_property='value'),
    State(component_id='dataset-selection', component_property='value'),
    State(component_id='annotation-date-store', component_property='data'),
    State(component_id='current-stored-depth', component_property='data')
)
def annotation_submission(_, annotation_text, dataset_selection, annotation_date, depth_for_update):
    if annotation_text is None:
        return (False, depth_for_update)
    assert annotationsFile.exists()
    annotations_df = pd.read_csv(annotationsFile)
    annotations_df.loc[len(annotations_df)] = [dataset_selection, annotation_date[:10], annotation_text]
    annotations_df.to_csv(annotationsFile, index=False)
    return (False, depth_for_update)


@callback(
    Output(component_id='annotation-modal', component_property='is_open', allow_duplicate=True),
    Output(component_id='current-stored-depth', component_property='data', allow_duplicate=True),
    Input(component_id='annotation-delete', component_property='n_clicks'),
    State(component_id='dataset-selection', component_property='value'),
    State(component_id='annotation-date-store', component_property='data'),
    State(component_id='current-stored-depth', component_property='data')
)
def delete_annotations_button(_, dataset_selection, annotation_date, depth_for_update ):

    assert annotationsFile.exists()
    df = pd.read_csv(annotationsFile)
    annotation_date_trunc = ''
    if annotation_date is not None:
        annotation_date_trunc = annotation_date[:10]
    filtered_df = df.query("not (dataset_name == @dataset_selection and date_only == @annotation_date_trunc)")
    filtered_df.to_csv(annotationsFile, index=False)
    return (False, depth_for_update)


@callback(
    Output('modal-title', 'children'),
    Output('annotation-modal', 'is_open', allow_duplicate=True),
    Output('annotation-date-store', 'data'),
    Output('commit-anchor', 'href'),
    Output('gitsha-modal', 'children'),
    Output('note-modal', 'children'),
    [Input(component_id={'type': 'generated-graph', 'index': ALL}, component_property='clickData')],
    prevent_initial_call = True
)
def display_click_data(click_data):
    '''
    Gets the click data and opens the modal
    '''
    index_string = ctx.triggered_id['index']
    parts = index_string.split('_')

    if len(parts) > 1:
        idxOfGraph = int(parts[-1])
    else:
        idxOfGraph = None


    if click_data[idxOfGraph] is not None:
        logger.info(click_data[idxOfGraph])
        point = click_data[idxOfGraph]['points'][0]
        point_date = point['x'][:10]
        dt_point_date = datetime.strptime(point_date, "%Y-%m-%d")
        yesterday_point_date = str(dt_point_date - timedelta(days=1))[:10]
        gitSHA = point['customdata'][0]
        annotation = point['customdata'][2]
        prevGitSHA = point['customdata'][3]
        commit_href = f"{config["code"]["url"]}/compare/{prevGitSHA}...{gitSHA}"

        return (f"Annotation for {point_date}", True, point['x'], commit_href, f"GitSHA: {gitSHA}", f"NOTE: {annotation}")
    return("ERROR", False, "ERROR", f"{config["code"]["url"]}/compare/", "ERROR", "ERROR")


##################################################
# Navigation callbacks

@callback(
    State(component_id='dataset-selection', component_property='value'),
    State(component_id='second-dataset-selection', component_property='value'),
    Input(component_id='previous-path', component_property='data'),
    Input(component_id='do-display-btn', component_property='n_clicks'),
    Output(component_id='current-dataset', component_property='children'),
    Output(component_id='current-subtest', component_property='children'),
)
def show_current_dataset(dataset, second_dataset, path, _):
    if path == "":
        path = "root level"
    if second_dataset == "None":
        return (f"Displayed dataset: {dataset}", f"Displayed subtest: {path}")
    else:
        return (f"Displayed datasets: {dataset}, {second_dataset}", f"Displayed subtest: {path}")


@callback(
    Output(component_id='current-stored-depth', component_property='data', allow_duplicate=True),
    Output(component_id='previous-path', component_property='data', allow_duplicate=True),
    Input(component_id='top-level-btn', component_property='n_clicks'),
    State(component_id='current-stored-depth', component_property='data'),
    State(component_id='previous-path', component_property='data'),
)
def go_to_top_level(btn, depth, path):
    #returns the previous path
    if depth != 0 and btn is not None:
        logger.info(f"Depth is: 0, path is: /")
        return(0, '')
    else:
        raise PreventUpdate


@callback(
    Output(component_id='current-stored-depth', component_property='data', allow_duplicate=True),
    Output(component_id='previous-path', component_property='data', allow_duplicate=True),
    Input(component_id='previous-level-btn', component_property='n_clicks'),
    State(component_id='current-stored-depth', component_property='data'),
    State(component_id='previous-path', component_property='data'),
)
def go_up_a_level(btn, depth, path):
    #returns the previous path
    if depth > 0 and btn is not None:
        prev_path = '|'.join(path.split('|')[:-1])
        logger.info(f"Depth is: {depth}, path: {path}, previous path: {prev_path}")
        return(depth - 1, prev_path)
    else:
        raise PreventUpdate


@callback(
    Output(component_id='previous-path', component_property='data'),
    Output(component_id='current-stored-depth', component_property='data'),
    Input(component_id={'type': 'generated-button', 'index': ALL}, component_property='n_clicks'),
    State(component_id='current-stored-depth', component_property='data'),
    State(component_id='previous-path', component_property='data'),
    prevent_initial_call = True
)
def dive_in_callback(btns, depth_store, prev_path):
    '''
    This handles the buttons to dive in
    '''
    if not any(btns):
        raise PreventUpdate
    button_triggered_id = ctx.triggered_id
    msg = f'Switch from depth: {depth_store} path: {prev_path}'
    depth_store += 1
    prev_path = button_triggered_id['index']
    logger.info(f'{msg} to depth: {depth_store} path: {prev_path}')
    return (prev_path, depth_store)


##################################################
# Light/dark mode

clientside_callback(
    '''
        (switchOn) => {
       document.documentElement.setAttribute("data-bs-theme", switchOn ? "light" : "dark");
       return window.dash_clientside.no_update
    }
    ''',
    Output("switch", "id"),
    Input("switch", "value"),
)


def main():
    app.run(host=config["server"]["host"],
            port=config["server"]["port"],
            debug=config["debugMode"])


if __name__ == '__main__':
    main()
