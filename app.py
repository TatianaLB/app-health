# Definición de la aplicación

from dash import Dash, html, dcc, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import pandas as pd
from src.model import train_models
from src.etl import prepare_patient_data_with_names, categorizar_edad, load_data
from src.graphics import create_gauge_chart, plot_feature_importance, plot_heatmap, plot_histogram_with_patient, plot_risk_distribution, plot_age_distribution

# Inicializar la aplicación Dash con el tema "BOOTSTRAP"
app = Dash(__name__, title="AppHealth", external_stylesheets=[dbc.themes.BOOTSTRAP])
# Declare server for Heroku deployment. Needed for Procfile.
server = app.server

# Entrenar los modelos
models = train_models()
diabetes_model, diabetes_accuracy = models['diabetes']
hypertension_model, hypertension_accuracy = models['hypertension']

# Diseño de la aplicación
app.layout = dbc.Container(
    fluid=True,  # Para que ocupe toda la página
    style={'backgroundColor': '#FAEBD7', 'padding': '20px'},
    children=[
        dcc.Store(id='prepared-patient-diabetes-store'),
        dcc.Store(id='prepared-patient-hypertension-store'),
        html.H1(
            "Bienvenido a tu Detector de Enfermedades de Confianza",
            style={'textAlign': 'center', 'color': '#444',"text-decoration": "underline"}
        ),
        dbc.Row(
            [
                # Formulario
                dbc.Col(
                    style={
                        'backgroundColor': '#FFAB91',
                        'padding': '20px',
                        'borderRadius': '10px',
                        'marginRight': '10px'
                    },
                    width=4,  # 4 columnas de ancho
                    children=[
                        html.Label("1- Introduzca su edad:"),
                        dcc.Input(id='age-input', type='number', placeholder='Edad', style={
                            'width': '100%', 'padding': '10px', 'marginBottom': '20px', 'borderRadius': '5px'}),
                        html.Div(
                            children=[
                                html.Label("2- Introduzca su BMI (Peso corporal [kg] / Altura^2 [m^2]):"),
                                dcc.Input(id='bmi-input', type='number', placeholder='BMI', style={
                                    'width': '100%', 'padding': '10px', 'marginBottom': '20px', 'borderRadius': '5px'})
                            ],
                            style={'marginBottom': '10px'}
                        ),
                        html.Div(
                            children=[
                                html.Label("3- En general, su salud es (1 Excelente, 5 Horrible):"),
                                dcc.Slider(id='health-slider', min=1, max=5, step=1,
                                        marks={i: str(i) for i in range(1, 6)}, value=3,
                                        tooltip={"placement": "bottom"})
                                ],
                            style={'marginBottom': '20px'}
                        ),
                        html.Label("4- Indique qué tipo de dolor de pecho ha experimentado:"),
                        dcc.RadioItems(id='chest-pain-radio', options=[
                            {'label': 'No siento dolor', 'value': 0},
                            {'label': 'Dolor no relacionado con angina: Es un dolor que no parece estar relacionado con el corazón. Puede ser un dolor muscular o de otra naturaleza (como dolor que aumenta al tocar el área o con ciertos movimientos).', 'value': 1},
                            {'label': 'Dolor atípico de angina: Es un dolor en el pecho que no sigue un patrón claro, no siempre ocurre con esfuerzo ni siempre se alivia con descanso.', 'value': 2},
                            {'label': 'Dolor típico de angina: Sensación de presión en el pecho que ocurre cuando hace algún esfuerzo físico o está bajo estrés, y suele aliviarse cuando descansa.', 'value': 3}
                        ], value=0, style={'marginBottom': '20px'}),
                        html.Label("5- ¿Siente algun dolor u opresión en el pecho al realizar ejercicio físico?"),
                        html.Label("(0: No siente ningún dolor u opresión durante el ejercicio / 6: Siente un dolor o una fuerte opresión que me obliga a detenerme por completo.)"),
                        dcc.Slider(id='pain-slider', min=0, max=6, step=0.1,
                                   marks={i: str(i) for i in range(7)}, value=3,
                                   tooltip={"placement": "bottom"}),
                        html.Div(
                            style={'textAlign': 'center', 'marginTop': '10px'},
                            children=[
                                html.Button("Mostrar resultados", id='submit-button',
                                            style={'backgroundColor': '#FF7043', 'color': 'white',
                                                   'border': 'none', 'padding': '10px 20px',
                                                   'borderRadius': '5px', 'cursor': 'pointer'})
                            ]
                        )
                    ]
                ),
                # Gráficos gauge
                dbc.Col(
                    id='results-container',
                    style={
                        'backgroundColor': '#FFFFFF',
                        'padding': '10px',
                        'borderRadius': '10px',
                    },
                    width=7
                )
            ],
            justify='center'
        ),
        html.Hr(),
        # Nuevo div para los gráficos de importancia de variables y heatmaps
        html.Div(
            id='importance-heatmap-container',
            style={'backgroundColor': '#FFFFFF', 'padding': '30px', 'borderRadius': '10px', 'marginTop': '20px', 'marginBottom': '20px', 'marginRight': '60px', 'marginLeft': '60px'},
            children=[]
        ),
        html.Hr(),
        # Añadir el selector de gráficos adicionales dentro de una fila y columna con estilo similar al formulario
        dbc.Row(
            [
                dbc.Col(
                    style={
                        'backgroundColor': '#FFAB91',
                        'padding': '20px',
                        'borderRadius': '10px',
                        'marginRight': '10px'
                    },
                    width=4,
                    children=[
                        html.Label("Seleccione los gráficos adicionales que desea visualizar:"),
                        dcc.Checklist(
                            id='additional-graphs-checklist',
                            options=[
                                {'label': 'Distribución de Riesgo de Diabetes', 'value': 'risk_diabetes'},
                                {'label': 'Distribución de Riesgo de Hipertensión', 'value': 'risk_hypertension'},
                                {'label': 'Distribución de BMI', 'value': 'bmi_distribution'},
                                {'label': 'Distribución de Edad', 'value': 'age_distribution'},
                                {'label': 'Distribución de Frecuencia Cardíaca Máxima', 'value': 'heart_rate_distribution'}
                            ],
                            value=[],
                            style={'marginBottom': '20px', 'display': 'inline-block', 'textAlign': 'left'}
                        ),
                        html.Div(
                            style={'textAlign': 'center', 'marginTop': '20px'},
                            children=[
                                html.Button("Mostrar gráficos seleccionados", id='show-graphs-button',
                                            style={'backgroundColor': '#FF7043', 'color': 'white',
                                                'border': 'none', 'padding': '10px 20px',
                                                'borderRadius': '5px', 'cursor': 'pointer'})
                            ]
                        )
                    ]
                ),
                # Columna para mostrar los gráficos seleccionados
                dbc.Col(
                    id='additional-graphs-container',
                    style={
                        'backgroundColor': '#FFFFFF',
                        'padding': '10px',
                        'borderRadius': '10px',
                    },
                    width=7
                )
            ],
            justify='center'
        )

    ]
)

@app.callback(
    [Output('results-container', 'children'),
     Output('importance-heatmap-container', 'children'),
     Output('prepared-patient-diabetes-store', 'data'),
     Output('prepared-patient-hypertension-store', 'data')],
    [Input('submit-button', 'n_clicks')],
    [State('age-input', 'value'),
     State('bmi-input', 'value'),
     State('health-slider', 'value'),
     State('chest-pain-radio', 'value'),
     State('pain-slider', 'value')]
)
def display_results(n_clicks, age, bmi, health, chest_pain, pain):
    if n_clicks:
        try:
            if age is None or bmi is None:
                # Devolver cuatro valores, asegurando que todos los outputs se cumplen
                return "Por favor, complete todos los campos antes de continuar.", [], None, None

            # Preparar los datos del paciente
            patient_diabetes = [bmi, categorizar_edad(age), health]
            patient_hypertension = [chest_pain, 200 - age, pain]

            diabetes_features_imp = ['BMI', 'Age', 'GenHlth']
            prepared_patient_diabetes = prepare_patient_data_with_names(patient_diabetes, diabetes_features_imp)
            hypertension_features_imp = ['cp', 'thalach', 'oldpeak']
            prepared_patient_hypertension = prepare_patient_data_with_names(patient_hypertension, hypertension_features_imp)

            # Obtener probabilidades
            diabetes_prob = diabetes_model.predict_proba(prepared_patient_diabetes)[0][1]
            hypertension_prob = hypertension_model.predict_proba(prepared_patient_hypertension)[0][1]

            # Crear gráficos de resultados principales
            gauge_diabetes = create_gauge_chart(diabetes_prob, "Nivel de Riesgo Diabetes")
            gauge_hypertension = create_gauge_chart(hypertension_prob, "Nivel de Riesgo Hipertensión")
            

            # Crear gráficos de importancia de variables y heatmaps
            diabetes_importances_imp = [0.2117133332645621, 0.15305638261809465, 0.11646282783471759]
            feature_importance_diabetes = plot_feature_importance(diabetes_features_imp, diabetes_importances_imp, title="Importancia de las Variables para Diabetes")
            hypertension_importances_imp = [0.1491391772301424, 0.13647081077164086, 0.12473399197765939]
            feature_importance_hypertension = plot_feature_importance(hypertension_features_imp, hypertension_importances_imp, title="Importancia de las Variables para Hipertensión")
            feature_importance_diabetes.update_layout(height=400, width=600)
            feature_importance_hypertension.update_layout(height=400, width=600)

            df_diabetes, df_hypertension = load_data()
            heatmap_diabetes = plot_heatmap(df_diabetes, diabetes_features_imp, "Heatmap de Variables para Diabetes")
            heatmap_hypertension = plot_heatmap(df_hypertension, hypertension_features_imp, "Heatmap de Variables para Hipertensión")
            heatmap_diabetes.update_layout(height=500, width=600)
            heatmap_hypertension.update_layout(height=500, width=600)

            # Gráficos alineados en filas y columnas
            results_graphs = [
                dbc.Row(
                    [
                        dbc.Col(dcc.Graph(figure=gauge_diabetes), width=9, style={'textAlign': 'center'}),
                    ],
                    justify='center'
                ),
                dbc.Row(
                    [
                        dbc.Col(dcc.Graph(figure=gauge_hypertension), width=9, style={'textAlign': 'center'}),
                    ],
                    justify='center'
                )
            ]

            importance_heatmap_graphs = [
                dbc.Row(
                    [
                        dbc.Col(dcc.Graph(figure=feature_importance_diabetes), width=6, style={'textAlign': 'center'}),
                        dbc.Col(dcc.Graph(figure=feature_importance_hypertension), width=6, style={'textAlign': 'center'})
                    ],
                    justify='center'
                ),
                dbc.Row(
                    [
                        dbc.Col(dcc.Graph(figure=heatmap_diabetes), width=6, style={'textAlign': 'center'}),
                        dbc.Col(dcc.Graph(figure=heatmap_hypertension), width=6, style={'textAlign': 'center'})
                    ],
                    justify='center'
                )
            ]

            return results_graphs, importance_heatmap_graphs, prepared_patient_diabetes.to_dict(), prepared_patient_hypertension.to_dict()

        except Exception as e:
            # Devolver cuatro valores en caso de excepción
            return f"Error al procesar los datos: {str(e)}", [], None, None

    # Devolver cuatro valores cuando no se ha hecho clic en el botón
    return "Introduzca los datos y haga click en Mostrar resultados.", [], None, None



# Añadir callback para manejar los gráficos adicionales seleccionados
@app.callback(
    Output('additional-graphs-container', 'children'),
    Input('show-graphs-button', 'n_clicks'),
    State('additional-graphs-checklist', 'value'),
    State('prepared-patient-diabetes-store', 'data'),
    State('prepared-patient-hypertension-store', 'data')
)
def display_additional_graphs(n_clicks, selected_graphs, prepared_patient_diabetes_data, prepared_patient_hypertension_data):
    if n_clicks:
        graphs = []

        # Cargar los datos necesarios
        df_diabetes, df_hypertension = load_data()
        diabetes_features_imp = ['BMI', 'Age', 'GenHlth']
        hypertension_features_imp = ['cp', 'thalach', 'oldpeak']

        # Convertir los datos almacenados de vuelta a DataFrames
        prepared_patient_diabetes = pd.DataFrame.from_dict(prepared_patient_diabetes_data)
        prepared_patient_hypertension = pd.DataFrame.from_dict(prepared_patient_hypertension_data)

        # Generar las gráficas según las opciones seleccionadas
        if 'risk_diabetes' in selected_graphs:
            diabetes_population_probabilities = diabetes_model.predict_proba(df_diabetes[diabetes_features_imp])[:, 1] 
            diabetes_prob = diabetes_model.predict_proba(prepared_patient_diabetes)[0][1]
            graph = plot_risk_distribution(diabetes_population_probabilities, diabetes_prob, "Distribución de Riesgo de Diabetes")
            graphs.append(dcc.Graph(figure=graph))
        
        if 'risk_hypertension' in selected_graphs:
            hypertension_population_probabilities = hypertension_model.predict_proba(df_hypertension[hypertension_features_imp])[:, 1]
            hypertension_prob = hypertension_model.predict_proba(prepared_patient_hypertension)[0][1]
            graph = plot_risk_distribution(hypertension_population_probabilities, hypertension_prob, "Distribución de Riesgo de Hipertensión")
            graphs.append(dcc.Graph(figure=graph))

        if 'bmi_distribution' in selected_graphs:
            graph = plot_histogram_with_patient(df_diabetes, prepared_patient_diabetes, 'BMI', "Distribución de BMI")
            graphs.append(dcc.Graph(figure=graph))

        if 'age_distribution' in selected_graphs:
            graph = plot_age_distribution(df_diabetes["Age"].to_numpy().astype(int), prepared_patient_diabetes['Age'].iloc[0].astype(int))
            graphs.append(dcc.Graph(figure=graph))

        if 'heart_rate_distribution' in selected_graphs:
            graph = plot_histogram_with_patient(df_hypertension, prepared_patient_hypertension, 'thalach', "Distribución de Frecuencia Cardíaca Máxima")
            graphs.append(dcc.Graph(figure=graph))

        return graphs

    return []

if __name__ == '__main__':
    app.run_server(debug=True)
