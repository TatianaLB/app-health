from plotly.graph_objects import Figure, Indicator
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import seaborn as sns

def create_gauge_chart(probability, title="Nivel de Riesgo"):
    fig = Figure()

    fig.add_trace(Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={'suffix': "%", 'font': {'size': 70}},
        title={'text': title, 'font': {'size': 20}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "black"},
            'bar': {'color': "red" if probability > 0.5 else "green"},
            'steps': [
                {'range': [0, 50], 'color': "lightgreen"},
                {'range': [50, 100], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "blue", 'width': 6},
                'thickness': 0.75,
                'value': probability * 100
            }
        }
    ))

    fig.update_layout(
        font={'color': "black", 'family': "Arial"}
        
    )
    return fig

def plot_feature_importance(features, importances, title="Importancia de las Variables"):
    # Ordenar las características por importancia de mayor a menor
    sorted_data = sorted(zip(importances, features), reverse=True)
    importances, features = zip(*sorted_data)
    
    data = {
        'Variables': features,
        'Importancia': importances
    }
    
    fig = px.bar(
        data,
        x='Importancia',
        y='Variables',
        orientation='h',
        title=title,
        color='Importancia',
        color_continuous_scale='oranges'
    )
    
    fig.update_layout(
        xaxis_title="Importancia",
        yaxis_title="Variables",
        title_x=0.5,
        height=350, 
        plot_bgcolor='white'
    , autosize=True, margin=dict(l=30, r=30, t=50, b=50))
    
    return fig


def plot_heatmap(data, features, title="Relación entre Variables"):
    # Crear una matriz de correlación
    correlation_matrix = data[features].corr()
    
    # Crear el heatmap utilizando Plotly
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.index,
        colorscale='Oranges',
        colorbar=dict(title='Correlación')
    ))
    
    # Añadir anotaciones a la figura con los valores de la correlación
    for i in range(len(correlation_matrix)):
        for j in range(len(correlation_matrix.columns)):
            fig.add_annotation(
                x=correlation_matrix.columns[j],
                y=correlation_matrix.index[i],
                text=str(round(correlation_matrix.values[i][j], 2)),
                showarrow=False,
                font=dict(color="black" if abs(correlation_matrix.values[i][j]) < 0.7 else "white")
            )
    
    fig.update_layout(
        title=title,
        xaxis_title="Variables",
        yaxis_title="Variables",
        title_x=0.5,
        plot_bgcolor='white',
        autosize=True,
        margin=dict(l=30, r=30, t=50, b=50)
    )
    
    return fig


def plot_histogram_with_patient(data, patient_value, feature, title):
    # Calcular el histograma manualmente para personalizar el borde
    counts, bins = np.histogram(data[feature], bins=13)
    
    # Crear la figura manualmente con barras que tengan contorno
    hist_fig = go.Figure()

    # Agregar las barras del histograma
    hist_fig.add_trace(go.Bar(
        x=bins[:-1],  # Se utiliza el límite inferior de cada bin
        y=counts,
        width=(bins[1] - bins[0]),  # Ancho de cada barra
        marker=dict(
            color='#FFB74D',  # Color naranja
            line=dict(color='black', width=1.5)  # Contorno negro
        ),
        opacity=0.7,
        name="Población"
    ))

    # Agregar la línea que representa el valor del paciente
    patient_val = patient_value[feature].iloc[0]
    hist_fig.add_trace(go.Scatter(
        x=[patient_val, patient_val], 
        y=[0, max(counts) * 1.1],  # Extender un poco más para que llegue al borde superior
        mode='lines',
        line=dict(color='red', width=3, dash='dash'),  # Línea roja con estilo dash
        name='Paciente'
    ))

    # Ajustar las etiquetas y el diseño del gráfico
    hist_fig.update_layout(
        title=dict(text=title, font=dict(size=18), x=0.5),
        xaxis_title=feature,
        yaxis_title="Frecuencia",
        xaxis=dict(title_font=dict(size=14), tickfont=dict(size=12), showgrid=False),
        yaxis=dict(title_font=dict(size=14), tickfont=dict(size=12), showgrid=False, range=[0, max(counts) * 1.2]),
        plot_bgcolor='white',
        bargap=0,  # Sin separación entre las barras
        legend=dict(font=dict(size=12), x=0.85, y=0.95, bgcolor='rgba(255, 255, 255, 0.5)'),
        margin=dict(l=20, r=20, t=50, b=50)
    )

    return hist_fig





def plot_risk_distribution(predicted_probabilities, patient_probability, title="Distribución de Riesgo"):
    # Calcular el histograma manualmente para personalizar el borde
    counts, bins = np.histogram(predicted_probabilities, bins=13)
    
    # Crear la figura manualmente con barras que tengan contorno
    hist_fig = go.Figure()

    # Agregar las barras del histograma
    hist_fig.add_trace(go.Bar(
        x=bins[:-1],  # Se utiliza el límite inferior de cada bin
        y=counts,
        width=(bins[1] - bins[0]),  # Ancho de cada barra
        marker=dict(
            color='#FFB74D',  # Color naranja
            line=dict(color='black', width=1)  # Contorno negro
        ),
        opacity=0.7,
        name="Población"
    ))

    # Agregar la línea que representa el valor del paciente
    hist_fig.add_trace(go.Scatter(
        x=[patient_probability, patient_probability],
        y=[0, max(counts)],  # Usar la altura máxima calculada
        mode='lines',
        line=dict(color='red', width=3, dash='dash'),  # Línea roja discontinua
        name='Paciente'
    ))

    # Actualizar las etiquetas y estilo del gráfico
    hist_fig.update_layout(
        title=dict(text=title, font=dict(size=18), x=0.5),
        xaxis_title="Probabilidad de Riesgo",
        yaxis_title="Frecuencia",
        xaxis=dict(title_font=dict(size=14), tickfont=dict(size=12), showgrid=False),
        yaxis=dict(title_font=dict(size=14), tickfont=dict(size=12), showgrid=False),
        plot_bgcolor='white',
        bargap=0,  # Sin separación entre las barras
        legend=dict(font=dict(size=12)),
        margin=dict(l=20, r=20, t=50, b=50),
    )

    return hist_fig



def plot_age_distribution(df_diabetes, patient_age_category, title="Distribución de Edad"):
    # Definir los rangos de edad para cada categoría
    rango_edades = {
        1: '18-24',
        2: '25-29',
        3: '30-34',
        4: '35-39',
        5: '40-44',
        6: '45-49',
        7: '50-54',
        8: '55-59',
        9: '60-64',
        10: '65-69',
        11: '70-74',
        12: '75-79',
        13: '80+'
    }

    # Asignar cada edad a su categoría correspondiente (de 1 a 13)
    bins = [18, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79, 100]
    age_categories = np.digitize(df_diabetes['Age'], bins)

    # Contar la cantidad de ocurrencias por cada categoría
    counts = [np.sum(age_categories == i) for i in range(1, 14)]

    # Crear la figura manualmente con barras que tengan contorno
    age_fig = go.Figure()

    # Agregar las barras del histograma con los contornos negros
    age_fig.add_trace(go.Bar(
        x=list(rango_edades.values()),  # Los nombres de los rangos de edad como etiquetas
        y=counts,
        marker=dict(
            color='orange',  # Color de las barras
            line=dict(color='black', width=1)  # Contorno negro
        ),
        opacity=0.6,
        name="Población"
    ))

    # Obtener el rango de edad correspondiente a la categoría del paciente
    if patient_age_category in rango_edades:
        patient_age_range = rango_edades[patient_age_category]
    else:
        patient_age_range = 'Desconocido'

    # Agregar la línea vertical que representa la edad del paciente
    age_fig.add_trace(go.Scatter(
        x=[patient_age_range, patient_age_range],
        y=[0, max(counts)],
        mode='lines',
        line=dict(color='red', width=2, dash='dash'),
        name='Paciente'
    ))

    # Actualizar las etiquetas y estilo del gráfico
    age_fig.update_layout(
        title=dict(text=title, font=dict(size=18), x=0.5),
        xaxis_title="Edad (rango)",
        yaxis_title="Frecuencia",
        xaxis=dict(title_font=dict(size=14), tickfont=dict(size=12)),
        yaxis=dict(title_font=dict(size=14), tickfont=dict(size=12)),
        plot_bgcolor='white',
        bargap=0,  # Sin separación entre las barras
        legend=dict(font=dict(size=12)),
        margin=dict(l=20, r=20, t=50, b=50)
    )

    return age_fig

