from plotly.graph_objects import Figure, Indicator
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

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

    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.index,
        colorscale='Oranges',
        colorbar=dict(title='Correlación')
    ))
    
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
    counts, bins = np.histogram(data[feature], bins=13)

    hist_fig = go.Figure()

    hist_fig.add_trace(go.Bar(
        x=bins[:-1],  
        y=counts,
        width=(bins[1] - bins[0]),  
        marker=dict(
            color='#FFB74D',  
            line=dict(color='black', width=1.5) 
        ),
        opacity=0.7,
        name="Población"
    ))

    patient_val = patient_value[feature].iloc[0]
    hist_fig.add_trace(go.Scatter(
        x=[patient_val, patient_val], 
        y=[0, max(counts) * 1.1],  
        mode='lines',
        line=dict(color='red', width=3, dash='dash'),  
        name='Paciente'
    ))

    hist_fig.update_layout(
        title=dict(text=title, font=dict(size=18), x=0.5),
        xaxis_title=feature,
        yaxis_title="Frecuencia",
        xaxis=dict(title_font=dict(size=14), tickfont=dict(size=12), showgrid=False),
        yaxis=dict(title_font=dict(size=14), tickfont=dict(size=12), showgrid=False, range=[0, max(counts) * 1.2]),
        plot_bgcolor='white',
        bargap=0, 
        legend=dict(font=dict(size=12), x=0.85, y=0.95, bgcolor='rgba(255, 255, 255, 0.5)'),
        margin=dict(l=20, r=20, t=50, b=50)
    )

    return hist_fig


def plot_risk_distribution(predicted_probabilities, patient_probability, title="Distribución de Riesgo"):
    counts, bins = np.histogram(predicted_probabilities, bins=13)

    hist_fig = go.Figure()

    hist_fig.add_trace(go.Bar(
        x=bins[:-1], 
        y=counts,
        width=(bins[1] - bins[0]),  
        marker=dict(
            color='#FFB74D',   
            line=dict(color='black', width=1)  
        ),
        opacity=0.7,
        name="Población"
    ))

    hist_fig.add_trace(go.Scatter(
        x=[patient_probability, patient_probability],
        y=[0, max(counts)],  
        mode='lines',
        line=dict(color='red', width=3, dash='dash'),  
        name='Paciente'
    ))

    hist_fig.update_layout(
        title=dict(text=title, font=dict(size=18), x=0.5),
        xaxis_title="Probabilidad de Riesgo",
        yaxis_title="Frecuencia",
        xaxis=dict(title_font=dict(size=14), tickfont=dict(size=12), showgrid=False),
        yaxis=dict(title_font=dict(size=14), tickfont=dict(size=12), showgrid=False),
        plot_bgcolor='white',
        bargap=0,  
        legend=dict(font=dict(size=12)),
        margin=dict(l=20, r=20, t=50, b=50),
    )

    return hist_fig

def plot_age_distribution(predicted_ages, patient_age, title="Comparación de su edad con la población"):
    # Definir los rangos de edad
    rango_edades = {
        0: '0-18', 1: '18-24', 2: '25-29', 3: '30-34',
        4: '35-39', 5: '40-44', 6: '45-49', 7: '50-54',
        8: '55-59', 9: '60-64', 10: '65-69', 11: '70-74',
        12: '75-79', 13: '80+'
    }

    counts, bins = np.histogram(predicted_ages, bins=13)

    hist_fig = go.Figure()

    hist_fig.add_trace(go.Bar(
        x=bins[:-1] + (bins[1] - bins[0]) / 2,  
        y=counts,
        width=(bins[1] - bins[0]),  
        marker=dict(
            color='#FFB74D',  
            line=dict(color='black', width=1) 
        ),
        opacity=0.7,
        name="Población"
    ))

    hist_fig.add_trace(go.Scatter(
        x=[patient_age, patient_age],
        y=[0, max(counts)],  
        mode='lines',
        line=dict(color='red', width=3, dash='dash'), 
        name='Paciente'
    ))

    hist_fig.update_layout(
        title=dict(text=title, font=dict(size=18), x=0.5),
        xaxis=dict(
            title="Edad (rango)",
            tickmode='array',
            tickvals=list(range(len(rango_edades))),
            ticktext=list(rango_edades.values()),
            tickfont=dict(size=12),
            title_font=dict(size=14)
        ),
        yaxis=dict(
            title="Frecuencia",
            title_font=dict(size=14),
            tickfont=dict(size=12)
        ),
        plot_bgcolor='white',
        bargap=0,  
        legend=dict(font=dict(size=12)),
        margin=dict(l=20, r=20, t=50, b=50),
    )

    return hist_fig
