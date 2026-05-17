# README — Proyecto de Detección de Fraude Bancario y Modelo Federado

````markdown
# Detección de Fraude Bancario con Métricas Personalizadas y Aprendizaje Federado

Proyecto desarrollado para el convenio **PLUS TI – Universidad del Valle 2025**, enfocado en la detección de fraude en transacciones bancarias utilizando modelos de Machine Learning, métricas personalizadas y estrategias de aprendizaje federado.

---

# Integrantes

- Silvia Illescas
- Davis Roldán
- Michelle Mejía

---

# Descripción General del Proyecto

El proyecto se divide en dos grandes componentes:

## Parte A — Optimización de métricas para reducción de falsos positivos

Se busca entrenar modelos de clasificación binaria capaces de detectar transacciones fraudulentas minimizando la cantidad de falsos positivos, utilizando métricas de evaluación personalizadas (`feval`) en LightGBM.

El objetivo específico asignado al grupo consiste en:

> Mejorar la detección de fraudes cometidos en establecimientos de hospedaje.

Esto implica desarrollar estrategias enfocadas en detectar patrones de fraude asociados a:
- Hoteles
- Hospedajes
- Reservas
- Comercios de travel/lodging
- Transacciones internacionales asociadas al turismo

---

## Parte B — Entrenamiento de Modelo Federado

Se implementa un modelo federado utilizando datasets de múltiples bancos con distribuciones de fraude diferentes.

El objetivo consiste en:
- Entrenar con los datasets etiquetados de Banco 1 y Banco 2.
- Generar inferencias sobre Banco 3, el cual no contiene la variable objetivo (`is_fraud`).

El reto principal es lograr un modelo con buena capacidad de generalización entre distintos países, perfiles de clientes y tipologías de fraude.

---

# Dataset

El proyecto utiliza datasets sintéticos basados en el estándar financiero:

## ISO 8583

El estándar ISO 8583 define mensajes de transacciones financieras utilizadas en redes de tarjetas de crédito y débito.

Los datasets incluyen:
- Transacciones legítimas
- Transacciones fraudulentas
- Variables bancarias
- Variables de comportamiento
- Variables geográficas
- Variables derivadas para fraude

---

# Variables Importantes Utilizadas

Algunas de las variables más relevantes para el proyecto son:

| Variable | Descripción |
|---|---|
| merchant_category_code | Categoría del comercio (MCC) |
| amount_transaction | Monto de la transacción |
| is_international | Indica si la transacción es internacional |
| is_online | Indica si la transacción es e-commerce |
| distance_from_home_km | Distancia entre cliente y comercio |
| hour_of_day | Hora local de la transacción |
| pos_entry_mode | Método de captura de tarjeta |
| fraud_type | Tipo de fraude |
| card_acceptor_name_location | Nombre y ubicación del comercio |

---

# Objetivo del Grupo

El enfoque principal del proyecto es optimizar la detección de fraude en establecimientos de hospedaje.

Para ello se investigarán patrones asociados a:
- Reservas internacionales
- Compras online en hoteles
- Transacciones de alto monto
- Distancias geográficas anómalas
- Actividad fuera del comportamiento habitual del cliente

---

# Tecnologías Utilizadas

## Lenguaje
- Python 3.11+

## Librerías principales

```python
pandas
numpy
matplotlib
seaborn
scikit-learn
lightgbm
xgboost
optuna
````

## Entorno recomendado

* Google Colab
* Jupyter Notebook
* VSCode

---

# Estructura del Proyecto

```plaintext
fraud-detection-project/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── submissions/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_model.ipynb
│   ├── 04_custom_metrics.ipynb
│   ├── 05_optimization.ipynb
│   └── 06_federated_model.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── features.py
│   ├── metrics.py
│   ├── train.py
│   ├── inference.py
│   └── utils.py
│
├── outputs/
│   ├── figures/
│   ├── metrics/
│   └── models/
│
├── report/
│   └── informe.pdf
│
├── requirements.txt
└── README.md
```

---

# Metodología

# 1. Exploratory Data Analysis (EDA)

Se realiza:

* análisis de distribución,
* balanceo de clases,
* detección de anomalías,
* análisis temporal,
* análisis geográfico,
* comportamiento por MCC.

---

# 2. Ingeniería de Variables

Se desarrollan variables derivadas orientadas a mejorar la detección de fraude.

Ejemplos:

```python
time_since_last_txn
txn_count_last_1h
txn_count_last_24h
hotel_amount_zscore
distance_from_home_km
```

También se generan features especializadas para establecimientos de hospedaje.

---

# 3. Modelo Base

Se implementa un modelo inicial utilizando:

```python
LightGBM
```

Métricas utilizadas:

* ROC-AUC
* Precision
* Recall
* F1-score

---

# 4. Métricas Personalizadas

Se desarrollan funciones `feval` personalizadas orientadas a:

* reducir falsos positivos,
* priorizar precisión,
* mejorar recall en fraudes de hospedaje.

Ejemplo conceptual:

```python
score = hotel_recall - alpha * false_positive_rate
```

---

# 5. Optimización

Se realiza:

* ajuste de hiperparámetros,
* tuning de threshold,
* balanceo de clases,
* calibración de probabilidades.

---

# 6. Modelo Federado

Se entrenan modelos utilizando datasets de múltiples bancos.

Objetivos:

* evitar overfitting a patrones locales,
* mejorar capacidad de generalización,
* detectar patrones de fraude no vistos previamente.

---

# Ejecución del Proyecto

# 1. Clonar repositorio

```bash
git clone https://github.com/michellemej22596/Proyecto2-SecurityDS-PlusTI.git
cd Proyecto2-SecurityDS-PlusTI
```

---

# 2. Crear entorno virtual

## Windows

```bash
python -m venv venv
venv\Scripts\activate
```

## Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

---

# 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

# 4. Ejecutar notebooks

Abrir Jupyter Notebook o Google Colab y ejecutar:

```plaintext
01_eda.ipynb
02_feature_engineering.ipynb
03_baseline_model.ipynb
04_custom_metrics.ipynb
05_optimization.ipynb
06_federated_model.ipynb
```

---

# Métrica Principal del Proyecto

La métrica principal utilizada para evaluar falsos positivos es:

FP Ratio = FP / (TP + FP)

Donde:

* FP = False Positives
* TP = True Positives

El objetivo consiste en minimizar esta ratio manteniendo alta capacidad de detección de fraude.

---

# Resultados Esperados

Se espera:

* reducir falsos positivos,
* mejorar detección en establecimientos de hospedaje,
* aumentar capacidad de generalización entre bancos,
* obtener un modelo robusto para fraude financiero.

---

# Recomendaciones

* Documentar cada experimento realizado.
* Mantener reproducibilidad de resultados.
* Evitar data leakage.
* Validar correctamente las divisiones temporales.
* Analizar drift entre bancos.

---

# Referencias

* ISO 8583 Financial Transaction Card Originated Messages
* LightGBM Documentation
* Scikit-learn Documentation
* PLUS TI – Universidad del Valle 2025

---

# Licencia

Proyecto desarrollado con fines académicos para el convenio PLUS TI – Universidad del Valle 2025.

```
```