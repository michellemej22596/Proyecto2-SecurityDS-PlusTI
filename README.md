# Detección de Fraude Bancario con Métricas Personalizadas y Aprendizaje Federado

## PLUS TI – Universidad del Valle 2025

### Integrantes

* Silvia Illescas
* Davis Roldán
* Michelle Mejía

---

# Resumen Ejecutivo

Este proyecto tiene como objetivo desarrollar modelos de Machine Learning para la detección de fraude en transacciones bancarias utilizando datasets sintéticos basados en el estándar ISO 8583.

El trabajo se divide en dos componentes principales:

### Parte A — Optimización de Métricas Personalizadas

Se diseñaron y evaluaron métricas personalizadas para LightGBM enfocadas en reducir la cantidad de falsos positivos manteniendo una capacidad de detección superior al 90%, con especial énfasis en fraudes asociados a establecimientos de hospedaje.

### Parte B — Modelo Federado

Se implementó una estrategia de aprendizaje federado basada en modelos locales entrenados sobre dos bancos distintos, permitiendo generar inferencias sobre un tercer banco sin acceso a etiquetas de fraude.

---

# Objetivos

## Objetivo General

Desarrollar modelos robustos para la detección de fraude financiero que permitan maximizar la detección de transacciones fraudulentas minimizando el impacto operativo generado por falsos positivos.

## Objetivos Específicos

* Analizar el comportamiento de fraude en transacciones financieras.
* Diseñar variables derivadas orientadas a la detección de anomalías.
* Implementar métricas personalizadas para LightGBM.
* Optimizar hiperparámetros utilizando Optuna.
* Evaluar el desempeño sobre datos temporales no observados.
* Implementar una estrategia federada para inferir fraude en un banco sin etiquetas.

---

# Dataset

Se utilizaron tres datasets sintéticos generados por PLUS TI, simulando operaciones bancarias reales bajo el estándar ISO 8583.

| Banco   | País                | Registros |
| ------- | ------------------- | --------: |
| Banco 1 | Bolivia (VIP)       |   100,000 |
| Banco 2 | Brasil (Privado)    |   100,000 |
| Banco 3 | Guatemala (Estatal) |   100,000 |

El dataset presenta aproximadamente un 5% de transacciones fraudulentas, constituyendo un problema altamente desbalanceado.

---

# Metodología

## 1. Exploratory Data Analysis (EDA)

Se realizó:

* análisis de calidad de datos,
* distribución de variables,
* balanceo de clases,
* comportamiento temporal,
* análisis geográfico,
* análisis por canal,
* análisis por categoría de comercio (MCC).

### Hallazgos relevantes

* Fraude global aproximado: 4.9%
* Canal ECOM con mayor incidencia de fraude.
* Montos fraudulentos significativamente superiores al promedio.
* Determinados modos de captura POS presentan patrones altamente asociados al fraude.

---

## 2. Ingeniería de Variables

Se desarrollaron variables orientadas a capturar anomalías de comportamiento.

### Variables principales

* time_since_last_txn_min
* txn_count_last_1h
* txn_count_last_24h
* amount_zscore_customer
* rapid_country_change
* hotel_new_country
* hotel_amount_zscore
* hotel_pos_anomaly
* hotel_x_amount_ratio
* hotel_x_distance

Estas variables permitieron capturar patrones de velocidad, comportamiento histórico y anomalías específicas relacionadas con establecimientos de hospedaje.

---

## 3. Modelo Base

Se implementó un modelo inicial utilizando LightGBM.

### Resultados Base

| Métrica   | Valor |
| --------- | ----: |
| ROC-AUC   | 0.903 |
| PR-AUC    | 0.772 |
| Precision | 84.9% |
| Recall    | 72.4% |
| F1 Score  | 0.782 |

Este modelo sirvió como punto de referencia para evaluar las métricas personalizadas.

---

## 4. Métricas Personalizadas

Se desarrollaron seis funciones de evaluación especializadas para la detección de fraude en hospedajes:

* hotel_fp_ratio
* hotel_recall_minus_fpratio
* hotel_fbeta_precision
* hotel_cost_matrix
* hotel_precision_at_min_recall
* composite_hotel_global

### Mejor Métrica

La métrica con mejor desempeño fue:

hotel_recall_minus_fpratio

Esta función permitió priorizar la detección de fraude en hospedajes manteniendo un recall superior al 90%.

---

## 5. Optimización

Se utilizó Optuna para realizar búsqueda automática de hiperparámetros.

### Mejor configuración obtenida

* Recall Hotel: 91.5%
* Precision Hotel: 6.7%
* FP Ratio Hotel: 93.3%
* AUC Global: 0.894
* PR-AUC Global: 0.794

La optimización logró mejorar consistentemente el desempeño respecto al modelo base bajo la restricción de mantener alta sensibilidad.

---

## 6. Modelo Federado

Se entrenaron modelos independientes utilizando los datasets etiquetados de Banco 1 y Banco 2.

Posteriormente se construyó un esquema federado basado en agregación de modelos locales para realizar inferencias sobre Banco 3.

### Resultados

| Modelo       |   AUC |
| ------------ | ----: |
| Banco 1      | 0.877 |
| Banco 2      | 0.809 |
| Federado     | 0.893 |
| Centralizado | 0.890 |

El modelo federado logró un rendimiento comparable e incluso ligeramente superior al modelo centralizado de referencia.

---

# Estructura del Proyecto

```plaintext
.
├── 01_eda.ipynb
├── 02_feature_engineering.ipynb
├── 03_baseline_model.ipynb
├── 04_custom_metrics.ipynb
├── 05_optimization.ipynb
├── 06_federated_model.ipynb
├── data/
├── outputs/
└── README.md
```

---

# Principales Hallazgos

* Las variables de comportamiento temporal fueron las más predictivas.
* Los cambios rápidos de país representan un fuerte indicador de fraude.
* Los patrones asociados a hospedajes presentan características diferenciadas respecto al fraude general.
* El modelo federado mostró capacidad de generalización entre bancos con perfiles distintos.
* Mantener niveles de detección superiores al 90% implica aceptar una proporción considerable de falsos positivos, característica común en sistemas antifraude reales.

---

# Conclusiones

* La ingeniería de variables tuvo un impacto mayor que la optimización de hiperparámetros.
* Las métricas personalizadas permitieron alinear el modelo con objetivos de negocio específicos.
* La estrategia federada permitió transferir conocimiento entre bancos sin requerir acceso a etiquetas del banco objetivo.
* El proyecto demuestra cómo adaptar modelos de Machine Learning a problemas financieros reales donde la reducción de falsos positivos es tan importante como la detección de fraude.

---

# Tecnologías Utilizadas

* Python
* Pandas
* NumPy
* LightGBM
* Scikit-Learn
* Optuna
* Matplotlib
* Seaborn

---

# Referencias

* ISO 8583 Financial Transaction Card Originated Messages
* LightGBM Documentation
* Scikit-Learn Documentation
* PLUS TI – Universidad del Valle 2025

---

# Licencia

Proyecto desarrollado con fines académicos para el convenio PLUS TI – Universidad del Valle 2025.
