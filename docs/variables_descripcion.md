# Descripción Completa de Variables — Dataset de Features

**Proyecto:** Detección de Fraude en Establecimientos de Hospedaje  
**PLUS TI – Universidad del Valle 2025**  
**Integrantes:** Silvia Illescas · Davis Roldán · Michelle Mejía  
**Notebook fuente:** `notebooks/02_feature_engineering.ipynb`  
**Archivo:** `data/processed/features_dataset.csv` — 100,003 filas · 111 columnas

---

## Índice

1. [Variables originales del dataset](#1-variables-originales-del-dataset)
2. [Variables temporales del preprocesamiento](#2-variables-temporales-del-preprocesamiento)
3. [Features de hospedaje — Notebook 01](#3-features-de-hospedaje--notebook-01)
4. [Features históricas por cliente](#4-features-históricas-por-cliente)
5. [Features históricas por canal](#5-features-históricas-por-canal)
6. [Features históricas de hospedaje](#6-features-históricas-de-hospedaje)
7. [Features temporales y de velocidad](#7-features-temporales-y-de-velocidad)
8. [Features de ventana deslizante — Half 1](#8-features-de-ventana-deslizante--half-1)
9. [Features hotel y comportamiento avanzado — Half 2](#9-features-hotel-y-comportamiento-avanzado--half-2)
10. [Variables excluidas — Leakage](#10-variables-excluidas--leakage)
11. [Tabla resumen](#11-tabla-resumen)

---

> ### Principio fundamental: garantía anti look-ahead
>
> Todas las features históricas se calculan usando **exclusivamente las transacciones anteriores** a la actual.  
> Para la transacción número N de un cliente, los agregados usan solo las N-1 transacciones previas.  
> **Técnica vectorizada usada:**
> ```python
> # suma ANTES de la fila actual = cumsum_hasta_actual - valor_actual
> suma_antes = df.groupby('client_id')['amount_usd'].cumsum() - df['amount_usd']
> promedio_antes = suma_antes / df.groupby('client_id').cumcount().replace(0, np.nan)
> ```
> Esta formulación evita filtraciones de información futura hacia el modelo.

---

## 1. Variables originales del dataset

Variables provistas directamente en los archivos CSV de PlusTI. No requieren cálculo adicional.

### Variables de identificación (no usar en el modelo)

| Variable | Descripción |
|---|---|
| `transaction_id` | UUID único de la transacción. Solo para trazabilidad y debugging. |
| `bank_code` / `bank_name` / `bank_country` / `bank_tier` | Metadatos del banco emisor. `bank_tier` puede ser útil en el modelo federado (vip / private / state). |
| `client_id` | Identificador del cliente. Se usa para agrupar transacciones, no como feature directa. |
| `pan_masked` / `pan_hash` | Tarjeta enmascarada y su hash. No aportan información predictiva. |

### Campos ISO 8583 útiles para el modelo

| Variable | Descripción y uso |
|---|---|
| `DE18_merchant_category_code` | Código MCC (4 dígitos) que identifica el rubro del comercio. Variable clave: permite filtrar hoteles (7011), distinguir ATMs (6011), restaurantes (5812), etc. Alta variabilidad entre categorías de fraude. |
| `DE19_acquirer_country_code` | País del adquirente en código numérico ISO 3166-1. Indica dónde ocurrió físicamente la transacción. Usado para detectar cambios de país. |
| `DE22_pos_entry_mode` | Modo de captura del PAN. `81`=ecom (mayor fraude 8%), `51`=chip EMV, `71`=contactless, `10`=manual (alto riesgo), `22`=banda magnética (clonación). Feature muy relevante. |
| `DE25_pos_condition_code` | Condición del POS. `59`=ecom, `0`=presencial, `8`=MOTO. Complementa `DE22`. |
| `DE42_card_acceptor_id` | ID único del comercio (15 caracteres). Permite agrupar transacciones por establecimiento específico. Usado para `merchant_txn_count_before`. |
| `DE52_pin_data_present` | Flag `Y`/`N` de si se usó PIN. Transacciones sin PIN en POS presencial son más sospechosas. |
| `DE55_emv_data_present` | Flag de uso de chip EMV. Ausencia de chip en tarjeta con chip = posible clonación de banda. |
| `DE60_pos_terminal_type` | Tipo de terminal: `POS-ATTENDED`, `ECOM-VIRTUAL`, `ATM-UNATTENDED`. Complementa canal. |

### Variables de contexto banco/cliente

| Variable | Descripción y uso |
|---|---|
| `channel` | Canal de la transacción: `POS`, `ECOM`, `ATM`, `MOTO`. ECOM tiene la mayor tasa de fraude (8%). Feature muy relevante. |
| `amount_usd` | Monto normalizado en USD. Variable base de la mayoría de features de monto. Media en fraude: $1,101 vs $396 en legítimos. |
| `is_international` | `True` si el comercio está en un país diferente al del cliente. Tasa de fraude internacional: 8.5% vs 3.8% local. |
| `distance_from_home_km` | Distancia en km entre la ciudad del cliente y el comercio. Media en fraude: 2,895 km vs 1,721 km en legítimos. Feature importante para hospedaje. |
| `hour_local` | Hora local de la transacción (0–23). Fraudes tienen mayor concentración en horas de madrugada. |
| `day_of_week` | Día de la semana (Mon–Sun). Miércoles y sábado tienen tasas ligeramente mayores de fraude. |
| `client_baseline_amount` | Monto base de gasto habitual del cliente, provisto externamente. Punto de referencia para calcular `amount_vs_baseline` y `amount_baseline_ratio`. |
| `client_home_city` | Ciudad de residencia del cliente. Usada indirectamente para calcular distancia. |
| `is_fraud` | 🎯 **Variable objetivo (target)**. `True` si la transacción es fraudulenta. Distribución: ~5% fraude / 95% legítimo. Problema de clasificación binaria desbalanceada. |

### Variables excluidas — LEAKAGE

Las siguientes variables reflejan el resultado de la autorización y **no deben usarse como features**. Conocerlas antes de la decisión equivale a hacer trampa.

| Variable | Por qué es leakage |
|---|---|
| `approved` | Indica si la transacción fue aprobada — decisión posterior a la que queremos predecir. |
| `DE39_response_code` | Código de respuesta del emisor (00=aprobada, 05=declinada, etc.) — post-autorización. |
| `response_description` | Descripción textual del código de respuesta — post-autorización. |
| `DE38_authorization_code` | Código emitido solo cuando la transacción es aprobada — post-autorización. |

---

## 2. Variables temporales del preprocesamiento

Extraídas del campo `DE7_transmission_datetime` durante la fase de limpieza en `01_eda.ipynb`.

> **Corrección aplicada:** el campo original tiene 9 dígitos en formato `MMDDHHmmss`. El notebook usaba `format="%Y%m%d%H%M%S"` incorrectamente, generando 27,959 registros con fecha `NaT` y perdiendo el 28% de los datos en el split. Se corrigió con `zfill(10)` + `format="%m%d%H%M%S"`, recuperando todos los 100,003 registros.

| Variable | Cómo se calculó | Para qué sirve |
|---|---|---|
| `transaction_datetime` | `pd.to_datetime(DE7.str.zfill(10), format="%m%d%H%M%S")`. Año por defecto 1900 — las diferencias temporales son correctas. | Base para calcular `hours_since_last_txn` y todas las ventanas deslizantes. |
| `transaction_month` | `.dt.month` del campo anterior. Valores 1–6. | **Separación train/test**: meses 1–5 = entrenamiento (83,817 filas), mes 6 = test final (16,186 filas). |
| `transaction_day` | `.dt.day` | Análisis de patrones por día del mes. |
| `transaction_date` | `.dt.date` | Agrupaciones diarias. |

---

## 3. Features de hospedaje — Notebook 01

Creadas durante el EDA inicial para enfocar el análisis en el objetivo del proyecto.

---

### `is_hotel`
**Cómo se calculó:**
```python
HOTEL_MCCS = [7011]
df['is_hotel'] = df['DE18_merchant_category_code'].isin(HOTEL_MCCS)
```
**Confirmación con PlusTI:** se cruzaron los MCCs presentes en el dataset contra el rango estándar ISO 8583 de hospedaje (3501–3999 + 7011). El rango 3501–3999 (cadenas individuales como Hilton, Marriott con código propio) **no aparece en los datos**. Solo el MCC 7011 (hoteles y moteles general) está presente, confirmado validando los nombres reales de los comercios (`HOTEL HILTON`, `HYATT INN`, etc.). El flag original usaba el rango completo innecesariamente.

**Para qué sirve:** etiqueta las 9,064 transacciones de hospedaje (9.1% del dataset). Es la variable base para todas las features hotel-específicas. Tasa de fraude en hotel: **4.58%**.

---

### `amount_vs_baseline`
**Cómo se calculó:**
```python
df['amount_vs_baseline'] = df['amount_usd'] - df['client_baseline_amount']
```
**Para qué sirve:** mide cuánto se aleja el monto actual del gasto habitual del cliente (en términos absolutos). Un valor positivo grande indica un gasto atípicamente alto. Media en fraude: −$456 vs −$1,172 en legítimos — los fraudulentos se acercan más al baseline o lo superan.

---

### `amount_baseline_ratio`
**Cómo se calculó:**
```python
df['amount_baseline_ratio'] = df['amount_usd'] / df['client_baseline_amount']
```
**Para qué sirve:** versión relativa de `amount_vs_baseline`. Un ratio de 2.0 significa que el cliente gastó el doble de su línea base. Media en fraude: **0.70** vs 0.25 en legítimos — los fraudes representan una fracción mucho mayor del baseline. Es una de las variables con mayor correlación con `is_fraud` (0.28) en el dataset original.

---

### `hotel_international`
**Cómo se calculó:**
```python
df['hotel_international'] = df['is_hotel'] & df['is_international']
```
**Para qué sirve:** combinación de alto riesgo: hotel + internacional. Detecta el patrón clásico de fraude donde se usa una tarjeta robada en un hotel de otro país. Presencia en fraude: **4.86%** vs 2.03% en legítimos.

---

### `hotel_high_distance`
**Cómo se calculó:**
```python
q75 = df['distance_from_home_km'].quantile(0.75)  # ~483 km
df['hotel_high_distance'] = df['is_hotel'] & (df['distance_from_home_km'] > q75)
```
**Para qué sirve:** identifica hoteles muy lejos del hogar del cliente (percentil 75 de distancia). Complementa `hotel_international` para casos dentro del mismo país pero con gran desplazamiento. Presencia en fraude: **5.00%** vs 2.15% en legítimos.

---

## 4. Features históricas por cliente

Todas calculadas con transacciones **anteriores** al momento de la transacción actual.  
Los **NaN en primeras transacciones son estructurales y esperados** — LightGBM los maneja nativamente.

---

### `client_txn_count_before`
**Cómo se calculó:**
```python
df = df.sort_values(['client_id', 'transaction_datetime'])
df['client_txn_count_before'] = df.groupby('client_id').cumcount()
# cumcount() retorna 0 para la primera txn, 1 para la segunda, etc.
```
**Para qué sirve:** indica cuántas transacciones previas tiene el cliente en el dataset. `0` = primera transacción registrada. Sirve como denominador para calcular promedios históricos y detectar clientes nuevos (mayor riesgo). Media en fraude: **13.9** vs 12.5 en legítimos — los clientes con más historial tienden a tener más fraude acumulado.

---

### `client_avg_amount_before`
**Cómo se calculó:**
```python
cumsum = df.groupby('client_id')['amount_usd'].cumsum()
suma_antes = cumsum - df['amount_usd']   # excluye el valor actual
df['client_avg_amount_before'] = suma_antes / df['client_txn_count_before'].replace(0, np.nan)
```
**Para qué sirve:** promedio del monto de todas las transacciones anteriores del cliente. Es la referencia personal de gasto. Se usa como denominador en `amount_vs_hist_avg` y `amount_zscore_customer`. Media en fraude: **$471** vs $422 en legítimos — los clientes que cometen fraude tienen historial de gasto ligeramente mayor. NaN en la primera transacción de cada cliente (4,000 filas, 4%).

---

### `client_std_amount_before`
**Cómo se calculó:**
```python
df['client_std_amount_before'] = df.groupby('client_id')['amount_usd'].transform(
    lambda x: x.expanding().std().shift(1)
)
# expanding().std() dentro del transform opera sobre la serie del grupo → shift es relativo al grupo
```
**Para qué sirve:** desviación estándar histórica del monto del cliente. Clientes con alta variabilidad (std alto) son más difíciles de detectar por monto. Es el denominador del z-score. NaN en las primeras 2 transacciones de cada cliente (8,000 filas, 8%) porque std requiere al menos 2 valores.

---

### `client_max_amount_before`
**Cómo se calculó:**
```python
df['client_max_amount_before'] = df.groupby('client_id')['amount_usd'].transform(
    lambda x: x.expanding().max().shift(1)
)
```
**Para qué sirve:** el mayor monto que el cliente ha gastado alguna vez antes. Es el techo de referencia personal. Usado para calcular `amount_vs_max_ever_ratio`. Media en fraude: **$1,663** vs $1,485 en legítimos.

---

### `amount_vs_hist_avg`
**Cómo se calculó:**
```python
df['amount_vs_hist_avg'] = df['amount_usd'] / df['client_avg_amount_before'].replace(0, np.nan)
```
**Para qué sirve:** cuántas veces supera el monto actual al promedio histórico del cliente. Un valor de 3.0 significa que el cliente está gastando el triple de lo habitual. Media en fraude: **8.48** vs 1.39 en legítimos — los fraudes en promedio son 8 veces el gasto habitual. Mediana fraude: 2.05 vs 0.54 — incluso la transacción "típica" de fraude dobla la media histórica del cliente.

---

### `amount_over_hist_std` (alias: `amount_zscore_customer`)
**Cómo se calculó:**
```python
df['amount_over_hist_std'] = (
    (df['amount_usd'] - df['client_avg_amount_before'])
    / df['client_std_amount_before'].replace(0, np.nan)
)
```
**Para qué sirve:** z-score del monto actual respecto al historial personal del cliente. Normaliza la anomalía del monto en unidades de desviación estándar, comparable entre clientes con diferentes perfiles de gasto. Mediana en fraude: **+0.96σ** vs −0.41σ en legítimos. Valores por encima de +2σ son altamente sospechosos.

---

### `client_intl_rate_before`
**Cómo se calculó:**
```python
intl_int    = df['is_international'].astype(int)
cumsum_intl = intl_int.groupby(df['client_id']).cumsum()
df['client_intl_rate_before'] = (cumsum_intl - intl_int) / df['client_txn_count_before'].replace(0, np.nan)
```
**Para qué sirve:** proporción histórica de transacciones internacionales del cliente (0.0 a 1.0). Un cliente que nunca usó su tarjeta en el exterior y aparece en un hotel internacional es más sospechoso que uno que viaja frecuentemente. Media en fraude: **0.26** vs 0.23 en legítimos.

---

## 5. Features históricas por canal

Misma lógica anti look-ahead pero calculada dentro del mismo canal de transacción.

---

### `client_channel_txn_count_before`
**Cómo se calculó:**
```python
df['client_channel_txn_count_before'] = df.groupby(['client_id', 'channel']).cumcount()
```
**Para qué sirve:** número de veces que el cliente ha usado ese canal específico antes. Un cliente que normalmente usa POS y de repente transacciona en ECOM tiene `0` en ese canal. Media en fraude: **6.49** vs 4.72 en legítimos — los fraudes ocurren en canales que el cliente ya usó antes, no en canales nuevos.

---

### `client_channel_avg_amount_before`
**Cómo se calculó:**
```python
gc = df.groupby(['client_id', 'channel'])
cumsum_ch = gc['amount_usd'].cumsum()
df['client_channel_avg_amount_before'] = (
    (cumsum_ch - df['amount_usd'])
    / df['client_channel_txn_count_before'].replace(0, np.nan)
)
```
**Para qué sirve:** gasto promedio histórico del cliente **en ese canal específico**. Más preciso que el promedio general porque cada canal tiene montos típicos diferentes (ATM: ~$350, ECOM: ~$550, POS: ~$285). Media en fraude: **$531** vs $419 en legítimos. NaN al 14.4% (primera transacción por cliente-canal).

---

### `amount_vs_channel_avg`
**Cómo se calculó:**
```python
df['amount_vs_channel_avg'] = df['amount_usd'] / df['client_channel_avg_amount_before'].replace(0, np.nan)
```
**Para qué sirve:** ratio del monto actual vs el promedio histórico en ese canal. Más sensible que `amount_vs_hist_avg` porque compara en contexto del canal. Media en fraude: **17.53** vs 1.91 en legítimos — en fraude el monto supera 17 veces el promedio del canal.

---

### `client_channel_std_before`
**Cómo se calculó:**
```python
df['client_channel_std_before'] = df.groupby(['client_id', 'channel'])['amount_usd'].transform(
    lambda x: x.expanding().std().shift(1)
)
```
**Para qué sirve:** desviación estándar del monto por canal. Es el denominador de `amount_zscore_channel`. NaN al 26.6% (primeras 2 transacciones por cliente-canal).

---

## 6. Features históricas de hospedaje

Calculadas agrupando por `(client_id, is_hotel)`, capturando solo el comportamiento previo en hoteles.

---

### `client_hotel_txn_count_before`
**Cómo se calculó:**
```python
gh = df.groupby(['client_id', 'is_hotel'])
df['client_hotel_txn_count_before'] = gh.cumcount()
df.loc[~df['is_hotel'], 'client_hotel_txn_count_before'] = 0  # 0 fuera de hotel
```
**Para qué sirve:** cuántas veces el cliente ha estado en hoteles antes. `0` significa que nunca ha tenido una transacción en hotel — ese cliente es un "turista nuevo" en el sistema, señal de alerta cuando el cargo es grande. Media en fraude: **1.34** vs 1.11 en legítimos.

---

### `client_hotel_avg_amount_before`
**Cómo se calculó:**
```python
cumsum_hotel = gh['amount_usd'].cumsum()
df['client_hotel_avg_amount_before'] = np.where(
    df['is_hotel'] & (df['client_hotel_txn_count_before'] > 0),
    (cumsum_hotel - df['amount_usd']) / df['client_hotel_txn_count_before'],
    np.nan
)
```
**Para qué sirve:** cuánto gasta habitualmente el cliente en hoteles. Un cargo hotelero que triplica este valor es anómalo. Solo tiene valor para transacciones de hotel con historial previo. NaN al 94.5% — correcto, la mayoría de transacciones no son de hotel. Media en fraude: **$961** vs $932 en legítimos.

---

### `client_hotel_intl_rate_before`
**Cómo se calculó:**
```python
intl_int2         = df['is_international'].astype(int)
cumsum_h_intl     = intl_int2.groupby([df['client_id'], df['is_hotel']]).cumsum()
df['client_hotel_intl_rate_before'] = np.where(
    df['is_hotel'] & (df['client_hotel_txn_count_before'] > 0),
    (cumsum_h_intl - intl_int2) / df['client_hotel_txn_count_before'],
    np.nan
)
```
**Para qué sirve:** proporción histórica de hoteles internacionales del cliente. Si un cliente siempre se hospeda en hoteles locales y de repente aparece en un hotel en el exterior, es sospechoso. Media en fraude: **0.29** vs 0.24 en legítimos.

---

### `client_hotel_first_time`
**Cómo se calculó:**
```python
df['client_hotel_first_time'] = (
    df['is_hotel'] & (df['client_hotel_txn_count_before'] == 0)
).astype(int)
```
**Para qué sirve:** flag binario — `1` si es la primera vez que el cliente transacciona en un hotel. Un cargo hotelero grande de un cliente sin ningún historial hotelero es altamente sospechoso. Frecuencia en legítimos: **39.9%** de hoteles son primera vez; en fraude: **34.9%** — curiosamente los fraudes ocurren más en clientes con algo de historial hotelero previo.

---

## 7. Features temporales y de velocidad

---

### `hours_since_last_txn`
**Cómo se calculó:**
```python
# shift(1) en SeriesGroupBy opera per-group → correcto, no cruza clientes
df['prev_txn_datetime'] = df.groupby('client_id')['transaction_datetime'].shift(1)
df['hours_since_last_txn'] = (
    (df['transaction_datetime'] - df['prev_txn_datetime']).dt.total_seconds() / 3600
)
df['hours_since_last_txn'] = df['hours_since_last_txn'].fillna(-1)  # -1 = primera txn
```
**Para qué sirve:** mide el tiempo transcurrido desde la transacción anterior del mismo cliente. Es el indicador más potente del dataset para fraude: en el **fraude la mediana es 2 horas**, mientras que en legítimos es **112 horas** (casi 5 días). El fraude ocurre en ráfagas cortas porque el ladrón quiere usar la tarjeta antes de que sea bloqueada.

---

### `is_rapid_succession`
**Cómo se calculó:**
```python
df['is_rapid_succession'] = (
    (df['hours_since_last_txn'] >= 0) & (df['hours_since_last_txn'] < 1)
).astype(int)
```
**Para qué sirve:** flag binario de *card-testing* — transacciones en ráfaga a menos de 1 hora de diferencia. Es el predictor más fuerte del notebook: **41.2% de las transacciones fraudulentas** son sucesión rápida vs solo **0.62% de las legítimas**. Correlación con `is_fraud`: **0.55**.

---

### `time_since_last_txn_min`
**Cómo se calculó:**
```python
df['time_since_last_txn_min'] = df['hours_since_last_txn'] * 60
```
**Para qué sirve:** versión en minutos de `hours_since_last_txn`. La mayor granularidad permite al modelo capturar mejor el rango de 0–60 minutos donde ocurre la mayoría del fraude en sucesión. Mediana fraude: **120 minutos** vs **6,744 minutos** en legítimos — una diferencia de 56×.

---

## 8. Features de ventana deslizante — Half 1

Calculadas con búsqueda binaria (`bisect_left`) para eficiencia O(n log n). Solo cuentan transacciones **anteriores** dentro de la ventana.

---

### `txn_count_last_1h`
**Cómo se calculó:**
```python
def sliding_txn_count(group, td):
    times = group['transaction_datetime'].values.astype('int64')
    window_ns = int(td.total_seconds() * 1e9)
    counts = np.zeros(len(times), dtype=int)
    for i in range(1, len(times)):
        window_start = times[i] - window_ns
        left = bisect_left(times, window_start, 0, i)  # O(log n)
        counts[i] = i - left
    return pd.Series(counts, index=group.index)

df['txn_count_last_1h'] = df.groupby('client_id', group_keys=False).apply(
    lambda grp: sliding_txn_count(grp, pd.Timedelta(hours=1))
)
```
**Para qué sirve:** número de transacciones del cliente en la última hora antes de la actual. Detecta el patrón de *card-testing* donde se hacen múltiples cargos en minutos. Media en fraude: **6.67** vs **5.11** en legítimos. Es el segundo predictor más fuerte después de `is_rapid_succession` (correlación 0.12).

---

### `txn_count_last_24h`
**Cómo se calculó:** igual que `txn_count_last_1h` con `pd.Timedelta(hours=24)`.

**Para qué sirve:** velocidad diaria del cliente. Detecta campañas de fraude que duran varias horas pero dentro del mismo día. Media en fraude: **13.88** vs **12.50** en legítimos. Diferencia moderada porque el fraude extendido en 24h es menos común que el de 1h.

---

### `txn_count_last_7d`
**Cómo se calculó:** igual con `pd.Timedelta(days=7)`.

**Para qué sirve:** actividad semanal del cliente. En este dataset los valores son casi idénticos a `txn_count_last_24h`, lo que sugiere que la mayoría de las transacciones del mismo cliente en una semana ocurren el mismo día (coherente con el patrón de fraude en ráfaga). Útil como contexto del nivel de actividad habitual del cliente.

---

### `amount_zscore_customer`
**Cómo se calculó:**
```python
df['amount_zscore_customer'] = df['amount_over_hist_std']  # alias con nombre estándar
# (amount_usd - client_avg_amount_before) / client_std_amount_before
```
**Para qué sirve:** z-score estándar del monto del cliente respecto a su historial personal. Es la forma normalizada de detectar montos anómalos independientemente del perfil de gasto del cliente (un cliente VIP que gasta $5,000 no es anómalo; uno de bajo gasto sí lo es). Mediana fraude: **+0.96σ** vs −0.41σ en legítimos.

---

### `amount_zscore_channel`
**Cómo se calculó:**
```python
df['amount_zscore_channel'] = (
    (df['amount_usd'] - df['client_channel_avg_amount_before'])
    / df['client_channel_std_before'].replace(0, np.nan)
)
```
**Para qué sirve:** z-score del monto comparando contra el historial del cliente **en el mismo canal**. Más preciso que `amount_zscore_customer` porque los montos en ECOM son estructuralmente diferentes a los de ATM. Mediana fraude: **+0.92σ** vs −0.40σ en legítimos. NaN al 26.6% por primeras transacciones por canal.

---

### `unique_merchants_last_24h`
**Cómo se calculó:**
```python
def sliding_unique(group, col, td):
    times  = group['transaction_datetime'].values.astype('int64')
    values = group[col].values
    window_ns = int(td.total_seconds() * 1e9)
    counts = np.zeros(len(times), dtype=int)
    for i in range(1, len(times)):
        window_start = times[i] - window_ns
        mask = times[:i] >= window_start
        counts[i] = len(set(values[:i][mask]))
    return pd.Series(counts, index=group.index)

df['unique_merchants_last_24h'] = df.groupby('client_id', group_keys=False).apply(
    lambda grp: sliding_unique(grp, 'DE42_card_acceptor_id', pd.Timedelta(hours=24))
)
```
**Para qué sirve:** número de comercios distintos visitados en las últimas 24 horas. Un ladrón que obtiene una tarjeta la usa en múltiples establecimientos rápidamente (*enumeration fraud*). Media en fraude: **13.63** vs **12.28** en legítimos.

---

### `unique_countries_last_24h`
**Cómo se calculó:** igual que `unique_merchants_last_24h` con `'DE19_acquirer_country_code'`.

**Para qué sirve:** número de países distintos donde el cliente ha transaccionado en las últimas 24 horas. Detecta *impossible travel* — físicamente imposible estar en 3 países en un día. Media en fraude: **3.42** vs **3.04** en legítimos. La diferencia es moderada porque el dataset sintético puede no capturar bien este patrón.

---

### `is_night_transaction`
**Cómo se calculó:**
```python
df['is_night_transaction'] = df['hour_local'].between(0, 4).astype(int)
```
**Para qué sirve:** flag binario — `1` si la hora local está entre las 00:00 y las 04:59. El fraude en hoteles ocurre más frecuentemente de madrugada (check-ins tardíos fraudulentos, cargos de habitación post-evento). En hospedaje: **14.0%** de transacciones fraudulentas son nocturnas vs **8.2%** de las legítimas. A nivel general: **10.8%** vs **8.4%**.

---

### `amount_vs_max_ever_ratio`
**Cómo se calculó:**
```python
df['amount_vs_max_ever_ratio'] = df['amount_usd'] / df['client_max_amount_before'].replace(0, np.nan)
```
**Para qué sirve:** ratio del monto actual vs el máximo histórico del cliente. Un valor > 1.0 significa que el cliente **nunca antes gastó tanto** — es una señal de alerta fuerte. El **29.1% de las transacciones fraudulentas** superan el máximo histórico del cliente, vs solo el **10.7% de las legítimas**. Media en fraude: **6.74** (casi 7 veces el máximo histórico en promedio, debido a outliers extremos).

---

## 9. Features hotel y comportamiento avanzado — Half 2

---

### `hotel_amount_zscore`
**Cómo se calculó:**
```python
# Std histórica de monto en hotel
gh = df.groupby(['client_id', 'is_hotel'])
df['client_hotel_std_before'] = gh['amount_usd'].transform(
    lambda x: x.expanding().std().shift(1)
)
df['hotel_amount_zscore'] = np.where(
    df['is_hotel'] & df['client_hotel_avg_amount_before'].notna(),
    (df['amount_usd'] - df['client_hotel_avg_amount_before']) / df['client_hotel_std_before'].replace(0, np.nan),
    np.nan
)
```
**Para qué sirve:** z-score del monto **dentro del contexto hotelero del cliente**. Más preciso que `amount_zscore_customer` para detectar cargos hoteleros anómalos, porque compara el cargo con el historial específico de hoteles del cliente (no con sus compras en supermercados o gasolineras). Mediana en fraude: **+0.88σ** vs −0.12σ en legítimos. NaN al 97.2% (solo aplica a hoteles con historial previo de hotel).

---

### `days_since_last_hotel_txn`
**Cómo se calculó:**
```python
df['_prev_hotel_dt'] = gh['transaction_datetime'].shift(1)  # shift por grupo (is_hotel=True)
df['days_since_last_hotel_txn'] = np.where(
    df['is_hotel'] & df['_prev_hotel_dt'].notna(),
    (df['transaction_datetime'] - df['_prev_hotel_dt']).dt.total_seconds() / 86400,
    -1  # -1 = primera txn en hotel o no es hotel
)
```
**Para qué sirve:** días transcurridos desde la última transacción del cliente en un hotel. Un fraude que carga dos hoteles en pocas horas es altamente sospechoso. Media en fraude: **0.83 días** vs 1.31 días en legítimos — los fraudes regresan al hotel más rápido. Solo tiene sentido para hoteles con historial previo (`-1` en los demás).

---

### `hotel_new_country`
**Cómo se calculó:**
```python
# Para cada txn de hotel, verificar si el país (DE19) ya apareció antes en hoteles del mismo cliente
seen = {}
for _, row in hotel_sub.iterrows():
    cid, country = row['client_id'], row['DE19_acquirer_country_code']
    seen.setdefault(cid, set())
    flag = 1 if country not in seen[cid] else 0
    seen[cid].add(country)
df.loc[hotel_idx, 'hotel_new_country'] = flags
```
**Para qué sirve:** `1` si el hotel está en un país que el cliente nunca había visitado antes (en hoteles). Detecta el patrón de tarjeta robada usada en destino turístico desconocido para el cliente. Frecuencia en fraude: **72.3%** vs **58.9%** en legítimos — el fraude ocurre más frecuentemente en países hoteleros nuevos para el cliente.

---

### `merchant_txn_count_before`
**Cómo se calculó:**
```python
df['merchant_txn_count_before'] = df.groupby(['client_id', 'DE42_card_acceptor_id']).cumcount()
```
**Para qué sirve:** cuántas veces el cliente ha usado **este comercio específico** antes. `0` significa que es un comercio completamente nuevo para el cliente. El **97.4% de todas las transacciones** (tanto fraude como legítimas) son en comercios nuevos para el cliente, lo que limita el poder discriminatorio de esta variable. Sin embargo, puede ser útil en combinación con otras variables.

---

### `rapid_country_change`
**Cómo se calculó:**
```python
df['_prev_country'] = df.groupby('client_id')['DE19_acquirer_country_code'].shift(1)
df['rapid_country_change'] = (
    (df['DE19_acquirer_country_code'] != df['_prev_country'])
    & (df['time_since_last_txn_min'] >= 0)    # no es la primera txn
    & (df['time_since_last_txn_min'] < 1440)  # menos de 24 horas
).astype(int)
```
**Para qué sirve:** detecta *impossible travel* — cambio de país en menos de 24 horas. Es la **variable más potente de Half 2** con correlación **0.29** con `is_fraud`. Frecuencia en fraude: **37.6%** vs **4.9%** en legítimos. En hospedaje específicamente: **48.2%** en fraude vs **5.2%** en legítimos — casi la mitad de los fraudes hoteleros involucran un cambio de país rápido previo.

---

### `amount_round_flag`
**Cómo se calculó:**
```python
df['amount_round_flag'] = (
    (df['amount_usd'] % 50 < 0.01) | (df['amount_usd'] % 50 > 49.99)
).astype(int)
```
**Para qué sirve:** flag para montos múltiplos exactos de $50 o $100 USD, patrón común en *card-testing* (un ladrón hace cargos redondos para verificar que la tarjeta funciona). **Variable débil en este dataset**: frecuencia casi idéntica en fraude (0.02%) y legítimo (0.05%). El dataset sintético de PlusTI no genera este patrón, pero es estándar en detección de fraude real.

---

### `client_weekend_rate_before`
**Cómo se calculó:**
```python
weekend_int = df['day_of_week'].isin(['Sat', 'Sun']).astype(int)
cumsum_wk   = weekend_int.groupby(df['client_id']).cumsum()
df['client_weekend_rate_before'] = (
    (cumsum_wk - weekend_int) / df['client_txn_count_before'].replace(0, np.nan)
)
```
**Para qué sirve:** proporción histórica de compras del cliente en fin de semana. Un cliente que nunca compra en fin de semana y aparece con un cargo hotelero un sábado es inusual. **Variable débil en este dataset**: media fraude **27.0%** vs legítimo **27.6%** — prácticamente sin diferencia. El patrón de fraude en este dataset no depende del día de la semana.

---

### `hotel_distance_zscore`
**Cómo se calculó:**
```python
# Media histórica de distancia en hoteles
cumsum_dist  = gh['distance_from_home_km'].cumsum()
df['client_hotel_avg_dist_before'] = np.where(
    df['is_hotel'] & (df['client_hotel_txn_count_before'] > 0),
    (cumsum_dist - df['distance_from_home_km']) / df['client_hotel_txn_count_before'],
    np.nan
)
# Std histórica de distancia en hoteles
df['client_hotel_std_dist_before'] = gh['distance_from_home_km'].transform(
    lambda x: x.expanding().std().shift(1)
)
df['hotel_distance_zscore'] = np.where(
    df['is_hotel'] & df['client_hotel_avg_dist_before'].notna(),
    (df['distance_from_home_km'] - df['client_hotel_avg_dist_before']) / df['client_hotel_std_dist_before'].replace(0, np.nan),
    np.nan
)
```
**Para qué sirve:** z-score de la distancia del hotel vs el historial de distancias hoteleras del cliente. Si el cliente normalmente se hospeda a 50 km de casa y este hotel está a 10,000 km, la anomalía geográfica se captura en este z-score. Mediana en fraude: **+0.94σ** vs −0.56σ en legítimos — los fraudes ocurren en hoteles más lejanos de lo habitual para ese cliente. NaN al 97.2% (solo aplica a hoteles con historial).

---

### `hour_deviation_from_usual`
**Cómo se calculó:**
```python
cumsum_hour = df['hour_local'].groupby(df['client_id']).cumsum()
df['client_avg_hour_before'] = (
    (cumsum_hour - df['hour_local']) / df['client_txn_count_before'].replace(0, np.nan)
)
df['hour_deviation_from_usual'] = (df['hour_local'] - df['client_avg_hour_before']).abs()
```
**Para qué sirve:** cuántas horas se aleja la hora actual de la hora media histórica del cliente. Captura el patrón de transacciones en horas inusuales para ese cliente específico. Media en fraude: **4.77 horas** vs **5.19 horas** en legítimos — diferencia pequeña, variable de señal moderada. En hospedaje específicamente los fraudes tienen mayor desviación (5.47 vs 5.23).

---

### `client_mcc_diversity_before`
**Cómo se calculó:**
```python
df['client_mcc_diversity_before'] = df.groupby('client_id')['DE18_merchant_category_code'].transform(
    lambda x: x.expanding().apply(
        lambda s: s.iloc[:-1].nunique() if len(s) > 1 else 0, raw=False
    )
)
```
**Para qué sirve:** número de categorías de comercio (MCCs) distintas que el cliente ha visitado históricamente. Mide la diversidad del perfil de gasto: un cliente que solo compra en 2 tipos de comercio y aparece en un hotel es más anómalo que uno con historial diverso. Media en fraude: **8.55 MCCs** vs **7.94 MCCs** en legítimos — los clientes que cometen fraude tienen perfiles de gasto ligeramente más diversos. Correlación con `is_fraud`: **0.032**.

---

## 10. Variables excluidas — Leakage

Estas variables capturan información que **solo existe después de que la transacción ya fue procesada**. Usarlas como features introduciría información del futuro en el modelo, inflando artificialmente las métricas y haciendo el modelo inútil en producción.

| Variable | Por qué es leakage | Qué revela |
|---|---|---|
| `approved` | La aprobación es consecuencia de la decisión de autorización — conocerla antes de decidir es trampa. | Si la transacción fue aprobada o declinada. |
| `DE39_response_code` | Código emitido por el emisor como **respuesta** a la solicitud. `00`=aprobada, `05`=declinada. | El resultado de la autorización. |
| `response_description` | Texto descriptivo del `DE39_response_code`. | Igual que el anterior, en forma legible. |
| `DE38_authorization_code` | Código de 6 caracteres emitido **solo** cuando la transacción es aprobada. Su ausencia ya revela que fue declinada. | Si fue aprobada y el código único de autorización. |

> **Nota:** estas variables sí pueden usarse en el EDA exploratorio para entender los datos, pero deben excluirse completamente de los sets de features del modelo.

---

## 11. Tabla resumen

### Features generadas — ordenadas por correlación absoluta con `is_fraud`

| # | Variable | Correlación |is_fraud| Nulos | Origen |
|---|---|---|---|---|
| 1 | `is_rapid_succession` | 0.5493 | 0% | Temporal |
| 2 | `rapid_country_change` | 0.2867 | 0% | Half 2 |
| 3 | `hours_since_last_txn` | 0.1284 | 0% | Temporal |
| 4 | `time_since_last_txn_min` | 0.1284 | 0% | Temporal |
| 5 | `txn_count_last_1h` | 0.1169 | 0% | Half 1 |
| 6 | `client_channel_txn_count_before` | 0.0939 | 0% | Canal |
| 7 | `amount_vs_channel_avg` | 0.0667 | 14% | Canal |
| 8 | `client_channel_avg_amount_before` | 0.0680 | 14% | Canal |
| 9 | `amount_vs_hist_avg` | 0.0446 | 4% | Cliente |
| 10 | `client_avg_amount_before` | 0.0398 | 4% | Cliente |
| 11 | `client_max_amount_before` | 0.0396 | 4% | Cliente |
| 12 | `client_txn_count_before` | 0.0366 | 0% | Cliente |
| 13 | `unique_merchants_last_24h` | 0.0365 | 0% | Half 1 |
| 14 | `txn_count_last_24h` | 0.0366 | 0% | Half 1 |
| 15 | `txn_count_last_7d` | 0.0366 | 0% | Half 1 |
| 16 | `client_std_amount_before` | 0.0355 | 8% | Cliente |
| 17 | `client_intl_rate_before` | 0.0351 | 4% | Cliente |
| 18 | `client_mcc_diversity_before` | 0.0319 | 0% | Half 2 |
| 19 | `client_hotel_intl_rate_before` | 0.0316 | 94.5% | Hotel |
| 20 | `unique_countries_last_24h` | 0.0493 | 0% | Half 1 |
| 21 | `amount_vs_max_ever_ratio` | 0.0383 | 4% | Half 1 |
| 22 | `hour_deviation_from_usual` | 0.0249 | 4% | Half 2 |
| 23 | `is_night_transaction` | 0.0183 | 0% | Half 1 |
| 24 | `hotel_amount_zscore` | 0.0176 | 97.2% | Half 2 |
| 25 | `hotel_international` | — | 0% | EDA |
| 26 | `hotel_high_distance` | — | 0% | EDA |
| 27 | `amount_zscore_customer` | 0.0020 | 8% | Half 1 |
| 28 | `amount_zscore_channel` | 0.0017 | 27% | Half 1 |
| 29 | `hotel_distance_zscore` | 0.0097 | 97.2% | Half 2 |
| 30 | `days_since_last_hotel_txn` | 0.0082 | 0% | Half 2 |
| 31 | `client_weekend_rate_before` | 0.0077 | 4% | Half 2 |
| 32 | `hotel_new_country` | 0.0071 | 0% | Half 2 |
| 33 | `amount_round_flag` | 0.0028 | 0% | Half 2 |
| 34 | `merchant_txn_count_before` | 0.0023 | 0% | Half 2 |
| 35 | `client_hotel_first_time` | 0.0080 | 0% | Hotel |
| 36 | `client_hotel_txn_count_before` | 0.0054 | 0% | Hotel |

### Conteo de columnas por categoría

| Categoría | Columnas |
|---|---|
| Identificadores (excluir del modelo) | 8 |
| Campos ISO 8583 originales | 35 |
| Variables de contexto banco/cliente | 17 |
| Variables temporales preprocesamiento | 4 |
| Features de hospedaje — EDA | 5 |
| Features históricas por cliente | 7 |
| Features históricas por canal | 4 |
| Features históricas de hospedaje | 4 |
| Features temporales y velocidad | 3 |
| Features ventana deslizante — Half 1 | 9 |
| Features hotel y comportamiento — Half 2 | 14 |
| **TOTAL** | **111** |

---

> **Variables recomendadas para el modelo base** (mayor poder discriminatorio + 0% nulos):  
> `is_rapid_succession`, `rapid_country_change`, `hours_since_last_txn`, `time_since_last_txn_min`,  
> `txn_count_last_1h`, `client_channel_txn_count_before`, `is_hotel`, `amount_vs_hist_avg`,  
> `client_txn_count_before`, `unique_merchants_last_24h`, `is_night_transaction`,  
> `hotel_new_country`, `client_mcc_diversity_before`, `client_hotel_first_time`

---

*Generado a partir de `data/processed/features_dataset.csv` — Notebook `02_feature_engineering.ipynb`*
