# Descripción de Variables — Dataset de Features

**Proyecto:** Detección de Fraude en Establecimientos de Hospedaje  
**PLUS TI – Universidad del Valle 2025**  
**Integrantes:** Silvia Illescas · Davis Roldán · Michelle Mejía

> **Archivo fuente:** `data/processed/features_dataset.csv`  
> **Total de columnas:** 97  
> **Filas:** 100,003 transacciones (banco BO-VIP)

---

## Índice de categorías

1. [Identificadores](#1-identificadores--no-usar-en-el-modelo)
2. [Campos ISO 8583 originales](#2-campos-iso-8583-originales)
3. [Variables de contexto banco/cliente](#3-variables-de-contexto-bancocliente)
4. [Variables temporales del preprocesamiento](#4-variables-temporales-del-preprocesamiento)
5. [Features de hospedaje — EDA](#5-features-de-hospedaje--notebook-01-eda)
6. [Features históricas por cliente](#6-features-históricas-por-cliente--notebook-02)
7. [Features históricas por canal](#7-features-históricas-por-canal--notebook-02)
8. [Features históricas de hospedaje](#8-features-históricas-de-hospedaje--notebook-02)
9. [Features temporales / velocidad](#9-features-temporales--velocidad--notebook-02)
10. [Features de ventana deslizante — Half 1](#10-features-de-ventana-deslizante--half-1-notebook-02)
11. [Half 2 — Pendiente compañera](#11-half-2--pendiente-compañera)
12. [Variables excluidas del modelo](#12-variables-excluidas-del-modelo-leakage)

---

## 1. Identificadores — NO usar en el modelo

| # | Variable | Descripción |
|---|---|---|
| 1 | `transaction_id` | UUID único de la transacción. Solo para trazabilidad. |
| 2 | `bank_code` | Código del banco emisor (ej. `BO-VIP`). |
| 3 | `bank_name` | Nombre del banco. |
| 4 | `bank_country` | País del banco emisor (ISO alfa-2). |
| 5 | `bank_tier` | Nivel del banco: `vip`, `private`, `state`. |
| 6 | `client_id` | ID interno del cliente. Usado para agrupar, no como feature directa. |
| 10 | `pan_masked` | Número de tarjeta parcialmente enmascarado. |
| 11 | `pan_hash` | Hash SHA del PAN completo. |

---

## 2. Campos ISO 8583 originales

Campos crudos del estándar financiero ISO 8583. Prefijo `DE` + número de Data Element.

| # | Variable | Descripción | Uso en modelo |
|---|---|---|---|
| 13 | `DE2_PAN` | Número de tarjeta en formato numérico. | ❌ No |
| 14 | `DE3_processing_code` | Tipo de operación (000000=compra, 010000=retiro, 200000=devolución). | ⚠️ Opcional |
| 15 | `DE4_amount_transaction` | Monto en moneda local sin decimales. | ❌ Usar `amount_usd` |
| 16 | `DE6_amount_cardholder_billing` | Monto en moneda de facturación del tarjetahabiente. | ❌ Usar `amount_usd` |
| 17 | `DE7_transmission_datetime` | Fecha/hora raw en formato `MMDDHHmmss` (9 dígitos). | ❌ Usar `transaction_datetime` |
| 18 | `DE9_conversion_rate_billing` | Tasa de cambio aplicada a la facturación. | ⚠️ Opcional |
| 19 | `DE11_STAN` | System Trace Audit Number — número de traza único (6 dígitos). | ❌ No |
| 20 | `DE12_local_time` | Hora local en formato `HHmmss`. | ❌ Usar `hour_local` |
| 21 | `DE13_local_date` | Fecha local en formato `MMDD`. | ❌ Usar `transaction_month` |
| 22 | `DE14_expiration_date` | Fecha de vencimiento de la tarjeta (`YYMM`). | ⚠️ Opcional |
| 23 | `DE15_settlement_date` | Fecha de liquidación de la transacción. | ❌ No |
| 24 | `DE18_merchant_category_code` | **MCC** — Código de categoría del comercio (4 dígitos). Variable clave. | ✅ Sí |
| 25 | `DE19_acquirer_country_code` | País del adquirente en código numérico ISO 3166-1. | ✅ Sí |
| 26 | `DE22_pos_entry_mode` | **Modo de captura del PAN:** `81`=ecom, `51`=chip, `71`=contactless, `10`=manual, `22`=banda. | ✅ Sí |
| 27 | `DE23_card_seq_number` | Número de secuencia de la tarjeta (reemisiones). | ⚠️ Opcional |
| 28 | `DE25_pos_condition_code` | **Condición del POS:** `0`=presencial, `59`=ecom, `8`=MOTO, `1`=CNP recurrente. | ✅ Sí |
| 29 | `DE32_acquiring_institution_id` | BIN del banco adquirente. | ❌ No |
| 30 | `DE35_track2_data_masked` | Datos de pista 2 enmascarados. Sin valor predictivo. | ❌ No |
| 31 | `DE37_retrieval_reference_number` | Número de referencia para disputas/conciliación (12 chars). | ❌ No |
| 33 | `DE39_response_code` | ⚠️ **LEAKAGE** — Código de respuesta del emisor post-autorización. | ❌ Excluir |
| 34 | `DE41_terminal_id` | ID del terminal POS o ATM. | ⚠️ Opcional |
| 35 | `DE42_card_acceptor_id` | **ID único del comercio** (15 chars). Útil para conteos por comercio. | ✅ Sí |
| 36 | `DE43_card_acceptor_name_location` | Nombre, ciudad y país del comercio (40 chars). | ⚠️ Requiere parsing |
| 37 | `DE49_currency_code_transaction` | Código numérico de moneda de la transacción (840=USD, 978=EUR). | ⚠️ Opcional |
| 38 | `DE50_currency_code_settlement` | Código de moneda de liquidación. | ❌ No |
| 39 | `DE51_currency_code_billing` | Código de moneda de facturación al cliente. | ❌ No |
| 40 | `DE52_pin_data_present` | Flag de uso de PIN: `Y`/`N`. | ✅ Sí |
| 41 | `DE55_emv_data_present` | Flag de uso de chip EMV: `Y`/`N`. | ✅ Sí |
| 42 | `DE58_authorizing_agent_id` | ID del agente autorizador. | ❌ No |
| 43 | `DE60_pos_terminal_type` | Tipo de terminal: `POS-ATTENDED`, `ECOM-VIRTUAL`, `ATM-UNATTENDED`. | ✅ Sí |
| 44 | `DE61_pos_extended_data` | Datos extendidos del POS (capacidades). | ⚠️ Opcional |
| 45 | `DE63_network_specific` | Red del banco (ej. `BO-VIP`). | ❌ No |
| 46 | `DE100_receiving_institution_id` | ID de la institución receptora. | ❌ No |
| 47 | `DE102_account_id_1` | Identificación de la cuenta origen enmascarada. | ❌ No |
| 48 | `DE123_pos_data_code` | Código de capacidades del POS (chip, contactless, titular presente). | ⚠️ Opcional |

---

## 3. Variables de contexto banco/cliente

Variables adicionales que acompañan el mensaje ISO 8583 en el dataset.

| # | Variable | Descripción | Uso en modelo |
|---|---|---|---|
| 8 | `channel` | Canal de la transacción: `POS`, `ECOM`, `ATM`, `MOTO`. | ✅ Sí |
| 9 | `card_brand` | Marca de la tarjeta: `VISA`, `MASTERCARD`, etc. | ⚠️ Opcional |
| 7 | `client_segment` | Segmento del cliente: `PLATINUM`, `BLACK`, `PRIVATE`, `INFINITE`. | ⚠️ Opcional |
| 49 | `amount_local` | Monto en moneda local del cliente. | ❌ Usar `amount_usd` |
| 50 | `amount_tx_currency` | Monto en la moneda de la transacción. | ❌ Usar `amount_usd` |
| 51 | `currency_tx_alpha` | Código alfabético de moneda (USD, EUR, BOB…). | ⚠️ Opcional |
| 52 | `amount_usd` | **Monto normalizado en USD.** Variable principal de monto. | ✅ Sí |
| 53 | `is_international` | `True` si el comercio está en un país diferente al del cliente. | ✅ Sí |
| 54 | `distance_from_home_km` | Distancia en km entre la ciudad del cliente y el comercio. | ✅ Sí |
| 55 | `hour_local` | Hora local de la transacción (0–23). | ✅ Sí |
| 56 | `day_of_week` | Día de la semana: `Mon`, `Tue`, …, `Sun`. | ✅ Sí |
| 57 | `approved` | ⚠️ **LEAKAGE** — Booleano: si la transacción fue aprobada. | ❌ Excluir |
| 58 | `response_description` | ⚠️ **LEAKAGE** — Descripción textual de la respuesta del emisor. | ❌ Excluir |
| 32 | `DE38_authorization_code` | ⚠️ **LEAKAGE** — Código de autorización emitido post-aprobación. | ❌ Excluir |
| 59 | `client_baseline_amount` | Monto base de gasto habitual del cliente (fuente externa al dataset). | ✅ Sí |
| 60 | `client_home_city` | Ciudad de residencia del cliente. | ⚠️ Opcional |
| **61** | **`is_fraud`** | 🎯 **TARGET** — `True` si la transacción es fraudulenta (~5% del total). | — |

---

## 4. Variables temporales del preprocesamiento

Extraídas del campo `DE7_transmission_datetime` con el formato correcto `%m%d%H%M%S`.

> **Corrección aplicada:** el campo original tiene 9 dígitos. Se aplica `zfill(10)` antes del parseo para evitar pérdida del 28% de registros que ocurría con el formato `%Y%m%d%H%M%S` incorrecto.

| # | Variable | Descripción |
|---|---|---|
| 62 | `transaction_datetime` | Datetime parseado (`1900-MM-DD HH:mm:ss`). Año 1900 por defecto; diferencias temporales son correctas. |
| 63 | `transaction_month` | Mes de la transacción (1–6). Mes 6 = **set de test final**. |
| 64 | `transaction_day` | Día del mes (1–31). |
| 65 | `transaction_date` | Fecha sin hora (`date` object). |

---

## 5. Features de hospedaje — Notebook 01 EDA

| # | Variable | Descripción | Nulos |
|---|---|---|---|
| 66 | `is_hotel` | `True` si `DE18_merchant_category_code == 7011`. **Confirmado con PlusTI:** el rango ISO 3501–3999 no aparece en el dataset; el único MCC de hospedaje es el 7011. | 0% |
| 67 | `amount_vs_baseline` | `amount_usd − client_baseline_amount`. Diferencia absoluta respecto al gasto habitual del cliente. | 0% |
| 68 | `amount_baseline_ratio` | `amount_usd / client_baseline_amount`. Ratio del monto actual vs línea base. | 0% |
| 69 | `hotel_international` | `True` si es transacción en hotel **y** es internacional. Combinación de alto riesgo. | 0% |
| 70 | `hotel_high_distance` | `True` si es hotel **y** `distance_from_home_km > percentil 75`. | 0% |

---

## 6. Features históricas por cliente — Notebook 02

> **Garantía anti look-ahead:** calculadas con `cumsum − valor_actual` (vectorizado) o `expanding().shift(1)` dentro del `transform` por grupo. Los NaN en primeras transacciones son **estructurales y correctos**.

| # | Variable | Descripción | Nulos |
|---|---|---|---|
| 71 | `client_txn_count_before` | Número de transacciones previas del mismo cliente. `0` = primera transacción. | 0% |
| 72 | `client_avg_amount_before` | Promedio de `amount_usd` de todas las transacciones anteriores del cliente. | 4% |
| 73 | `client_std_amount_before` | Desviación estándar histórica del monto del cliente. NaN en las primeras 2 txns. | 8% |
| 74 | `client_max_amount_before` | Monto máximo registrado anteriormente para este cliente. | 4% |
| 75 | `amount_vs_hist_avg` | `amount_usd / client_avg_amount_before`. Ratio del monto actual vs promedio histórico del cliente. | 4% |
| 76 | `amount_over_hist_std` | Z-score: `(amount_usd − client_avg_amount_before) / client_std_amount_before`. Alias: `amount_zscore_customer`. | 8% |
| 77 | `client_intl_rate_before` | Proporción histórica de transacciones internacionales del cliente (0.0–1.0). | 4% |

---

## 7. Features históricas por canal — Notebook 02

> Misma lógica anti look-ahead pero agrupando por `(client_id, channel)`.

| # | Variable | Descripción | Nulos |
|---|---|---|---|
| 78 | `client_channel_txn_count_before` | Nº de txns previas del cliente **en el mismo canal** (POS/ECOM/ATM/MOTO). | 0% |
| 79 | `client_channel_avg_amount_before` | Promedio histórico del monto del cliente en ese canal. | 14% |
| 80 | `amount_vs_channel_avg` | `amount_usd / client_channel_avg_amount_before`. Ratio vs promedio del canal. | 14% |
| 92 | `client_channel_std_before` | Desviación estándar histórica del monto en ese canal. Insumo para `amount_zscore_channel`. | 27% |

---

## 8. Features históricas de hospedaje — Notebook 02

> Agrupando por `(client_id, is_hotel)`. Los NaN al 94.5% son **esperados**: la mayoría de transacciones no son en hotel.

| # | Variable | Descripción | Nulos |
|---|---|---|---|
| 81 | `client_hotel_txn_count_before` | Nº de txns previas del cliente en comercios de hospedaje (MCC 7011). `0` si nunca fue a un hotel. | 0% |
| 82 | `client_hotel_avg_amount_before` | Promedio histórico de monto del cliente en hotels. NaN si no hay historial hotelero. | 94.5% |
| 83 | `client_hotel_intl_rate_before` | Proporción histórica de hotels internacionales del cliente. NaN si sin historial. | 94.5% |
| 84 | `client_hotel_first_time` | `1` si es la primera vez que el cliente transacciona en un hotel. Señal de alerta. | 0% |

---

## 9. Features temporales / velocidad — Notebook 02

| # | Variable | Descripción | Fraude vs Legítimo |
|---|---|---|---|
| 85 | `hours_since_last_txn` | Horas transcurridas desde la transacción anterior del mismo cliente. `−1` si es la primera txn. | Fraude: mediana **2.1h** / Legítimo: mediana 119h |
| 86 | `is_rapid_succession` | `1` si `hours_since_last_txn < 1`. Flag de *card-testing* o uso intensivo. | Fraude: **41%** / Legítimo: 0.6% |
| 87 | `time_since_last_txn_min` | Ídem `hours_since_last_txn` expresado en **minutos**. Mayor granularidad para el modelo. | Fraude: mediana **126 min** / Legítimo: mediana 7,157 min |

---

## 10. Features de ventana deslizante — Half 1, Notebook 02

> Calculadas con ventana temporal deslizante hacia atrás (solo transacciones anteriores a la actual). Implementadas con `bisect_left` sobre tiempos ordenados → **O(n log n)**.

| # | Variable | Descripción | Fraude vs Legítimo |
|---|---|---|---|
| 88 | `txn_count_last_1h` | Nº de transacciones del cliente en la **última hora** antes de la actual. | Fraude: 6.7 / Legítimo: 5.1 |
| 89 | `txn_count_last_24h` | Nº de transacciones del cliente en las **últimas 24 horas**. | Fraude: 13.9 / Legítimo: 12.5 |
| 90 | `txn_count_last_7d` | Nº de transacciones del cliente en los **últimos 7 días**. | Fraude: 13.9 / Legítimo: 12.5 |
| 91 | `amount_zscore_customer` | Z-score del monto respecto al historial personal del cliente. Alias de `amount_over_hist_std` con nombre estándar del enunciado. | Fraude: mediana +0.96σ / Legítimo: −0.41σ |
| 93 | `amount_zscore_channel` | Z-score del monto respecto al historial del cliente **en el mismo canal**. Más preciso que el z-score global. | Fraude: mediana +0.92σ / Legítimo: −0.40σ |
| 94 | `unique_merchants_last_24h` | Nº de comercios distintos visitados por el cliente en las últimas 24h. Patrón de *enumeration fraud*. | Fraude: 13.6 / Legítimo: 12.3 |
| 95 | `unique_countries_last_24h` | Nº de países distintos en las últimas 24h. Detecta *impossible travel*. | Fraude: 3.4 / Legítimo: 3.0 |
| 96 | `is_night_transaction` | `1` si `hour_local` está entre 00:00 y 04:59. Más relevante en hospedaje (check-ins tardíos). | Hotel fraude: **14%** / Hotel legítimo: 8.2% |
| 97 | `amount_vs_max_ever_ratio` | `amount_usd / client_max_amount_before`. Ratio > 1 significa que el monto supera el máximo histórico del cliente. | Fraude: **29%** supera máx. / Legítimo: 10.7% |

---

## 11. Half 2 — Pendiente (compañera)

Partir del archivo `data/processed/features_dataset.csv` y agregar al final del notebook `02_feature_engineering.ipynb`.

| # | Variable | Descripción | Tipo |
|---|---|---|---|
| 98 | `hotel_amount_zscore` | Z-score del monto actual en hotel vs historial hotelero del cliente: `(amount_usd − client_hotel_avg_before) / hotel_std_before`. | 🏨 Hotel |
| 99 | `days_since_last_hotel_txn` | Días desde la última transacción del cliente en hotel. `−1` si nunca fue. | 🏨 Hotel |
| 100 | `hotel_new_country` | `1` si el `DE19_acquirer_country_code` de este hotel nunca apareció en txns hoteleras anteriores del cliente. | 🏨 Hotel 🟢 Innovadora |
| 101 | `merchant_txn_count_before` | Nº de veces que el cliente usó este comercio específico (`DE42_card_acceptor_id`) antes. `0` = comercio nuevo. | 🏪 Merchant |
| 102 | `rapid_country_change` | `1` si el país de esta txn difiere del de la txn anterior **y** `time_since_last_txn_min < 1440` (24h). *Impossible travel*. | ✈️ Viaje 🟢 Innovadora |
| 103 | `amount_round_flag` | `1` si el monto es múltiplo exacto de 50 o 100 USD. Patrón de *card-testing* (prueba con montos redondos). | 💳 Fraude pattern |
| 104 | `client_weekend_rate_before` | Proporción histórica de compras del cliente en fin de semana (`Sat` / `Sun`). | 📅 Comportamiento |
| 105 | `hotel_distance_zscore` | Z-score de `distance_from_home_km` respecto al historial de distancias hoteleras del cliente. | 🏨 Hotel 🟢 Innovadora |
| 106 | `hour_deviation_from_usual` | `|hour_local − media_histórica_hora_cliente|`. Cuánto se aleja la hora actual de la hora habitual del cliente. | 🟢 Innovadora |
| 107 | `client_mcc_diversity_before` | Nº de MCCs distintos visitados históricamente por el cliente (expanding nunique). Mide diversidad de gasto. | 🟢 Innovadora |

---

## 12. Variables excluidas del modelo (leakage)

Estas variables **no deben usarse como features** porque reflejan decisiones tomadas **después** de la transacción (post-autorización). Usarlas como input introduciría información futura al modelo.

| Variable | Razón de exclusión |
|---|---|
| `approved` | Resultado de la autorización — se conoce después de la decisión |
| `DE39_response_code` | Código de respuesta del emisor — post-autorización |
| `response_description` | Descripción textual del código de respuesta — post-autorización |
| `DE38_authorization_code` | Código emitido solo si la transacción es aprobada — post-autorización |

---

## Resumen de conteos

| Categoría | Columnas |
|---|---|
| Identificadores (excluir) | 8 |
| Campos ISO 8583 originales | 35 |
| Variables de contexto banco/cliente | 17 |
| Variables temporales preprocesamiento | 4 |
| Features hospedaje — EDA | 5 |
| Features históricas por cliente | 7 |
| Features históricas por canal | 4 |
| Features históricas de hospedaje | 4 |
| Features temporales / velocidad | 3 |
| Features ventana deslizante — Half 1 | 9 |
| **Half 2 pendiente** | **10** |
| **TOTAL actual** | **97** |
| **TOTAL con Half 2** | **107** |

---

*Generado automáticamente a partir de `data/processed/features_dataset.csv`*  
*Notebook fuente: `notebooks/02_feature_engineering.ipynb`*
