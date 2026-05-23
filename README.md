# DL Proyecto — Clasificación de Sentimiento en Reseñas Hoteleras (AHR)

Proyecto de Deep Learning para clasificación binaria de sentimiento (positivo/negativo) sobre reseñas hoteleras en español del dataset Andalusian Hotel Reviews (AHR), comparando varias arquitecturas recurrentes e híbridas:

- LSTM
- BiGRU
- Conv1D + BiLSTM (guardado como ConvLSTM en nombres de artefactos)
- Transformer BETO (Fine-tuning parcial de dccuchile/bert-base-spanish-wwm-uncased)

El flujo va desde EDA y preprocesamiento hasta entrenamiento de modelos y comparación final de métricas/figuras.

## 1. Estructura del Proyecto

```text
DL_Proyecto/
├── 01_EDA_Preprocessing.ipynb
├── 02_Model_LSTM.ipynb
├── 03_Model_BiGRU.ipynb
├── 04_Model_ConvLSTM.ipynb
├── 05_Results_Comparison.ipynb
├── pyproject.toml
├── uv.lock
├── README.md
├── data/
│   └── raw/
│       ├── Big_AHR.csv
│       └── Balanced_AHR.csv
├── 06_Model_Transformer.ipynb
└── results/
		├── data/
		│   ├── big_ahr/
		│   │   ├── dataset_clean.csv
		│   │   ├── meta.json
		│   │   ├── tokenizer.pkl
		│   │   ├── X_train.npy / X_val.npy / X_test.npy
		│   │   └── y_train.npy / y_val.npy / y_test.npy
		│   └── balanced_ahr/
		│       ├── dataset_clean.csv
		│       ├── meta.json
		│       ├── tokenizer.pkl
		│       ├── X_train.npy / X_val.npy / X_test.npy
		│       └── y_train.npy / y_val.npy / y_test.npy
		├── figures/
		│   ├── 01_class_distribution.png
		│   ├── 02_text_length_distribution.png
		│   ├── 03_wordclouds.png
		│   ├── LSTM_*.png
		│   ├── BiGRU_*.png
		│   ├── ConvLSTM_*.png
		│   ├── Transformer_*.png
		│   └── FINAL_*.png
		├── metrics/
		│   ├── LSTM_results.json
		│   ├── LSTM_summary.csv
		│   ├── BiGRU_results.json
		│   ├── BiGRU_summary.csv
		│   ├── ConvLSTM_results.json
		│   ├── ConvLSTM_summary.csv
		│   ├── Transformer_results.json
		│   ├── Transformer_summary.csv
		│   ├── ALL_models_comparison.csv
		│   └── FINAL_summary.csv
		└── models/
				├── LSTM/
				├── GRU/
				└── Combinado/
```

## 2. ¿Qué Hace Cada Notebook?

### 01_EDA_Preprocessing.ipynb

Objetivo: exploración de datos y generación de artefactos preprocesados para entrenamiento.

Hace lo siguiente:

1. Verifica/importa dependencias y configura semilla/GPU.
2. Carga `data/raw/Big_AHR.csv` y `data/raw/Balanced_AHR.csv`.
3. EDA:
	 - Distribución de clases
	 - Longitud de texto
	 - Nubes de palabras
4. Limpieza de texto en español:
	 - minúsculas, remoción de URLs/HTML/símbolos
	 - stopwords
	 - fusión de `title + review_text`
	 - eliminación de neutros (label = 3)
5. Tokenización y padding (Keras Tokenizer).
6. Split train/val/test estratificado.
7. Guarda artefactos en `results/data/{big_ahr,balanced_ahr}/`.

Salida principal:

- `X_train.npy`, `X_val.npy`, `X_test.npy`
- `y_train.npy`, `y_val.npy`, `y_test.npy`
- `tokenizer.pkl`
- `meta.json`
- `dataset_clean.csv`
- Figuras EDA en `results/figures/`

### 02_Model_LSTM.ipynb

Objetivo: entrenar y evaluar 4 configuraciones de LSTM en ambos datasets (`balanced`, `big_ahr`).

Hace lo siguiente:

1. Carga los artefactos preprocesados desde `results/data/`.
2. Define arquitecturas `LSTM_C1` a `LSTM_C4`.
3. Entrena con callbacks (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau).
4. Aplica `class_weight` en dataset desbalanceado.
5. Evalúa métricas: Accuracy, Precision, Recall, F1, ROC-AUC.
6. Guarda modelos, curvas, matrices de confusión, curvas ROC y resúmenes.

Salida principal:

- Modelos `.keras` en `results/models/LSTM/`
- Métricas en `results/metrics/LSTM_results.json` y `results/metrics/LSTM_summary.csv`
- Figuras `LSTM_*.png` en `results/figures/`

### 03_Model_BiGRU.ipynb

Objetivo: entrenar y comparar configuraciones BiGRU (bidireccionales) sobre ambos datasets.

Hace lo siguiente:

1. Reutiliza los artefactos de preprocesamiento.
2. Define configuraciones `BiGRU_C1` a `BiGRU_C3`.
3. Incluye opción de mecanismo de atención (clase definida, no activada por defecto).
4. Entrena/evalúa y guarda métricas/figuras equivalentes al notebook LSTM.

Salida principal:

- Modelos `.keras` en `results/models/GRU/`
- Métricas en `results/metrics/BiGRU_results.json` y `results/metrics/BiGRU_summary.csv`
- Figuras `BiGRU_*.png` en `results/figures/`

### 04_Model_ConvLSTM.ipynb

Objetivo: entrenar modelo híbrido Conv1D + BiLSTM (nombrado como ConvLSTM en archivos de salida).

Hace lo siguiente:

1. Carga los datos preprocesados.
2. Define configuraciones `ConvLSTM_C1` y `ConvLSTM_C2`.
3. Combina extracción local (Conv1D) + dependencias secuenciales (BiLSTM).
4. Entrena/evalúa en ambos datasets y guarda artefactos.

Salida principal:

- Modelos `.keras` en `results/models/Combinado/`
- Métricas en `results/metrics/ConvLSTM_results.json` y `results/metrics/ConvLSTM_summary.csv`
- Figuras `ConvLSTM_*.png` en `results/figures/`

### 06_Model_Transformer.ipynb

Objetivo: realizar ajuste fino parcial (*fine-tuning*) del modelo de lenguaje pre-entrenado BETO (`dccuchile/bert-base-spanish-wwm-uncased`) para clasificación de sentimiento.

Hace lo siguiente:

1. Carga los textos limpios generados en el Notebook 1.
2. Tokeniza los textos utilizando el tokenizador de subpalabras de BETO.
3. Aplica estrategia de congelamiento parcial: congela las primeras 11 capas de atención del codificador y mantiene entrenable la capa 12 y la cabeza de clasificación final.
4. Entrena con Hugging Face `Trainer` y guarda métricas de evaluación.

Salida principal:

- Modelo ajustado (guardado localmente)
- Métricas en `results/metrics/Transformer_results.json` y `results/metrics/Transformer_summary.csv`
- Figuras `Transformer_*.png` en `results/figures/`

### 05_Results_Comparison.ipynb

Objetivo: consolidar y comparar resultados de todas las familias de modelos, incluyendo redes clásicas (LSTM, BiGRU, ConvLSTM) y el Transformer (BETO).

Hace lo siguiente:

1. Carga `LSTM_results.json`, `BiGRU_results.json`, `ConvLSTM_results.json` y `Transformer_results.json`.
2. Construye tabla global y guarda `ALL_models_comparison.csv`.
3. Genera visualizaciones finales:
	 - Heatmap de F1
	 - Promedio por familia
	 - Comparación por dataset
	 - Análisis de hiperparámetros
	 - Eficiencia (F1 vs parámetros)
	 - Curvas de entrenamiento de los mejores modelos
4. Guarda `FINAL_summary.csv`.

Salida principal:

- `results/metrics/ALL_models_comparison.csv`
- `results/metrics/FINAL_summary.csv`
- Figuras `FINAL_*.png` en `results/figures/`

## 3. Orden Recomendado de Ejecución

Ejecuta los notebooks en este orden:

1. `01_EDA_Preprocessing.ipynb`
2. `02_Model_LSTM.ipynb`
3. `03_Model_BiGRU.ipynb`
4. `04_Model_ConvLSTM.ipynb`
5. `06_Model_Transformer.ipynb`
6. `05_Results_Comparison.ipynb`

Si no ejecutas el notebook 1 primero, los notebooks de modelado no encontrarán los archivos en `results/data/`.

## 4. Requisitos

- Python >= 3.12
- GPU con CUDA (Opcional, pero muy recomendado para el Transformer)
- Dependencias definidas en `pyproject.toml` más las librerías de Deep Learning y NLP:
	- tensorflow
	- torch (PyTorch)
	- transformers (Hugging Face)
	- accelerate (Hugging Face)
	- numpy
	- pandas
	- scikit-learn
	- matplotlib
	- seaborn
	- nltk
	- wordcloud

### Nota para GPUs NVIDIA Modernas (Arquitectura Blackwell - Serie RTX 5000)
Si utilizas una GPU basada en Blackwell (como la RTX 5060 Ti) bajo Windows, la versión estable de PyTorch (con CUDA 12.1/12.4) fallará con un error crítico de compilación. Para solucionarlo e instalar PyTorch con soporte para CUDA 12.8:

```bash
uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 --force-reinstall
```

## 5. Cómo Correr el Proyecto

### Opción A: con uv (recomendado, porque existe `uv.lock`)

```bash
uv sync
uv run jupyter notebook
```

### Opción B: con venv + pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install jupyter
pip install tensorflow numpy pandas scikit-learn matplotlib seaborn nltk wordcloud
jupyter notebook
```

Luego abre y ejecuta los notebooks en el orden recomendado.

## 6. Configuración de Datos de Entrada

Ubica los CSV originales en:

- `data/raw/Big_AHR.csv`
- `data/raw/Balanced_AHR.csv`

En el notebook 01, la ruta esperada es:

```python
DATA_DIR = "./data/raw/"
```

Si ejecutas en Google Colab, ajusta `DATA_DIR` a tu ruta de Drive.

## 7. Resultados Esperados

Al finalizar todo el pipeline deberías tener:

1. Datasets procesados y tokenizadores en `results/data/`.
2. Modelos entrenados por familia en `results/models/`.
3. Métricas y tablas de comparación en `results/metrics/`.
4. Visualizaciones intermedias y finales en `results/figures/`.

## 8. Problemas Comunes

- Error de archivos faltantes en notebooks 2/3/4:
	- Ejecuta primero `01_EDA_Preprocessing.ipynb`.

- Error de NLTK stopwords/punkt:
	- Re-ejecuta la celda de setup del notebook 01 (incluye `nltk.download(...)`).

- Sin GPU:
	- El código cae automáticamente a CPU, pero el entrenamiento será más lento.

## 9. Nota sobre Nomenclatura

Aunque el notebook 4 se llama `04_Model_ConvLSTM.ipynb`, la arquitectura implementada es híbrida Conv1D + BiLSTM. En los artefactos se conserva la etiqueta `ConvLSTM` para mantener consistencia en nombres de archivos.

