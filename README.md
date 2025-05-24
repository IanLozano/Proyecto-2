# 🎬 API de Clasificación de Géneros de Películas

Este repositorio contiene un proyecto completo de procesamiento de texto, entrenamiento de un modelo MLP y despliegue de una API para predecir los géneros de películas a partir del título, año y sinopsis.

---

## 📌 Objetivo

Predecir los géneros (multilabel) más probables de una película usando un modelo entrenado sobre sinopsis y metadatos.

---

## 🧠 Modelo MLP

Modelo de red neuronal multicapa entrenado con PyTorch:

- **Input**: texto concatenado ("title (year): plot")

- **Vectorización**: `CountVectorizer` con `max_features=1000`

- **Etiquetas**: 24 géneros codificados con `MultiLabelBinarizer`

- **Modelo**: MLP con las siguientes capas:

  - Linear(1000, 512)
  - ReLU
  - Dropout(0.3)
  - Linear(512, 24)
  - Sigmoid

- **Entrenamiento**:

  - `optimizer = Adam(lr=1e-3)`
  - `loss = BCELoss()`
  - `epochs = 10`, `batch_size = 64`

### 🔁 Entrenamiento

```bash
python3 entrenar_modelo.py
```

Este script genera:

- `mlp_model.pt`: pesos del modelo
- `vectorizer.pkl`: vectorizador CountVectorizer
- `binarizer.pkl`: binarizador multilabel

---

## 🌐 API REST con Flask + Flask-RESTX

API expuesta para recibir texto y devolver géneros predichos.

### 📂 Estructura

- `main.py`: script principal de la API
- `POST /predict/` → Entrada JSON: `{ "input_text": "..." }`
- `GET /predict/get` → Parámetros: `title`, `year`, `plot`

### 🔁 Ejemplo POST

```json
{
  "input_text": "The Matrix (1999): A hacker discovers reality is a simulation controlled by machines."
}
```

Respuesta:

```json
{
  "predicted_genres": ["Action", "Sci-Fi"]
}
```

### 🧪 Swagger UI

Disponible en:

```
http://<IP_PUBLICA>:5000/docs
```

---

## ☁️ Despliegue en AWS EC2

### 1. Crear instancia EC2 (Amazon Linux o Ubuntu)

- Tipo `t2.micro` o superior
- Abrir puerto 5000 en el **grupo de seguridad**

### 2. Conectarse por SSH

```bash
ssh -i "clave.pem" ubuntu@<IP_PUBLICA>
```

### 3. Instalar dependencias

```bash
sudo apt update
sudo apt install python3-pip screen -y
pip3 install flask flask-restx torch scikit-learn joblib
```

### 4. Subir archivos

- `main.py`
- `mlp_model.pt`
- `vectorizer.pkl`
- `binarizer.pkl`

Usa `scp` o FileZilla para copiarlos.

### 5. Ejecutar con `screen`

```bash
screen -S api
python3 main.py
# Luego: Ctrl+A y después D
```

API quedará funcionando en segundo plano.

---

## 🔗 Uso Externo

- **Swagger**: `http://<IP_PUBLICA>:5000/docs`
- **POST**: `http://<IP_PUBLICA>:5000/predict/`
- **GET**: `http://<IP_PUBLICA>:5000/predict/get?title=...&year=...&plot=...`

---

## 🧾 Requisitos

- Python >= 3.8
- Flask, Flask-RESTX
- PyTorch
- Scikit-learn
- Joblib

---

## ✍️ Autor

Proyecto académico desarrollado para demostrar procesamiento de texto, clasificación multilabel y despliegue en servidor en la nube usando Flask.

---

## ✅ Estado

-
