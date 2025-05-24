from flask import Flask, request
from flask_restx import Api, Resource, fields
import joblib
import torch
import numpy as np

# === Cargar artefactos ===
vectorizer = joblib.load("vectorizer.pkl")
mlb = joblib.load("binarizer.pkl")

class MLPClassifier(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, output_dim),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

input_dim = 1000
output_dim = len(mlb.classes_)
model = MLPClassifier(input_dim, output_dim)
model.load_state_dict(torch.load("mlp_model.pt", map_location=torch.device("cpu")))
model.eval()

# === Configuración Flask ===
app = Flask(__name__)
api = Api(app, version='1.0', title='🎬 API Clasificación de Géneros de Películas',
          description='Clasifica los géneros a partir de título, año y sinopsis',
          doc='/docs')

ns = api.namespace('predict', description='Predicción de géneros')

# === Modelo entrada POST ===
entrada_post = api.model('EntradaPost', {
    'input_text': fields.String(required=True, description='Título (año): sinopsis')
})

# === POST: predicción desde JSON ===
@ns.route('/')
class PrediccionPost(Resource):
    @ns.expect(entrada_post)
    def post(self):
        input_text = request.json['input_text']
        return procesar_input(input_text)

# === GET: predicción desde campos separados ===
@ns.route('/get')
@ns.doc(params={
    'title': 'Título de la película',
    'year': 'Año de la película',
    'plot': 'Sinopsis de la película'
})
class PrediccionGet(Resource):
    def get(self):
        title = request.args.get('title', '')
        year = request.args.get('year', '')
        plot = request.args.get('plot', '')
        input_text = f"{title} ({year}): {plot}"
        return procesar_input(input_text)

# === Función de predicción común ===
def procesar_input(texto):
    X = vectorizer.transform([texto]).toarray()
    X_tensor = torch.tensor(X).float()
    with torch.no_grad():
        probs = model(X_tensor)[0].numpy()
    etiquetas = mlb.classes_[probs >= 0.5].tolist()
    return {'predicted_genres': etiquetas}

# === Ejecutar aplicación ===
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
