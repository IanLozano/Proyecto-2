import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib
import ast
from torch.utils.data import DataLoader, TensorDataset

# === Cargar datos desde URL corregida ===
df = pd.read_csv("https://github.com/albahnsen/MIAD_ML_and_NLP/raw/main/datasets/dataTraining.zip", index_col=0)
df['genres'] = df['genres'].apply(ast.literal_eval)
df['input_text'] = df['title'] + " (" + df['year'].astype(str) + "): " + df['plot']

# === Vectorización (1000 características) ===
vectorizer = CountVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['input_text']).toarray()

# === Binarización de etiquetas ===
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['genres'])

# === División de datos ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Dataset y DataLoader ===
batch_size = 64
train_dataset = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).float())
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# === Definir modelo MLP ===
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

input_dim = 1000
output_dim = y.shape[1]
model = MLPClassifier(input_dim, output_dim)

# === Entrenamiento con mini-batches ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

model.train()
for epoch in range(10):
    total_loss = 0.0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

# === Guardar artefactos ===
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(mlb, "binarizer.pkl")
torch.save(model.state_dict(), "mlp_model.pt")
