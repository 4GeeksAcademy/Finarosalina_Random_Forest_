# Código extraído desde explore.ipynb



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



test_data = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/Finarosalina_diabetes_machine-learning-py-template/main/data/processed/clean_test.csv")
train_data = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/Finarosalina_diabetes_machine-learning-py-template/main/data/processed/clean_train.csv")


test_data.head()
train_data.head()

X_train=train_data.drop(["Outcome"], axis = 1)
y_train=train_data["Outcome"]
X_test=test_data.drop(["Outcome"], axis=1)
y_test=test_data["Outcome"]
X_train
y_train

X_test.to_csv("/workspaces/Finarosalina_Random_Forest_/data/processed/X_test_data")
X_train.to_csv("/workspaces/Finarosalina_Random_Forest_/data/processed/X_train_data")

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_split=10, min_samples_leaf=5)
model.fit(X_train, y_train)


import matplotlib.pyplot as plt
from sklearn import tree

# Crear figura de 2x2
fig, axis = plt.subplots(2, 2, figsize=(15, 15))


tree.plot_tree(model.estimators_[0], ax=axis[0, 0],
               feature_names=X_train.columns.tolist(),
               class_names=["0", "1"],
               filled=True)

tree.plot_tree(model.estimators_[1], ax=axis[0, 1],
               feature_names=X_train.columns.tolist(),
               class_names=["0", "1"],
               filled=True)

tree.plot_tree(model.estimators_[2], ax=axis[1, 0],
               feature_names=X_train.columns.tolist(),
               class_names=["0", "1"],
               filled=True)

tree.plot_tree(model.estimators_[3], ax=axis[1, 1],
               feature_names=X_train.columns.tolist(),
               class_names=["0", "1"],
               filled=True)

plt.tight_layout()
plt.show()


y_pred = model.predict(X_test)
y_pred

from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)

from pickle import dump
model_filename = "/workspaces/Finarosalina_Random_Forest_/models/random_forest_classifier_default_42.sav"
dump(model, open(model_filename, "wb"))

print(f"Modelo guardado exitosamente en: {model_filename}")


import json

# Ruta del archivo .ipynb
notebook_path = "/workspaces/Finarosalina_Random_Forest_/src/explore.ipynb"

# Leer el archivo .ipynb como JSON
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Extraer el código de las celdas tipo "code"
code_cells = []
for cell in notebook.get('cells', []):
    if cell.get('cell_type') == 'code':
        code = ''.join(cell.get('source', []))
        code_cells.append(code)

# Ruta del archivo .py donde se guardará el código
output_path = "/workspaces/Finarosalina_Random_Forest_/src/app.py"

# Combinar el código y escribirlo en el archivo .py
with open(output_path, 'w', encoding='utf-8') as f:
    f.write("# Código extraído desde explore.ipynb\n\n")
    f.write("\n\n".join(code_cells))

print("✅ Código copiado exitosamente a app.py")
