# Código extraído desde explore.ipynb



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



base_url = "https://raw.githubusercontent.com/4GeeksAcademy/Finarosalina_diabetes_machine-learning-py-template/main/data/processed/"

X_train2 = pd.read_csv(base_url + "X_train2.csv")
X_test2 = pd.read_csv(base_url + "X_test2.csv")
y_train2 = pd.read_csv(base_url + "y_train2.csv")
y_test2 = pd.read_csv(base_url + "y_test2.csv")


X_train2.head()

X_test2.head()



X_test2.head()
X_test2.shape

X_train2.to_csv("/workspaces/Finarosalina_Random_Forest_/data/processed/X_train2.csv", index=False)
X_test2.to_csv("/workspaces/Finarosalina_Random_Forest_/data/processed/X_test2.csv", index=False)
y_train2.to_csv("/workspaces/Finarosalina_Random_Forest_/data/processed/y_train2.csv", index=False)
y_test2.to_csv("/workspaces/Finarosalina_Random_Forest_/data/processed/y_test2.csv", index=False)


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=150, max_depth=5, min_samples_split=10, min_samples_leaf=5)
model.fit(X_train2, y_train2)


y_pred = model.predict(X_test2)
y_pred

from sklearn.metrics import accuracy_score

accuracy_score(y_test2, y_pred)

from sklearn.tree import DecisionTreeClassifier

simple_features = ["Glucose", "BMI", "Age"]  # para poder visualizarlo, lo pinto sólo con 3 variables
X_simple = X_train2[simple_features]


simple_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
simple_tree.fit(X_simple, y_train2)

plt.figure(figsize=(15, 10))
tree.plot_tree(simple_tree,
               feature_names=simple_features,
               class_names=["0", "1"],
               filled=True)
plt.show()


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

