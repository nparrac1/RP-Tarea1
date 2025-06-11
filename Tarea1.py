import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# === Cargar y preparar datos ===
data, meta = arff.loadarff("Autism-Child-Data.arff")
df = pd.DataFrame(data)

# Decodificar variables tipo bytes
for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].str.decode("utf-8")

# Reemplazar '?' con NaN y eliminar filas incompletas
df.replace("?", pd.NA, inplace=True)
df.dropna(inplace=True)

# Columnas que se eliminarán (filtran la clase o no aportan)
cols_to_drop = ['Class/ASD', 'result', 'age_desc', 'austim']

# Separar variables predictoras y variable objetivo
X = df.drop(columns=cols_to_drop)
y = df["Class/ASD"].map({"NO": 0, "YES": 1})

# Codificar variables categóricas
for col in X.columns:
    if X[col].dtype == object:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

# Dividir el dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# === Probar distintas combinaciones de parámetros ===
results = []
estimators_range = [50, 100, 150]
depth_range = [5, 10, 15]
split_range = [2, 4, 6]

for n in estimators_range:
    for d in depth_range:
        for s in split_range:
            clf = RandomForestClassifier(
                n_estimators=n,
                max_depth=d,
                min_samples_split=s,
                random_state=42
            )
            clf.fit(X_train, y_train)

            acc_train = accuracy_score(y_train, clf.predict(X_train))
            acc_test = accuracy_score(y_test, clf.predict(X_test))

            results.append({
                "n_estimators": n,
                "max_depth": d,
                "min_samples_split": s,
                "accuracy_train": acc_train,
                "accuracy_test": acc_test
            })

# Convertir resultados a DataFrame
df_results = pd.DataFrame(results)
print(df_results)

# Mostrar las 5 mejores combinaciones según accuracy en test
print("\nTop 5 combinaciones con mejor desempeño en test:")
print(df_results.sort_values("accuracy_test", ascending=False).head())

# === Graficar relación entre max_depth y accuracy (n_estimators=100, min_samples_split=2)
subset = df_results[
    (df_results["n_estimators"] == 100) &
    (df_results["min_samples_split"] == 2)
]

plt.figure(figsize=(8, 5))
plt.plot(subset["max_depth"], subset["accuracy_train"], label="Train Accuracy", marker='o')
plt.plot(subset["max_depth"], subset["accuracy_test"], label="Test Accuracy", marker='o')
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.title("Efecto de max_depth en Accuracy (n=100, min_samples_split=2)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



# === Gráfico mejorado con más valores de max_depth ===
depth_range_ext = [3, 5, 7, 10, 12, 15]
train_acc = []
test_acc = []

for d in depth_range_ext:
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=d,
        min_samples_split=2,
        random_state=42
    )
    clf.fit(X_train, y_train)
    train_acc.append(accuracy_score(y_train, clf.predict(X_train)))
    test_acc.append(accuracy_score(y_test, clf.predict(X_test)))

# Graficar nuevo resultado más detallado
plt.style.use('ggplot')   # Cambia el estilo del gráfico (opcional)
plt.figure(figsize=(8, 5))
plt.plot(depth_range_ext, train_acc, marker='o', label="Train Accuracy", linewidth=2)
plt.plot(depth_range_ext, test_acc, marker='o', label="Test Accuracy", linewidth=2)
plt.title("Relación entre max_depth y Accuracy (n_estimators=100, min_samples_split=2)", fontsize=12)
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.ylim(0.85, 1.01)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()




#3
print('PASOO 3----------------------------------')
from sklearn.metrics import confusion_matrix, classification_report

# Crear y entrenar modelo óptimo
best_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=4,
    random_state=42
)
best_model.fit(X_train, y_train)

# Predicción
y_pred = best_model.predict(X_test)

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
print("Matriz de Confusión:")
print(cm)

# Reporte de clasificación (incluye precision, recall y f1-score)
report = classification_report(y_test, y_pred, target_names=["NO", "YES"], output_dict=True)
df_report = pd.DataFrame(report).transpose()
print("\nReporte de Clasificación:")
print(df_report)

# Accuracy promedio por clase (igual que macro avg recall)
accuracy_promedio = df_report.loc[["NO", "YES"], "recall"].mean()
print(f"\nAccuracy global: {accuracy_score(y_test, y_pred):.2f}")
print(f"Accuracy promedio por clase: {accuracy_promedio:.2f}")





print('PASOO 4----------------------------------')
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# Número original de características
n_features = X.shape[1]

# Función para evaluar modelo con reducción de dimensionalidad
def evaluar_con_pca(n_componentes):
    print(f"\n--- PCA con {n_componentes} componentes ---")
    # Crear pipeline con PCA + RandomForest
    pipeline = Pipeline([
        ("pca", PCA(n_components=n_componentes)),
        ("clf", RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=4, random_state=42))
    ])
    
    # Entrenar
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Métricas
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    recall_prom = (report["0"]["recall"] + report["1"]["recall"]) / 2

    print(f"Accuracy: {acc:.2f}")
    print(f"Accuracy promedio por clase (recall): {recall_prom:.2f}")
    print(f"Precisión clase YES: {report['1']['precision']:.2f}")
    print(f"Recall clase YES: {report['1']['recall']:.2f}")
    print(f"Precisión clase NO: {report['0']['precision']:.2f}")
    print(f"Recall clase NO: {report['0']['recall']:.2f}")

# Probar con la mitad de las características
evaluar_con_pca(n_features // 2)

# Probar con solo 2 componentes
evaluar_con_pca(2)
