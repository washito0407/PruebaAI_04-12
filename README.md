# PruebaAI_04-12
Se incluto imagenes y proceso dentro del documento, acontinuación el codigo completo

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns
datos=pd.read_csv("zoo.csv")

datos.head()
#Ddatasheet
datos.describe()


# carga del conjunto de los datos zoo
data = pd.read_csv('zoo.csv')

# clasificacion 
X = data.drop(['animal_name', 'class_type'], axis=1)  # Características
y = data['class_type']  # Etiquetas

#conjunto de entrenamiento al 70% y prueba al 30%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear el modelo de árbol de decisión
model = DecisionTreeClassifier(random_state=42)

# Entrenar el modelo con el conjunto de entrenamiento
model.fit(X_train, y_train)


from sklearn.metrics import confusion_matrix

# Hacer predicciones en el conjunto de prueba
predictions = model.predict(X_test)

# Calcular la matriz de confusión
conf_matrix = confusion_matrix(y_test, predictions)

# Mostrar la matriz de confusión como mapa de calor
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', xticklabels=range(1, 8), yticklabels=range(1, 8))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Matriz de Confusión')
plt.show()

# Crear un DataFrame con las instancias a predecir
animals_to_predict = pd.DataFrame({
    "hair": [0, 1, 1, 0, 0, 0],
    "feathers": [0, 1, 0, 0, 0, 0],
    "eggs": [1, 1, 0, 1, 1, 1],
    "milk": [0, 0, 1, 0, 0, 0],
    "airborne": [1, 1, 0, 0, 0, 0],
    "aquatic": [0, 0, 0, 1, 0, 1],
    "predator": [0, 0, 1, 1, 1, 1],
    "toothed": [0, 0, 1, 1, 1, 1],
    "backbone": [0, 1, 1, 1, 1, 0],
    "breathes": [1, 1, 1, 1, 1, 1],
    "venomous": [0, 0, 0, 1, 0, 1],
    "fins": [0, 0, 0, 1, 0, 0],
    "legs": [6, 2, 2, 0, 4, 100],
    "tail": [0, 1, 1, 1, 1, 0],
    "domestic": [0, 1, 0, 0, 0, 0]
}, index=["Mariposa", "Canario", "Mono", "Tiburón", "Lagarto", "Cien pies"])

# Predecir las clases
predictions = model.predict(animals_to_predict)

# Imprimir las predicciones
for animal_name, prediction in zip(animals_to_predict.index, predictions):
    print(f'{animal_name}: Valor de clasificacion de la especie: {prediction}')

    import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Cargar los datos
data = pd.read_csv('zoo.csv')

# Separar las características (X) y la variable objetivo (y)
X = data.drop(['animal_name', 'class_type'], axis=1)
y = data['class_type']

# Entrenar el modelo
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# Obtener importancia de las características
feature_importance = model.feature_importances_

# Crear un DataFrame para mostrar la importancia de cada característica
feature_importance_df = pd.DataFrame({"Caracteristica": X.columns, "Importancia": feature_importance})
feature_importance_df = feature_importance_df.sort_values(by="Importancia", ascending=False)

# Mostrar el DataFrame ordenado por importancia
print("Importancia de las características:")
print(feature_importance_df)

