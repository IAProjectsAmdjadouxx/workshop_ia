import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def generate_data(num_samples=1000):
    X = []
    y = []
    operations = ['addition', 'soustraction', 'multiplication']

    for _ in range(num_samples):
        a = np.random.randint(1, 100)  # Génére un nombre aléatoire entre 1 et 100
        b = np.random.randint(1, 100)  # Génére un autre nombre aléatoire entre 1 et 100
        operation = np.random.choice(operations)  # Choisi une opération aléatoire

        # Effectue l'opération choisie et attribuer une étiquette correspondante
        if operation == 'addition':
            result = a + b
            label = 0
        elif operation == 'soustraction':
            result = a - b
            label = 1
        elif operation == 'multiplication':
            result = a * b
            label = 2

        # Ajoute les données générées à notre liste :
        # X = [a, b, result] et y = label
        X.append([a, b, result])
        y.append(label)

    return np.array(X), np.array(y)

# Générer les données d'entraînement et de validation
X, y = generate_data()

#--------------------------------------------------------------------------------------------------------------

# Convertir les étiquettes en vecteurs one-hot
y = to_categorical(..., num_classes=3)

#--------------------------------------------------------------------------------------------------------------

# Créer le modèle de réseau de neurones
model = Sequential()

# Ajouter une couche dense avec 64 neurones et une activation ReLU
model.add(Dense(..., input_dim=3, activation=...))

# Ajouter une autre couche dense avec 32 neurones et une activation ReLU
model.add(Dense(..., activation=...))

# Ajouter la couche de sortie avec 3 neurones (une pour chaque classe) et une activation softmax
model.add(Dense(..., activation=...))


#--------------------------------------------------------------------------------------------------------------

# Compiler le modèle avec l'optimiseur 'adam' et la perte 'categorical_crossentropy'
model.compile(optimizer=..., loss=..., metrics=[...])

#--------------------------------------------------------------------------------------------------------------

#Diviser les données en jeu d'entraînement et de validation
X_train, X_validation, y_train, y_validation = train_test_split(..., ..., test_size=0.2, random_state=42)


#--------------------------------------------------------------------------------------------------------------

#Entraîner le modèle avec 20 époques et une taille de lot de 32 et utiliser les données de validation
model.fit(X_train, y_train, epochs=..., batch_size=..., validation_data=(..., ...))

#--------------------------------------------------------------------------------------------------------------

#Évaluer le modèle sur les données de validation et afficher la perte et la précision
loss, accuracy = model.evaluate(X_val, y_val)

#--------------------------------------------------------------------------------------------------------------

#Faire des prédictions sur les nouvelles données et afficher les résultats
new_data = np.array([[10, 5, 15], [20, 4, 16], [7, 6, 42]])

predictions = model.predict(...)

#--------------------------------------------------------------------------------------------------------------

# Affiche les résultats de prédiction
operations = ['addition', 'soustraction', 'multiplication']

for i, (data, prediction) in enumerate(zip(new_data, predicted_classes)):
    print(f'Data {i+1}: {data}')
    print(f'Predicted operation: {operations[prediction]}\n')
