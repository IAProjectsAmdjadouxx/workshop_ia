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
        a = np.random.randint(1, 100)
        b = np.random.randint(1, 100)
        operation = np.random.choice(operations)

        if operation == 'addition':
            result = a + b
            label = 0
        elif operation == 'soustraction':
            result = a - b
            label = 1
        elif operation == 'multiplication':
            result = a * b
            label = 2

        X.append([a, b, result])
        y.append(label)

    return np.array(X), np.array(y)

X, y = generate_data()
y = to_categorical(y, num_classes=3)

model = Sequential()
model.add(Dense(64, input_dim=3, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val))

loss, accuracy = model.evaluate(X_val, y_val)
print(f'Loss: {loss:.4f}')
print(f'Accuracy: {accuracy*100:.2f}%')

new_data = np.array([[10, 5, 15], [20, 4, 16], [7, 6, 42]])
predictions = model.predict(new_data)
predicted_classes = np.argmax(predictions, axis=1)

operations = ['addition', 'soustraction', 'multiplication']
for i, (data, prediction) in enumerate(zip(new_data, predicted_classes)):
    print(f'Data {i+1}: {data}')
    print(f'Predicted operation: {operations[prediction]}\n')