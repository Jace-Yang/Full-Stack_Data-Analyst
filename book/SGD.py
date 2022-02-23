import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

def eggholder(X):
	return (-(X[1] + 47) * np.sin(np.sqrt(abs(X[0]/2 + (X[1] + 47)))) - X[0] * np.sin(np.sqrt(abs(X[0] - (X[1] + 47)))))

np.random.seed(2022)
X = np.random.uniform(low=-512, high=512, size=(100000, 2))
F = np.array([eggholder(i) for i in X])
y = F + np.random.normal(0, 0.3, 100000)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2022)

model = Sequential([layers.Dense(units=16, activation='relu', name='FC_1', input_dim=2),
                    #layers.Dense(units=16, activation='relu', name='FC_2'),
                    #layers.BatchNormalization(),
                    layers.Dense(units=1, activation='linear')])
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.01, nesterov=True), 
              #optimizer='adam', 
              loss='mse',
              metrics=[tf.keras.metrics.RootMeanSquaredError()])
model.summary()
history = model.fit(x=X_train, 
                    y=y_train, 
                    epochs=2000,
                    batch_size=1000,
                    #steps_per_epoch=10,
                    validation_data=(X_test, y_test), 
                    verbose=1)