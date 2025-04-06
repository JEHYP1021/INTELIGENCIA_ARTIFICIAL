import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

celsios = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

modelo = tf.keras.Sequential([tf.keras.Input(shape=(1,)), tf.keras.layers.Dense(units=1)])

modelo.compile(optimizer = tf.keras.optimizers.Adam(0.1), loss = 'mean_squared_error')
if __name__=="__main__":
    print("Comenzando entrenamiento...")
    historial = modelo.fit(celsios, fahrenheit, epochs=1000, verbose=False)
    print("Modelo entrenado!")

plt.xlabel("#Epoca")
plt.ylabel("Magnitud de perdida")
plt.plot(historial.history["loss"])
plt.show()

print("Hagamos una predicción:")
resultado = modelo.predict(np.array([100.0]))
print(f'El resultado es: {resultado[0][0]:.2f} fahrenheit')