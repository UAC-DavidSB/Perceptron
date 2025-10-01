import numpy as np
import random

class PerceptronStep:
    def __init__(self, learning_rate=0.1, epochs=15):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
    
    def step_activation(self, x):
        """Función de activación escalón (Step)"""
        return 1 if x >= 0 else 0
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Inicialización con ruido
        self.weights = np.array([random.uniform(-0.3, 0.3) for _ in range(n_features)])
        self.bias = random.uniform(-0.3, 0.3)
        
        print("🔧 PERCEPTRÓN AND - FUNCIÓN ESCALÓN")
        print(f"Configuración: LR={self.lr}, Épocas={self.epochs}")
        print("Función de activación: Escalón (Step)")
        print("-" * 50)
        
        early_stop = self.epochs - 4
        
        for epoch in range(self.epochs):
            total_error = 0
            correct_predictions = 0
            
            for i in range(n_samples):
                # Forward pass
                linear_output = np.dot(X[i], self.weights) + self.bias
                prediction = self.step_activation(linear_output)
                
                error = y[i] - prediction
                total_error += abs(error)
                
                if prediction == y[i]:
                    correct_predictions += 1
                
                # Actualizar con ruido controlado
                noise = random.uniform(-0.03, 0.03)
                self.weights += self.lr * error * X[i] + noise
                self.bias += self.lr * error + noise
            
            accuracy = (correct_predictions / n_samples) * 100
            
            if epoch % 3 == 0 or epoch == self.epochs - 1:
                print(f"Época {epoch:2d} | Error: {total_error:.4f} | Precisión: {accuracy:.1f}%")
                print(f"  Salida lineal: {np.dot(X, self.weights) + self.bias}")
            
            # Parada temprana
            if epoch >= early_stop and accuracy > 85:
                print(f"⚡ Parada temprana en época {epoch}")
                break
    
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        predictions = np.array([self.step_activation(x) for x in linear_output])
        
        # Mostrar valores intermedios
        print(f"\nValores lineales: {linear_output}")
        print(f"Predictions after step: {predictions}")
        
        return predictions

def test_and_extensions():
    """Probar el perceptrón AND con casos extendidos"""
    print("\n🧪 PRUEBAS EXTENDIDAS - COMPUERTA AND")
    print("=" * 50)
    
    # Casos de prueba adicionales
    test_cases = [
        # Casos básicos AND
        ([0, 0], 0),
        ([0, 1], 0),
        ([1, 0], 0),
        ([1, 1], 1),
        
        # Casos con ruido (simulando datos reales)
        ([0.1, 0.1], 0),   # Casi 0,0
        ([0.9, 0.9], 1),   # Casi 1,1
        ([0.8, 0.2], 0),   # Casi 1,0
        ([0.1, 0.8], 0),   # Casi 0,1
        
        # Casos límite
        ([0.6, 0.6], 0),   # Podría ser ambiguo
        ([0.4, 0.4], 0),   # Claramente bajo
    ]
    
    print("Casos de prueba AND extendidos:")
    print("Entrada | Esperado | Predicho | Correcto")
    print("--------|----------|----------|----------")
    
    correct = 0
    for inputs, expected in test_cases:
        # Convertir a array numpy para la predicción
        input_array = np.array([inputs])
        prediction = model_and.predict(input_array)[0]
        
        # Redondear entradas para mostrar
        display_inputs = [round(x) for x in inputs]
        is_correct = prediction == expected
        status = "✅" if is_correct else "❌"
        
        print(f"{display_inputs} | {expected:8} | {prediction:8} | {status}")
        
        if is_correct:
            correct += 1
    
    accuracy = (correct / len(test_cases)) * 100
    print(f"\n📊 Precisión en pruebas extendidas: {accuracy:.1f}%")

def test_real_world_and():
    """Simular casos del mundo real que se comportan como AND"""
    print("\n🌍 CASOS REALES - COMPORTAMIENTO AND")
    print("=" * 50)
    
    # Ejemplo 1: Sistema de seguridad (ambas condiciones deben cumplirse)
    security_cases = [
        # [puerta_cerrada, alarma_activada, sistema_activo]
        ([1, 1], 1),  # Puerta cerrada Y alarma activada → Sistema activo
        ([1, 0], 0),  # Puerta cerrada pero alarma no activada
        ([0, 1], 0),  # Alarma activada pero puerta abierta
        ([0, 0], 0),  # Nada activado
    ]
    
    print("🔒 Sistema de Seguridad (AND):")
    for inputs, expected in security_cases:
        prediction = model_and.predict(np.array([inputs]))[0]
        status = "ACTIVO" if prediction == 1 else "INACTIVO"
        expected_status = "ACTIVO" if expected == 1 else "INACTIVO"
        print(f"  Puerta: {inputs[0]}, Alarma: {inputs[1]} → {status} (Esperado: {expected_status})")
    
    # Ejemplo 2: Aprobación de préstamo (ambos criterios)
    loan_cases = [
        # [buen_historial, ingresos_suficientes, aprobado]
        ([1, 1], 1),  # Buen historial Y ingresos suficientes
        ([1, 0], 0),  # Buen historial pero ingresos insuficientes
        ([0, 1], 0),  # Ingresos suficientes pero mal historial
    ]
    
    print("\n💰 Aprobación de Préstamo (AND):")
    for inputs, expected in loan_cases:
        prediction = model_and.predict(np.array([inputs]))[0]
        status = "APROBADO" if prediction == 1 else "RECHAZADO"
        print(f"  Historial: {inputs[0]}, Ingresos: {inputs[1]} → {status}")

# DATOS AND LÓGICO
print("📊 ENTRENAMIENTO AND LÓGICO")
print("Función de activación: ESCALÓN")
print("=" * 50)

X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])

# Mostrar tabla AND
print("Tabla AND Verdadera:")
print("X0 | X1 | Y")
print("---+----+---")
for i in range(len(X_and)):
    print(f" {X_and[i][0]} |  {X_and[i][1]} | {y_and[i]}")

print("\n" + "=" * 50)

# ENTRENAMIENTO
model_and = PerceptronStep(learning_rate=0.1, epochs=12)
model_and.fit(X_and, y_and)

# PREDICCIÓN
print("\n🎯 RESULTADOS FINALES - FUNCIÓN ESCALÓN")
print("=" * 40)
predictions = model_and.predict(X_and)

print("\nComparación final:")
for i in range(len(X_and)):
    status = "✅" if predictions[i] == y_and[i] else "⚠️"
    print(f"{status} Entrada: {X_and[i]} → Esperado: {y_and[i]}, Predicho: {predictions[i]}")

accuracy = np.mean(predictions == y_and) * 100
print(f"\n📈 Precisión final: {accuracy:.1f}%")
print("💡 Característica: Salida binaria (0 o 1) - Ideal para problemas lineales separables")

# EJECUTAR PRUEBAS
test_and_extensions()
test_real_world_and()