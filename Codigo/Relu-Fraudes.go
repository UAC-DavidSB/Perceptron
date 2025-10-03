package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

type PerceptronFraudReLU struct {
	weights      []float64
	bias         float64
	learningRate float64
	epochs       int
}

func NewPerceptronFraudReLU(learningRate float64, epochs int) *PerceptronFraudReLU {
	rand.Seed(time.Now().UnixNano())
	return &PerceptronFraudReLU{
		learningRate: learningRate,
		epochs:       epochs,
	}
}

// Leaky ReLU
func (p *PerceptronFraudReLU) leakyRelu(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0.01 * x
}

// Entrenamiento
func (p *PerceptronFraudReLU) Train(X [][]float64, y []float64) {
	nFeatures := len(X[0])
	p.weights = make([]float64, nFeatures)

	// Inicialización aleatoria
	for i := range p.weights {
		p.weights[i] = rand.Float64()*0.4 - 0.2
	}
	p.bias = rand.Float64()*0.4 - 0.2

	earlyStop := p.epochs - 10

	for epoch := 0; epoch < p.epochs; epoch++ {
		totalError := 0.0
		correctPredictions := 0

		for i, sample := range X {
			linearOutput := 0.0
			for j, feature := range sample {
				linearOutput += feature * p.weights[j]
			}
			linearOutput += p.bias

			output := p.leakyRelu(linearOutput)
			error := y[i] - output
			totalError += math.Abs(error)

			prediction := 0.0
			if output > 0.5 {
				prediction = 1
			}
			if prediction == y[i] {
				correctPredictions++
			}

			// Derivada Leaky ReLU
			delta := error
			if linearOutput <= 0 {
				delta *= 0.01
			}

			// Actualización con un poco de regularización
			for j := range p.weights {
				p.weights[j] += p.learningRate * delta * sample[j]
				p.weights[j] *= 0.999 // regularización L2
			}
			p.bias += p.learningRate * delta
		}

		accuracy := float64(correctPredictions) / float64(len(X)) * 100

		if epoch%20 == 0 || epoch == p.epochs-1 {
			fmt.Printf("Época %3d | Error: %.4f | Precisión: %.1f%%\n", epoch, totalError, accuracy)
		}

		if epoch >= earlyStop && accuracy > 90 {
			fmt.Printf("⚡ Parada temprana en época %d\n", epoch)
			break
		}
	}
}

// Predicción
func (p *PerceptronFraudReLU) Predict(X [][]float64) []float64 {
	predictions := make([]float64, len(X))
	for i, sample := range X {
		linearOutput := 0.0
		for j, feature := range sample {
			linearOutput += feature * p.weights[j]
		}
		linearOutput += p.bias

		output := p.leakyRelu(linearOutput)
		if output > 0.5 {
			predictions[i] = 1
		} else {
			predictions[i] = 0
		}
	}
	return predictions
}

// Generar dataset sintético
func generateData(nSamples int, nFeatures int) ([][]float64, []float64) {
	X := make([][]float64, nSamples)
	y := make([]float64, nSamples)

	for i := 0; i < nSamples; i++ {
		X[i] = make([]float64, nFeatures)
		sum := 0.0
		for j := 0; j < nFeatures; j++ {
			val := rand.Float64()
			X[i][j] = val
			sum += val
		}
		avg := sum / float64(nFeatures)

		// Regla artificial para etiquetar fraude
		if avg > 0.55 {
			y[i] = 1
		} else {
			y[i] = 0
		}
	}
	return X, y
}

func main() {
	// Generamos dataset sintético
	X, y := generateData(100, 5)

	// Separar 80% train / 20% test
	trainSize := int(0.8 * float64(len(X)))
	X_train := X[:trainSize]
	y_train := y[:trainSize]
	X_test := X[trainSize:]
	y_test := y[trainSize:]

	// Crear y entrenar perceptrón
	perceptron := NewPerceptronFraudReLU(0.015, 200)
	perceptron.Train(X_train, y_train)

	// Evaluación en test
	predictions := perceptron.Predict(X_test)

	fmt.Println("\n🎯 RESULTADOS EN TEST")
	fmt.Println("======================")
	correct := 0
	for i, pred := range predictions {
		status := "✅"
		if pred != y_test[i] {
			status = "⚠️"
		} else {
			correct++
		}
		fmt.Printf("%s Ejemplo %d: Esperado=%.0f, Predicho=%.0f\n", status, i+1, y_test[i], pred)
	}

	accuracy := float64(correct) / float64(len(predictions)) * 100
	fmt.Printf("\n📈 Precisión en test: %.1f%%\n", accuracy)
}
