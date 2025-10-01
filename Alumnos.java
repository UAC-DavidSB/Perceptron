import java.util.*;

public class PerceptronRiesgoAcademico {
    static class Perceptron {
        double[] pesos;
        double bias;
        double tasaAprendizaje;

        public Perceptron(int entradas, double tasaAprendizaje) {
            this.pesos = new double[entradas];
            this.bias = 0.0;
            this.tasaAprendizaje = tasaAprendizaje;

            Random random = new Random();
            for (int i = 0; i < entradas; i++) {
                pesos[i] = random.nextDouble() * 2 - 1; // entre -1 y 1
            }
        }

        // Activacion tanh
        private double activacion(double x) {
            return Math.tanh(x);
        }

        // Derivada de tanh
        private double derivada(double x) {
            return 1 - Math.pow(Math.tanh(x), 2);
        }

        // Prediccion
        public double predecir(double[] entradas) {
            double suma = bias;
            for (int i = 0; i < pesos.length; i++) {
                suma += pesos[i] * entradas[i];
            }
            return activacion(suma);
        }

        // Entrenamiento
        public void entrenar(List<double[]> entradas, List<Double> salidas, int epocas) {
            for (int e = 0; e < epocas; e++) {
                int correctos = 0;

                for (int i = 0; i < entradas.size(); i++) {
                    double[] x = entradas.get(i);
                    double salidaEsperada = salidas.get(i);
                    double suma = bias;

                    for (int j = 0; j < pesos.length; j++) {
                        suma += pesos[j] * x[j];
                    }

                    double salidaPredicha = activacion(suma);
                    double error = salidaEsperada - salidaPredicha;

                    // Actualizacion de pesos con derivada de tanh
                    double gradiente = error * derivada(suma);
                    for (int j = 0; j < pesos.length; j++) {
                        pesos[j] += tasaAprendizaje * gradiente * x[j];
                    }
                    bias += tasaAprendizaje * gradiente;

                    if (Math.abs(error) < 0.3) correctos++;
                }

                double precision = (double) correctos / entradas.size();
                if (e % 100 == 0)
                    System.out.printf("Epoca %d - Precision: %.3f%n", e, precision);
            }
        }
    }

    // Normalizacion de valores
    private static double normalizar(double valor, double min, double max) {
        return (valor - min) / (max - min) * 2 - 1; // Escala [-1, 1]
    }

    public static void main(String[] args) {
        // Dataset: horasEstudio, asistencia, promedio, horasSueño => riesgo (1=alto, 0=bajo)
        double[][] datos = {
            {5, 0.9, 80, 6, 0},
            {2, 0.6, 55, 5, 1},
            {8, 1.0, 95, 8, 0},
            {3, 0.5, 50, 4, 1},
            {6, 0.8, 85, 7, 0},
            {1, 0.4, 40, 5, 1},
            {7, 0.9, 90, 7, 0},
            {2, 0.7, 60, 6, 1},
            {4, 0.6, 70, 6, 1},
            {5, 0.85, 88, 7, 0},
            {3, 0.55, 65, 5, 1},
            {6, 0.95, 92, 8, 0},
            {1, 0.5, 45, 5, 1},
            {4, 0.7, 75, 6, 0},
            {2, 0.6, 58, 5, 1},
            {7, 0.9, 89, 8, 0},
            {3, 0.55, 62, 5, 1},
            {8, 1.0, 96, 8, 0},
            {5, 0.8, 82, 7, 0},
            {2, 0.5, 50, 5, 1}
        };

        // Normalizamos entradas
        List<double[]> entradas = new ArrayList<>();
        List<Double> salidas = new ArrayList<>();

        for (double[] fila : datos) {
            double[] entradaNorm = {
                normalizar(fila[0], 0, 10),  // horas estudio
                normalizar(fila[1], 0, 1),   // asistencia
                normalizar(fila[2], 0, 100), // promedio
                normalizar(fila[3], 0, 10)   // horas sueño
            };
            entradas.add(entradaNorm);
            salidas.add(fila[4]);
        }

        Perceptron p = new Perceptron(4, 0.05);
        p.entrenar(entradas, salidas, 2000);

        // Evaluacion
        int correctos = 0;
        for (int i = 0; i < entradas.size(); i++) {
            double pred = p.predecir(entradas.get(i));
            double salidaBinaria = pred >= 0 ? 1 : 0;
            System.out.printf("Prediccion: %.3f -> %.0f | Esperado: %.0f%n", pred, salidaBinaria, salidas.get(i));
            if (salidaBinaria == salidas.get(i)) correctos++;
        }

        double precisionFinal = (double) correctos / entradas.size();
        System.out.printf("Precision final: %.3f%n", precisionFinal);
    }
}
