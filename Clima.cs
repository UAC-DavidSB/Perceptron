using System;
using System.Collections.Generic;

namespace PerceptronClima
{
    class Perceptron
    {
        private double[] pesos;
        private double bias;
        private double tasaAprendizaje;
        private string funcionActivacion;

        public Perceptron(int entradas, double tasaAprendizaje = 0.1, string funcionActivacion = "sigmoid")
        {
            this.pesos = new double[entradas];
            Random rand = new Random();

            for (int i = 0; i < entradas; i++)
                this.pesos[i] = rand.NextDouble() * 2 - 1; // valores entre -1 y 1

            this.bias = rand.NextDouble() * 2 - 1;
            this.tasaAprendizaje = tasaAprendizaje;
            this.funcionActivacion = funcionActivacion;
        }

        private double Activar(double x)
        {
            return funcionActivacion switch
            {
                "lineal" => x,
                "escalon" => x >= 0 ? 1 : 0,
                "sigmoid" => 1.0 / (1.0 + Math.Exp(-x)),
                "relu" => Math.Max(0, x),
                "tanh" => Math.Tanh(x),
                _ => 1.0 / (1.0 + Math.Exp(-x))
            };
        }

        public double Predecir(double[] entradas)
        {
            double suma = bias;
            for (int i = 0; i < entradas.Length; i++)
                suma += entradas[i] * pesos[i];

            return Activar(suma);
        }

        public void Entrenar(List<(double[] entradas, double salida)> datos, int epocas = 20)
        {
            for (int e = 0; e < epocas; e++)
            {
                double errorTotal = 0;
                foreach (var dato in datos)
                {
                    double salidaPredicha = Predecir(dato.entradas);
                    double error = dato.salida - salidaPredicha;
                    errorTotal += Math.Abs(error);

                    // Actualizar pesos
                    for (int i = 0; i < pesos.Length; i++)
                        pesos[i] += tasaAprendizaje * error * dato.entradas[i];

                    bias += tasaAprendizaje * error;
                }

                Console.WriteLine($"Epoca {e + 1} - Error total: {errorTotal:F4}");
            }
        }
    }

    class Program
    {
        static void Main(string[] args)
        {
            var datosEntrenamiento = new List<(double[] entradas, double salida)>
            {
                (new double[]{18, 90, 1006}, 1),
                (new double[]{20, 85, 1007}, 1),
                (new double[]{22, 80, 1008}, 1),
                (new double[]{24, 75, 1010}, 0),
                (new double[]{26, 65, 1012}, 0),
                (new double[]{28, 60, 1013}, 0),
                (new double[]{30, 50, 1015}, 0),
                (new double[]{19, 88, 1005}, 1),
                (new double[]{21, 82, 1008}, 1),
                (new double[]{23, 70, 1010}, 0),
                (new double[]{25, 68, 1011}, 0),
                (new double[]{27, 55, 1014}, 0),
                (new double[]{17, 92, 1004}, 1),
                (new double[]{18, 89, 1006}, 1),
                (new double[]{29, 58, 1015}, 0),
                (new double[]{31, 48, 1016}, 0),
                (new double[]{20, 83, 1007}, 1),
                (new double[]{22, 78, 1009}, 0),
                (new double[]{24, 72, 1011}, 0),
                (new double[]{19, 87, 1006}, 1)
            };



            // Normalizacion simple (opcional)
            for (int i = 0; i < datosEntrenamiento.Count; i++)
            {
                for (int j = 0; j < datosEntrenamiento[i].entradas.Length; j++)
                    datosEntrenamiento[i].entradas[j] /= 100.0; // Escalar valores
            }

            var perceptron = new Perceptron(3, 0.3, "escalon");

            Console.WriteLine("Entrenando perceptron para prediccion del clima...");
            perceptron.Entrenar(datosEntrenamiento, 200);

            Console.WriteLine("\n=== Pruebas ===");
            foreach (var dato in datosEntrenamiento)
            {
                double pred = perceptron.Predecir(dato.entradas);
                Console.WriteLine($"Entrada: [{string.Join(", ", dato.entradas)}] => Prediccion: {Math.Round(pred, 3)} (Esperado: {dato.salida})");
            }

            Console.WriteLine("\nEntrenamiento finalizado.");
        }
    }
}
