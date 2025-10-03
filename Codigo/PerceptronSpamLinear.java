import java.util.Random;

public class PerceptronSpamLinear {
    private double[] weights;
    private double bias;
    private double learningRate;
    private int epochs;
    private Random random;
    
    public PerceptronSpamLinear(double learningRate, int epochs) {
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.random = new Random();
    }
    
    private double linearActivation(double x) {
        /** FUNCIÓN DE ACTIVACIÓN LINEAL */
        return x;  // Simplemente retorna la entrada
    }
    
    public void train(double[][] X, double[] y) {
        int nFeatures = X[0].length;
        weights = new double[nFeatures];
        
        // Inicialización para función lineal
        for (int i = 0; i < nFeatures; i++) {
            weights[i] = random.nextDouble() * 0.3 - 0.15;
        }
        bias = random.nextDouble() * 0.3 - 0.15;
        
        System.out.println("🔧 PERCEPTRÓN SPAM - FUNCIÓN LINEAL");
        System.out.println("Configuración: LR=" + learningRate + ", Épocas=" + epochs);
        System.out.println("Función de activación: LINEAL");
        System.out.println("--------------------------------------------------");
        
        int earlyStop = epochs - 6;
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalError = 0;
            int correctPredictions = 0;
            
            for (int i = 0; i < X.length; i++) {
                // Calcular salida lineal
                double linearOutput = 0;
                for (int j = 0; j < nFeatures; j++) {
                    linearOutput += X[i][j] * weights[j];
                }
                linearOutput += bias;
                
                // FUNCIÓN LINEAL - sin transformación
                double output = linearActivation(linearOutput);
                double error = y[i] - output;
                totalError += Math.abs(error);
                
                // Clasificación: umbral 0.5 para salida lineal
                int prediction = output > 0.5 ? 1 : 0;
                if (prediction == (int)y[i]) {
                    correctPredictions++;
                }
                
                // Actualización con ruido para función lineal
                double noise = random.nextDouble() * 0.08 - 0.04;
                for (int j = 0; j < nFeatures; j++) {
                    weights[j] += learningRate * error * X[i][j] + noise;
                }
                bias += learningRate * error + noise;
            }
            
            double accuracy = (double) correctPredictions / X.length * 100;
            
            if (epoch % 15 == 0 || epoch == epochs - 1) {
                System.out.printf("Época %3d | Error: %.4f | Precisión: %.1f%%%n", 
                    epoch, totalError, accuracy);
                
                // Mostrar algunas salidas lineales
                System.out.printf("  Salidas lineales: ");
                for (int i = 0; i < Math.min(3, X.length); i++) {
                    double linearOut = 0;
                    for (int j = 0; j < nFeatures; j++) {
                        linearOut += X[i][j] * weights[j];
                    }
                    linearOut += bias;
                    System.out.printf("%.3f ", linearOut);
                }
                System.out.println();
            }
            
            // Parada temprana
            if (epoch >= earlyStop && accuracy > 80) {
                System.out.printf("⚡ Parada temprana en época %d%n", epoch);
                break;
            }
        }
    }
    
    public double[] predict(double[][] X) {
        double[] predictions = new double[X.length];
        
        for (int i = 0; i < X.length; i++) {
            double linearOutput = 0;
            for (int j = 0; j < X[i].length; j++) {
                linearOutput += X[i][j] * weights[j];
            }
            linearOutput += bias;
            
            // Clasificación con función lineal
            predictions[i] = linearOutput > 0.5 ? 1 : 0;
        }
        
        return predictions;
    }
    
    public void showLinearOutputs(double[][] X) {
        System.out.println("\n🔍 SALIDAS LINEALES (sin activación):");
        for (int i = 0; i < X.length; i++) {
            double linearOutput = 0;
            for (int j = 0; j < X[i].length; j++) {
                linearOutput += X[i][j] * weights[j];
            }
            linearOutput += bias;
            System.out.printf("Email %d: valor lineal = %.3f%n", i+1, linearOutput);
        }
    }
    
    public static void main(String[] args) {
        // Datos de spam con características continuas
        double[][] X_spam = {
            {0.2, 0.1, 0.0, 0.1, 0.0},  // no spam
            {0.8, 0.7, 1.0, 0.9, 0.8},  // spam
            {0.3, 0.2, 0.0, 0.2, 0.1},  // no spam
            {0.9, 0.8, 1.0, 0.8, 0.9},  // spam
            {0.4, 0.3, 0.0, 0.1, 0.0},  // no spam
            {0.7, 0.6, 1.0, 0.7, 0.6},  // spam
            {0.5, 0.4, 0.5, 0.4, 0.3},  // caso límite
            {0.6, 0.5, 0.6, 0.5, 0.4}   // caso límite
        };
        double[] y_spam = {0, 1, 0, 1, 0, 1, 0, 1};
        
        PerceptronSpamLinear perceptron = new PerceptronSpamLinear(0.05, 100);
        perceptron.train(X_spam, y_spam);
        
        // Mostrar salidas lineales
        perceptron.showLinearOutputs(X_spam);
        
        double[] predictions = perceptron.predict(X_spam);
        
        System.out.println("\n🎯 RESULTADOS FINALES - FUNCIÓN LINEAL");
        System.out.println("======================================");
        
        int correct = 0;
        for (int i = 0; i < predictions.length; i++) {
            String status = (predictions[i] == y_spam[i]) ? "✅" : "⚠️";
            String predictedType = predictions[i] == 1 ? "SPAM" : "NO SPAM";
            String actualType = y_spam[i] == 1 ? "SPAM" : "NO SPAM";
            
            System.out.printf("%s Email %d: Esperado=%s, Predicho=%s%n", 
                status, i+1, actualType, predictedType);
            
            if (predictions[i] == y_spam[i]) correct++;
        }
        
        double accuracy = (double) correct / predictions.length * 100;
        System.out.printf("\n📈 Precisión final: %.1f%%%n", accuracy);
        System.out.println("💡 Característica: Salida continua - Sensible a la escala de los datos");
    }
}