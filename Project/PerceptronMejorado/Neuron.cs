using Newtonsoft.Json;
using System;

namespace PerceptronVideo
{
    public class Neuron
    {
        [JsonProperty("lastActivation")]
        public double LastActivation { get; private set; }

        [JsonProperty("weights")]
        public double[] Weights { get; private set; }

        [JsonProperty("bias")]
        public double Bias { get; set; }

        [JsonConstructor]
        private Neuron(double lastActivation, double[] weights, double bias)
        {
            LastActivation = lastActivation;
            Weights = weights;
            Bias = bias;
        }

        public Neuron(int numberOfInputs, Random r)
        {
            Bias = 10 * r.NextDouble() - 5;
            Weights = new double[numberOfInputs];

            for (var i = 0; i < numberOfInputs; i++)
            {
                Weights[i] = 10 * r.NextDouble() - 5;
            }
        }

        public double Activate(double[] inputs)
        {
            var activation = Bias;

            for (var i = 0; i < Weights.Length; i++)
            {
                activation += Weights[i] * inputs[i];
            }

            LastActivation = activation;
            return Sigmoid(activation);
        }

        public static double Sigmoid(double input)
        {
            return 1 / (1 + Math.Exp(-input));
        }

        public static double SigmoidDerivated(double input)
        {
            var y = Sigmoid(input);
            return y * (1 - y);
        }
    }
}
