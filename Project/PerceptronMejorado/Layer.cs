using Newtonsoft.Json;
using System;
using System.Collections.Generic;

namespace PerceptronVideo
{
    public class Layer
    {
        [JsonProperty("neurons")]
        public List<Neuron> Neurons { get; private set; }

        [JsonProperty("numberOfNeurons")]
        public int NumberOfNeurons { get; private set; }

        [JsonProperty("outputs")]
        public double[] Outputs { get; private set; }

        [JsonConstructor]
        private Layer(List<Neuron> neurons, double[] outputs)
        {
            Neurons = neurons;
            NumberOfNeurons = neurons.Count;
            Outputs = outputs;
        }

        public Layer(int numberOfNeurons, int numberOfInputs, Random r)
        {
            NumberOfNeurons = numberOfNeurons;
            Neurons = new List<Neuron>();

            for (var i = 0; i < NumberOfNeurons; i++)
            {
                Neurons.Add(new Neuron(numberOfInputs, r));
            }
        }

        public double[] Activate(double[] inputs)
        {
            var outputs = new List<double>();

            for (var i = 0; i < NumberOfNeurons; i++)
            {
                outputs.Add(Neurons[i].Activate(inputs));
            }

            Outputs = outputs.ToArray();
            return outputs.ToArray();
        }
    }
}
