using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.Serialization;

namespace PerceptronVideo
{
    public class Perceptron
    {
        [JsonProperty("layers", Order = 1)]
        public List<Layer> Layers { get; private set; }

        [JsonProperty("sigmas", Order = 2)]
        public List<double[]> Sigmas { get; private set; }

        [JsonProperty("deltas", Order = 3)]
        public List<double[,]> Deltas { get; private set; }

        [JsonConstructor]
        private Perceptron(List<Layer> layers, List<double[]> sigmas, List<double[,]> deltas)
        {
            Layers = layers;
            Sigmas = sigmas;
            Deltas = deltas;
        }

        public Perceptron(int[] neuronsPerLayer)
        {
            var r = new Random();
            Layers = new List<Layer>();

            for (var i = 0; i < neuronsPerLayer.Length; i++)
            {
                var layer = new Layer(neuronsPerLayer[i], i == 0 ? neuronsPerLayer[i] : neuronsPerLayer[i - 1], r);
                Layers.Add(layer);
            }
        }

        public double[] Activate(double[] inputs)
        {
            var outputs = new double[0];

            for (var i = 1; i < Layers.Count; i++)
            {
                outputs = Layers[i].Activate(inputs);
                inputs = outputs;
            }

            return outputs;
        }

        private double IndividualError(double[] realOutput, double[] desiredOutput)
        {
            double err = 0;

            for (var i = 0; i < realOutput.Length; i++)
            {
                err += Math.Pow(realOutput[i] - desiredOutput[i], 2);
            }

            return err;
        }

        private double GeneralError(List<double[]> input, List<double[]> desiredOutput)
        {
            double err = 0;

            for (var i = 0; i < input.Count; i++)
            {
                err += IndividualError(Activate(input[i]), desiredOutput[i]);
            }

            return err;
        }

        [JsonIgnore]
        List<string> log;
        public bool Learn(List<double[]> input, List<double[]> desiredOutput, double alpha, double maxError, int maxIterations, string networkFilePath = null, int saveNetworkAfterIterations = 1)
        {
            double err = 99999;
            log = new List<string>();

            var iteration = maxIterations;
            while (err > maxError)
            {
                ApplyBackPropagation(input, desiredOutput, alpha);
                err = GeneralError(input, desiredOutput);

                if ((iteration - maxIterations) % 1000 == 0)
                {
                    Console.WriteLine(err + " iterations: " + (iteration - maxIterations));
                }

                if (networkFilePath != null)
                {
                    if ((iteration - maxIterations) % saveNetworkAfterIterations == 0)
                    {
                        SaveNetwork(networkFilePath);
                        Console.WriteLine("Save net to " + networkFilePath);
                    }
                }

                log.Add(err.ToString());
                maxIterations--;

                if (Console.KeyAvailable)
                {
                    File.WriteAllLines(@"LogTail.txt", log.ToArray());
                    return true;
                }

                if (maxIterations <= 0)
                {
                    Console.WriteLine("MINIMO LOCAL");
                    File.WriteAllLines(@"LogTail.txt", log.ToArray());
                    return false;
                }
            }

            File.WriteAllLines(@"LogTail.txt", log.ToArray());
            return true;
        }

        private void SetSigmas(double[] desiredOutput)
        {
            Sigmas = new List<double[]>();
            for (var i = 0; i < Layers.Count; i++)
            {
                Sigmas.Add(new double[Layers[i].NumberOfNeurons]);
            }
            for (var i = Layers.Count - 1; i >= 0; i--)
            {
                for (var j = 0; j < Layers[i].NumberOfNeurons; j++)
                {
                    if (i == Layers.Count - 1)
                    {
                        var y = Layers[i].Neurons[j].LastActivation;
                        Sigmas[i][j] = (Neuron.Sigmoid(y) - desiredOutput[j]) * Neuron.SigmoidDerivated(y);
                    }
                    else
                    {
                        var sum = 0.0;
                        for (var k = 0; k < Layers[i + 1].NumberOfNeurons; k++)
                        {
                            sum += Layers[i + 1].Neurons[k].Weights[j] * Sigmas[i + 1][k];
                        }

                        Sigmas[i][j] = Neuron.SigmoidDerivated(Layers[i].Neurons[j].LastActivation) * sum;
                    }
                }
            }
        }

        private void SetDeltas()
        {
            Deltas = new List<double[,]>();

            for (var i = 0; i < Layers.Count; i++)
            {
                Deltas.Add(new double[Layers[i].NumberOfNeurons, Layers[i].Neurons[0].Weights.Length]);
            }
        }

        private void AddDelta()
        {
            for (var i = 1; i < Layers.Count; i++)
            {
                for (var j = 0; j < Layers[i].NumberOfNeurons; j++)
                {
                    for (var k = 0; k < Layers[i].Neurons[j].Weights.Length; k++)
                    {
                        Deltas[i][j, k] += Sigmas[i][j] * Neuron.Sigmoid(Layers[i - 1].Neurons[k].LastActivation);
                    }
                }
            }
        }

        private void UpdateBias(double alpha)
        {
            for (var i = 0; i < Layers.Count; i++)
            {
                for (var j = 0; j < Layers[i].NumberOfNeurons; j++)
                {
                    Layers[i].Neurons[j].Bias -= alpha * Sigmas[i][j];
                }
            }
        }

        private void UpdateWeights(double alpha)
        {
            for (var i = 0; i < Layers.Count; i++)
            {
                for (var j = 0; j < Layers[i].NumberOfNeurons; j++)
                {
                    for (var k = 0; k < Layers[i].Neurons[j].Weights.Length; k++)
                    {
                        Layers[i].Neurons[j].Weights[k] -= alpha * Deltas[i][j, k];
                    }
                }
            }
        }

        private void ApplyBackPropagation(List<double[]> input, List<double[]> desiredOutput, double alpha)
        {
            SetDeltas();

            for (var i = 0; i < input.Count; i++)
            {
                Activate(input[i]);
                SetSigmas(desiredOutput[i]);
                UpdateBias(alpha);
                AddDelta();
            }

            UpdateWeights(alpha);
        }

        public void SaveNetwork(string neuralNetworkPath)
        {
            try
            {
                var json = JsonConvert.SerializeObject(this, Formatting.Indented);

                if (File.Exists(neuralNetworkPath))
                {
                    File.Delete(neuralNetworkPath);
                }

                File.WriteAllText(neuralNetworkPath, json);
            }
            catch (SerializationException ex)
            {
                Console.WriteLine("Failed to serialize. Reason: " + ex.Message);
            }


            //var fs = new FileStream(neuralNetworkPath, FileMode.Create);
            //var formatter = new BinaryFormatter();

            //try
            //{
            //    formatter.Serialize(fs, this);
            //}
            //catch (SerializationException e)
            //{
            //    Console.WriteLine("Failed to serialize. Reason: " + e.Message);
            //    throw;
            //}
            //finally
            //{
            //    fs.Close();
            //}
        }

        public static Perceptron LoadNetwork(string neuralNetworkPath)
        {
            Perceptron p = null;

            try
            {
                if (File.Exists(neuralNetworkPath))
                {
                    var json = File.ReadAllText(neuralNetworkPath);
                    p = JsonConvert.DeserializeObject<Perceptron>(json);
                }
            }
            catch (SerializationException ex)
            {
                Console.WriteLine("Failed to deserialize. Reason: " + ex.Message);
            }

            return p;


            //var fs = new FileStream(neuralNetworkPath, FileMode.Open);
            //Perceptron p = null;

            //try
            //{
            //    var formatter = new BinaryFormatter();

            //    // Deserialize the hashtable from the file and 
            //    // assign the reference to the local variable.
            //    p = (Perceptron)formatter.Deserialize(fs);
            //}
            //catch (SerializationException e)
            //{
            //    Console.WriteLine("Failed to deserialize. Reason: " + e.Message);
            //    throw;
            //}
            //finally
            //{
            //    fs.Close();
            //}

            //return p;
        }
    }
}
