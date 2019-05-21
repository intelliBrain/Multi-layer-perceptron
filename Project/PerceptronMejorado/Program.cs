using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace PerceptronVideo
{
    class Program
    {
        static readonly string inputPath = @"..\..\..\DataSets\AND.csv";
        static readonly string outputPath = @"..\..\..\DataSets\salida.csv";
        static readonly string neuralNetworkPath = @"..\..\..\DataSets\NN.json";

        static readonly int inputCount = 2;
        static readonly int outputCount = 1;

        //static bool saveNetwork = true;
        static bool loadNetwork = false;

        static readonly double inputMax = 1;
        static readonly double inputMin = 0;

        static readonly double outputMax = 1;
        static readonly double outputMin = 0;

        static readonly List<double[]> input = new List<double[]>();
        static readonly List<double[]> output = new List<double[]>();

        static void ReadData()
        {
            var data = File.ReadAllText(inputPath).Replace("\r", "");//.Replace(",", ".");
            var row = data.Split(Environment.NewLine.ToCharArray());

            for (var i = 0; i < row.Length; i++)
            {
                var rowData = row[i].Split(';');

                var inputs = new double[inputCount];
                var outputs = new double[outputCount];

                for (var j = 0; j < rowData.Length; j++)
                {
                    if (j < inputCount)
                    {
                        inputs[j] = Normalize(double.Parse(rowData[j]), inputMin, inputMax);
                        // Console.WriteLine(inputs[j]);
                    }
                    else
                    {
                        outputs[j - inputCount] = Normalize(double.Parse(rowData[j]), outputMin, outputMax);
                        //Console.WriteLine(outputs[j - inputCount]);
                    }
                }

                input.Add(inputs);
                output.Add(outputs);
            }

        }

        private static double Normalize(double val, double min, double max)
        {
            return (val - min) / (max - min);
        }

        private static double InverseNormalize(double val, double min, double max)
        {
            return val * (max - min) + min;
        }

        private static void Evaluate(Perceptron p, double from, double to, double step)
        {
            var output = "";
            for (var i = from; i < to; i += step)
            {
                var res = p.Activate(new double[] { Normalize(i, inputMin, inputMax) })[0];


                output += i + ";" + InverseNormalize(res, outputMin, outputMax) + "\n";
                Console.WriteLine(i + ";" + res + "\n");
            }

            File.WriteAllText(outputPath, output);
        }


        public static void Main()
        {
            Perceptron p;

            var net_def = new int[] { inputCount, 10, 10, outputCount };
            var learning_rate = 0.3;
            var max_error = 0.0001;
            var max_iter = 1000000;

            loadNetwork = File.Exists(neuralNetworkPath);

            if (!loadNetwork)
            {
                ReadData();
                p = new Perceptron(net_def);

                while (!p.Learn(input, output, learning_rate, max_error, max_iter, neuralNetworkPath, 10000))
                {
                    p = new Perceptron(net_def);
                }
            }
            else
            {
                p = Perceptron.LoadNetwork(neuralNetworkPath);
            }

            p.SaveNetwork(neuralNetworkPath);

            //Evaluate(p, 0, 5, 0.1);


            while (true)
            {

                var inputs = new double[inputCount];
                for (var i = 0; i < inputCount;)
                {
                    Console.WriteLine($"Value #{i}: ");
                    var inputValue = Console.ReadLine();

                    if (!string.IsNullOrEmpty(inputValue))
                    {
                        if (double.TryParse(inputValue, out var input))
                        {
                            var inputNormalized = Normalize(input, inputMin, inputMax);
                            Console.WriteLine($"{input:F3} => {inputNormalized:F5}");

                            inputs[i] = inputNormalized;
                            i++;
                        }
                    }
                    else
                    {
                        goto byebye;
                    }
                }

                var outputs = p.Activate(inputs);
                //for (var i = 0; i < outputCount; i++)
                //{
                var sbInputs = new StringBuilder();
                var sbOutputs = new StringBuilder();

                foreach (var inp in inputs)
                {
                    if (sbInputs.Length > 0) sbInputs.Append(", ");
                    sbInputs.Append($"{inp:F5}");
                }

                foreach (var o in outputs)
                {
                    var outValueDenormalized = InverseNormalize(o, outputMin, outputMax);

                    if (sbOutputs.Length > 0) sbOutputs.Append(", ");
                    sbOutputs.Append($"{o:F3}");
                }

                Console.Write($"Inputs: {sbInputs} => Outputs: {sbOutputs}");
                //}

                Console.WriteLine("");
            }

        byebye:
            Console.WriteLine("bye bye!");
        }
    }
}
