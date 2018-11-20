using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork3
{
    class NeuralNetwork
    {
        private double _learningRate;
        public double LearningRate
        {
            get
            {
                return _learningRate;
            }
            set
            {
                if (value > 1.0)
                {
                    _learningRate = 1.0;
                }
                else if (value < 0.0)
                {
                    _learningRate = 0.0;
                }
                else
                {
                    _learningRate = value;
                }
            }
        }

        double[][][] _synapses; //[layer] [input neuron] [output neuron]
        double[][] _neurons; //[layer] [neuron]
        double[][] _biases; //[layer] [neuron]

        public double[][][] Synapses
        {
            get
            {
                return _synapses;
            }
        }

        public double[][] Neurons
        {
            get
            {
                return _neurons;
            }
        }

        public double[][] Biases
        {
            get
            {
                return _biases;
            }
        }

        public NeuralNetwork(int[] neuronDistribution, double learningRate) //index = layer ; value = number of neurons ;
        {
            LearningRate = learningRate;
            _neurons = new double[neuronDistribution.Length][];
            _synapses = new double[neuronDistribution.Length - 1][][];
            _biases = new double[neuronDistribution.Length - 1][];

            for (int i = 0; i < neuronDistribution.Length; i++)
            {
                _neurons[i] = new double[neuronDistribution[i]];
                if(i != 0)
                {
                    _biases[i - 1] = new double[neuronDistribution[i]];
                }                

                if (i != neuronDistribution.Length - 1)
                {
                    _synapses[i] = new double[neuronDistribution[i]][];
                    for (int j = 0; j < neuronDistribution[i]; j++)
                    {
                        _synapses[i][j] = new double[neuronDistribution[i + 1]];
                        for (int axon = 0; axon < _synapses[i][j].Length; axon++)
                        {
                            _synapses[i][j][axon] = RandomSeed.Instance.rand.NextDouble() * RandomSeed.Instance.rand.Next(-1, 2);
                        }
                    }
                }
            }
        }

        public void SaveBrain()
        {
            FileStream file = File.OpenWrite(DateTime.Now.Day.ToString() + 
                                             DateTime.Now.Hour.ToString() + 
                                             DateTime.Now.Minute.ToString() + 
                                             DateTime.Now.Second.ToString() + 
                                             "_neuralNet.txt");
            StreamWriter sw = new StreamWriter(file);

            sw.WriteLine("BEGIN SYNAPSES");
            for (int layer = 0; layer < _synapses.Length; layer++)
            {
                for(int neuron = 0; neuron < _synapses[layer].Length; neuron++)
                {
                    for(int axon = 0; axon < _synapses[layer][neuron].Length; axon++)
                    {
                        sw.Write(_synapses[layer][neuron][axon].ToString());
                        if(axon != _synapses[layer][neuron].Length - 1 || neuron != _synapses[layer].Length - 1)
                        {
                            sw.Write(";");
                        }
                    }
                }
                sw.WriteLine();
            }

            sw.WriteLine("BEGIN BIASES");
            for(int layer = 0; layer < _biases.Length; layer++)
            {
                for(int neuron = 0; neuron < _biases[layer].Length; neuron++)
                {
                    sw.Write(_biases[layer][neuron].ToString());
                    if(neuron != _biases[layer].Length - 1)
                    {
                        sw.Write(";");
                    }
                }
                sw.WriteLine();
            }

            sw.Close();
        }

        public bool LoadBrain(string path)
        {
            int state = 0;
            int layer = 0;
            string[] allLines;

            try
            {
                allLines = File.ReadAllLines(path);
            }
            catch(IOException)
            {
                return false;
            }

            for (int i = 0; i < allLines.Length; i++)
            {
                if(allLines[i].Contains("BEGIN SYNAPSES"))
                {
                    state = 1;
                    layer = 0;
                }
                else if(allLines[i].Contains("BEGIN BIASES"))
                {
                    state = 2;
                    layer = 0;
                }
                else
                {
                    switch (state)
                    {
                        case 0:
                            break;
                        case 1:
                            string[] rawAxons = allLines[i].Split(';');
                            int neuron = 0;
                            int indexAxon = 0;
                            for(int axon = 0; axon < rawAxons.Length; axon++)
                            {
                                if(indexAxon == _synapses[layer].Length)
                                {
                                    neuron++;
                                    indexAxon = 0;
                                }
                                try
                                {
                                    _synapses[layer][neuron][indexAxon] = Convert.ToDouble(rawAxons[axon]);
                                }
                                catch (FormatException) { }
                                indexAxon++;
                            }
                            layer++;
                            break;
                        case 2:
                            string[] rawBiases = allLines[i].Split(';');
                            for(int bias = 0; bias < rawBiases.Length; bias++)
                            {
                                try
                                {
                                    _biases[layer][bias] = Convert.ToDouble(rawBiases[bias]);
                                }
                                catch (FormatException) { }
                            }
                            layer++;
                            break;
                    }
                }
            }

            return true;
        }

        public double[] Think(double[] inputs)
        {
            for (int i = 0; i < _neurons[0].Length; i++)
            {
                _neurons[0][i] = inputs[i];
            }

            for (int layer = 1; layer < _neurons.Length; layer++)
            {
                for (int neuron = 0; neuron < _neurons[layer].Length; neuron++)
                {
                    double total = 0;
                    //for (int axon = 0; axon < _synapses[layer - 1][neuron].Length; axon++)
                    //{
                    //    total += _neurons[layer - 1][axon] * _synapses[layer - 1][axon][neuron];
                    //}
                    for(int prevAxon = 0; prevAxon < _synapses[layer - 1].Length; prevAxon++)
                    {
                        total += _neurons[layer - 1][prevAxon] * _synapses[layer - 1][prevAxon][neuron];
                    }

                    total += _biases[layer - 1][neuron];
                    _neurons[layer][neuron] = Sigmoid(total);
                }
            }

            return _neurons[_neurons.Length - 1];
        }

        private double[][] PropagateError(double[] desiredOutput, double[] currentOutput)
        {
            double[][] allDeltas = new double[_neurons.Length - 1][];
            double[] deltas = new double[currentOutput.Length];

            for (int i = 0; i < deltas.Length; i++)
            {
                deltas[i] = (desiredOutput[i] - currentOutput[i]) * (currentOutput[i] * (1.0 - currentOutput[i]));
            }

            allDeltas[_neurons.Length - 2] = deltas;
            BackPropagation(allDeltas, deltas, _neurons.Length - 2);
            return allDeltas;
        }

        private void BackPropagation(double[][] allDeltas, double[] deltasLastLayer, int layer)
        {
            if (layer == 0)
            {
                return;
            }
            else
            {
                double[] deltas = new double[_neurons[layer].Length];

                for (int neuron = 0; neuron < _neurons[layer].Length; neuron++)
                {
                    double error = 0.0;
                    for (int nextNeuron = 0; nextNeuron < _neurons[layer + 1].Length; nextNeuron++)
                    {
                        error += (_synapses[layer][neuron][nextNeuron] * deltasLastLayer[nextNeuron]);
                    }

                    deltas[neuron] = error * _neurons[layer][neuron] * (1.0 - _neurons[layer][neuron]);
                }

                allDeltas[layer - 1] = deltas;
                BackPropagation(allDeltas, deltas, layer - 1);
            }
        }

        private void UpdateWeightsAndBiases(double[][] allDeltas)
        {
            for (int layer = 0; layer < _biases.Length; layer++)
            {
                for(int neuron = 0; neuron < _biases[layer].Length; neuron++)
                {
                    _biases[layer][neuron] += _learningRate * allDeltas[layer][neuron];
                }
            }

            for (int layer = 0; layer < _synapses.Length; layer++)
            {
                for (int neuron = 0; neuron < _synapses[layer].Length; neuron++)
                {
                    for (int axon = 0; axon < _synapses[layer][neuron].Length; axon++)
                    {
                        _synapses[layer][neuron][axon] += _learningRate * allDeltas[layer][axon] * _neurons[layer][neuron];
                    }
                }
            }
        }

        public int Train(double[][] trainingInputs, double[][] trainingExpectedOutputs, int maxEpochs, int batchPerEpoch, double minError)
        {
            int i = 0;
            int iterations = 1;
            int epoch = 1;
            int epochIteration = 1;
            double error = 0.0;

            while (true)
            {
                double[] outputs = Think(trainingInputs[i]);

                for(int j = 0; j < outputs.Length; j++)
                {
                    error += Math.Pow((trainingExpectedOutputs[i][j] - outputs[j]), 2); 
                }

                if(epochIteration == batchPerEpoch)
                {
                    if (minError > (error / batchPerEpoch))
                        return iterations;

                    epochIteration = 0;
                    error = 0;
                    ++epoch;
                    if (epoch > maxEpochs)
                        return iterations;
                }              

                double[][] desiredChanges = PropagateError(trainingExpectedOutputs[i], outputs);
                UpdateWeightsAndBiases(desiredChanges);

                ++epochIteration;
                ++iterations;
                if (++i >= trainingInputs.Length) i = 0;
            }
        }

        private double Sigmoid(double input)
        {
            return 1 / (1 + Math.Pow(Math.E, -input));
        }
    }
}
