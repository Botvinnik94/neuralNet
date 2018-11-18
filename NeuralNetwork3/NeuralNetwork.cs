using System;
using System.Collections.Generic;
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

        public NeuralNetwork(int[] neuronDistribution, double learningRate) //index = layer ; value = number of neurons ;
        {
            LearningRate = learningRate;
            _neurons = new double[neuronDistribution.Length][];
            _synapses = new double[neuronDistribution.Length - 1][][];

            for (int i = 0; i < neuronDistribution.Length; i++)
            {
                _neurons[i] = new double[neuronDistribution[i]];

                if (i != neuronDistribution.Length - 1)
                {
                    _synapses[i] = new double[neuronDistribution[i]][];
                    for (int j = 0; j < neuronDistribution[i]; j++)
                    {
                        _synapses[i][j] = new double[neuronDistribution[i + 1]];
                        for (int axon = 0; axon < _synapses[i][j].Length; axon++)
                        {
                            _synapses[i][j][axon] = RandomSeed.Instance.rand.NextDouble();
                        }
                    }
                }
            }
        }

        public double[] Think(double[] inputs)
        {
            for (int i = 0; i < _neurons[0].Length; i++)
            {
                _neurons[0][i] = inputs[i];
            }

            for (int i = 1; i < _neurons.Length; i++)
            {
                for (int j = 0; j < _neurons[i].Length; j++)
                {
                    double total = 0;
                    for (int axon = 0; axon < _synapses[i - 1][j].Length; axon++)
                    {
                        total += _neurons[i - 1][axon] * _synapses[i - 1][axon][j];
                    }

                    _neurons[i][j] = Sigmoid(total);
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

        private void UpdateWeights(double[][] allDeltas)
        {
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

        public void Train(double[][] trainingInputs, double[][] trainingExpectedOutputs)
        {
            for (int i = 0; i < trainingInputs.Length; i++)
            {
                //double sumError = 0.0;
                double[] outputs = Think(trainingInputs[i]);
                //for(int j = 0; j < outputs.Length; j++)
                //{
                //    sumError += Math.Pow((trainingExpectedOutputs[i][j] - outputs[j]), 2);
                //}
                double[][] desiredChanges = PropagateError(trainingExpectedOutputs[i], outputs);
                UpdateWeights(desiredChanges);
            }
        }

        private double Sigmoid(double input)
        {
            return 1 / (1 + Math.Pow(Math.E, -input));
        }
    }
}
