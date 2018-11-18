using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork3
{
    class Program
    {
        static void Main(string[] args)
        {
            NeuralNetwork nn = new NeuralNetwork(new int[] { 1, 1, 1 }, 0.1);
            string[] allData = File.ReadAllLines("learningData.txt");
            double[][] trainingInputs = new double[allData.Length][];
            double[][] expectedOutputs = new double[allData.Length][];

            for (int i = 0; i < allData.Length; i++)
            {
                string[] keyVal = allData[i].Split(',');
                trainingInputs[i] = new double[] { Convert.ToDouble(keyVal[0]) };
                expectedOutputs[i] = new double[] { Convert.ToDouble(keyVal[1]) };
            }

            Console.WriteLine();

            for (int i = 0; i < 1; i++)
            {
                nn.Train(trainingInputs, expectedOutputs);
                Console.WriteLine("w1 = " + nn.Synapses[0][0][0] + "\tw2 = " + nn.Synapses[1][0][0]);
            }

            Console.WriteLine("========================");
            double result = nn.Think(new double[] { 1.0 })[0];
            Console.WriteLine("Input: 1.0 => Hidden: " + nn.Neurons[1][0] + " => Output: " + result);
            result = nn.Think(new double[] { 0.0 })[0];
            Console.WriteLine("Input: 0.0 => Hidden: " + nn.Neurons[1][0] + " => Output: " + result);
            Console.WriteLine();
        }
    }
}
