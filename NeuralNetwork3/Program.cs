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
            NeuralNetwork nn = new NeuralNetwork(new int[] { 2, 2, 2 }, 0.5);
            string[] allData = File.ReadAllLines("learningData.txt");
            double[][] trainingInputs = new double[allData.Length][];
            double[][] expectedOutputs = new double[allData.Length][];

            for (int i = 0; i < allData.Length; i++)
            {
                string[] keyVal = allData[i].Split(',');
                trainingInputs[i] = new double[] { Convert.ToDouble(keyVal[0]), Convert.ToDouble(keyVal[1]) };
                expectedOutputs[i] = new double[] { Convert.ToDouble(keyVal[2]), Convert.ToDouble(keyVal[3]) };
            }        

            Console.WriteLine("Trained for " + nn.Train(trainingInputs, expectedOutputs, 1000, 100, 0.1) + " iterations");

            Console.WriteLine();

            double[] result = nn.Think(new double[] { 0.0, 0.0 });
            Console.WriteLine("Input: [0, 0] => Output: " + result[0] + " " + result[1]);
            
            result = nn.Think(new double[] { 0.0, 1.0 });
            Console.WriteLine("Input: [0, 1] => Output: " + result[0] + " " + result[1]);

            result = nn.Think(new double[] { 1.0, 0.0 });
            Console.WriteLine("Input: [1, 0] => Output: " + result[0] + " " + result[1]);

            result = nn.Think(new double[] { 1.0, 1.0 });
            Console.WriteLine("Input: [1, 1] => Output: " + result[0] + " " + result[1]);

            Console.WriteLine();
        }
    }
}
