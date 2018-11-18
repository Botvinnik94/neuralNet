using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork3
{
    class RandomSeed
    {
        public Random rand;

        private static readonly Lazy<RandomSeed> instance = new Lazy<RandomSeed>(() => new RandomSeed());

        private RandomSeed()
        {
            rand = new Random();
        }

        public static RandomSeed Instance
        {
            get
            {
                return instance.Value;
            }
        }
    }
}
