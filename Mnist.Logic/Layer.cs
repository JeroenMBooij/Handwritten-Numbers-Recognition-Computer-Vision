using System;

namespace Mnist.Logic
{
    public class Layer
    {
        public int NumberOfNodes { get; set; }
        public double[,] Weights { get; set; }
        public double[] Biases { get; set; }
        public double[] Outputs { get; set; }
        public double[] Sum { get; set; }

        public Layer PreviousLayer { get; set; }

        public Layer(int nNodes, Layer previousLayer = null)
        {
            NumberOfNodes = nNodes;
            PreviousLayer = previousLayer;

            if (PreviousLayer is not null)
            {
                Weights = new double[nNodes, previousLayer.NumberOfNodes];
                SetRandomWeights();

                Biases = new double[nNodes];
                SetRandomBiases();
            }

            Outputs = new double[nNodes];
        }

        const int seed = 123;
        private Random _random = new Random(seed);

        private void SetRandomBiases()
        {
            for (int i = 0; i < Biases.Length; i++)
            {
                Biases[i] = ((double)_random.Next(-10, 10) / 10);
            }
        }

        private void SetRandomWeights()
        {
            for (int i = 0; i < Weights.GetLength(0); i++)
            {
                for (int j = 0; j < Weights.GetLength(1); j++)
                {
                    Weights[i, j] = ((double)_random.Next(-10, 10) / 10);
                }
            }
        }
    }
}
