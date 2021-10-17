using System;

namespace Mnist.Logic
{
    public class NeuralMath
    {
        private NeuralNetwork _n;

        public NeuralMath(NeuralNetwork n)
        {
            _n = n;
        }

        public double CostFunction(double expected, double input)
        {
            return 0.5 * Math.Pow((expected - input), 2); ;
        }

        public double Sigmoid(double input)
        {
            return 1.0 / (1.0 + Math.Exp(-1.0 * input));
        }

        public double SigmoidPrime(double input)
        {
            // calculate the derivative of the Sigmoid function
            return Sigmoid(input) * (1 - Sigmoid(input));
        }

        public double PartialCostPartialOutput(int index, int i)
        {
            return (_n.OutputLayer.Outputs[i] - _n.MNISTData.DesiredOutput[index, i]) * SigmoidPrime(_n.OutputLayer.Sum[i]);
        }

        public double PartialCostPartialOutputBias(int index, int i)
        {
            return PartialCostPartialOutput(index, i) * 1;
        }

        public double PartialCostPartialOutputWeight(int index, int i, int j)
        {
            return PartialCostPartialOutput(index, i) * _n.HiddenLayer.Outputs[j];
        }

        public double PartialCostPartialHiddenOutput(int index, int i, int j)
        {
            return PartialCostPartialOutput(index, i) * _n.OutputLayer.Weights[i, j] * SigmoidPrime(_n.HiddenLayer.Sum[j]);
        }

        public double PartialCostPartialHiddenBias(int index, int i, int j)
        {
            return PartialCostPartialHiddenOutput(index, i, j) * 1;
        }

        public double PartialCostPartialHiddenWeight(int index, int i, int j, int k)
        {
            return PartialCostPartialHiddenOutput(index, i, j) * _n.InputLayer.Outputs[k];
        }
    }
}
