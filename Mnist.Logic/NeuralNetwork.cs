using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Mnist.Logic
{
    public class NeuralNetwork
    {
        #region Hyper-parameters

        //Amount of datapoints in a mini-batch.
        public int MiniBatchSize { get; set; } = 10;      

        //The learning factor èta.
        public double ETA { get; set; } = 3.0;

        //Amount of input-neuronen (layer 0).
        private const int n0 = 784;
        //Amount of hidden neuronen (layer 1).
        private const int n1 = 12;
        //Amount of output-neuronen (layer 2).
        private const int n2 = 10;                         

        #endregion

        #region Data Constants
        private const int nTraining = 60000;              
        private const int nTest = 10000;
        #endregion

        public Layer InputLayer { get; private set; }
        public Layer HiddenLayer { get; private set; }
        public Layer OutputLayer { get; private set; }           


        public MNISTData TestData { get; set; }
        public MNISTData TrainingData { get; set; }
        public MNISTData MNISTData
        { 
            get
            {
                if (_training)
                    return TrainingData;

                return TestData;
            }
        }

        //Toggle between training en testing.
        private bool _training = true;

        private NeuralMath _nnMath { get; set; }


        public NeuralNetwork()
        {
            InputLayer = new Layer(n0);
            HiddenLayer = new Layer(n1, InputLayer);
            OutputLayer = new Layer(n2, HiddenLayer);

            TestData = new MNISTData(@"Resources\mnist_test.csv", nTest, n0, n2);
            TrainingData = new MNISTData(@"Resources\mnist_train.csv", nTraining, n0, n2);

            _nnMath = new NeuralMath(this);
        }


        public double UpdateNetwork(int index)
        {
            SetInputLayer(index);

            CalculateHiddenLayer();

            CalculateOutputLayer();

            return CalculateCost(index);
        }

        #region Update Network
        private void SetInputLayer(int index)
        {
            for (int k = 0; k < n0; k++)
                InputLayer.Outputs[k] = MNISTData.Inputs[index, k];
        }

        private void CalculateHiddenLayer()
        {
            HiddenLayer.Sum = new double[n1];

            for (int j = 0; j < n1; j++)
            {
                HiddenLayer.Sum[j] = HiddenLayer.Biases[j];

                for (int k = 0; k < n0; k++)
                    HiddenLayer.Sum[j] += InputLayer.Outputs[k] * HiddenLayer.Weights[j, k];

                HiddenLayer.Outputs[j] = _nnMath.Sigmoid(HiddenLayer.Sum[j]);
            }
        }

        private void CalculateOutputLayer()
        {
            OutputLayer.Sum = new double[n2];

            for (int i = 0; i < n2; i++)
            {
                OutputLayer.Sum[i] = OutputLayer.Biases[i];

                for (int j = 0; j < n1; j++)
                    OutputLayer.Sum[i] += HiddenLayer.Outputs[j] * OutputLayer.Weights[i, j];

                OutputLayer.Outputs[i] = _nnMath.Sigmoid(OutputLayer.Sum[i]);
            }
        }

        private double CalculateCost(int index)
        {
            double cost = 0.0;
            for (int i = 0; i < n2; i++)
                cost += _nnMath.CostFunction(MNISTData.DesiredOutput[index, i], OutputLayer.Outputs[i]);

            return cost;
        }
        #endregion


        public void TrainNetworkMiniBatchGradientDescent(Action<int> UpdateCallback)
        {
            _training = true;

            for (int index = 0;  index < TrainingData.Inputs.GetLength(0);)
            {
                double[] dCdb2avg = new double[n2];
                double[,] dCdw2avg = new double[n2, n1];
                double[] dCdb1avg = new double[n1];
                double[,] dCdw1avg = new double[n1, n0];

                for (int record = 0; record < MiniBatchSize; record++, index++)
                {
                    UpdateNetwork(index);

                    // output layer gradient for biases and weights 
                    for (int i = 0; i < n2; i++)
                    {
                        var dCdb2 = _nnMath.PartialCostPartialOutputBias(index, i);
                        dCdb2avg[i] += ((double)1 / n2) * dCdb2;

                        // update weights from previous (hidden) layer
                        for (int j = 0; j < n1; j++)
                        {
                            var dCdw2 = _nnMath.PartialCostPartialOutputWeight(index, i, j);
                            dCdw2avg[i, j] += ((double)1 / n2) * dCdw2;
                        }
                    }

                    // hidden layer gradient for biases and weights 
                    for (int i = 0; i < n2; i++)
                    {
                        for (int j = 0; j < n1; j++)
                        {
                            var dCdb1 = _nnMath.PartialCostPartialHiddenBias(index, i, j);
                            dCdb1avg[j] += ((double)1 / n1) * dCdb1;

                            for (int k = 0; k < n0; k++)
                            {
                                var dCdw1 = _nnMath.PartialCostPartialHiddenWeight(index, i, j, k);
                                dCdw1avg[j, k] += ((double)1 / n1) * dCdw1;
                            }
                        }
                    }
                }

                UpdateHiddenLayer(dCdb1avg, dCdw1avg);

                UpdateOutputLayer(dCdb2avg, dCdw2avg);

                UpdateCallback(index);
            }

            _training = false;
        }

        #region Train Network
        private void UpdateHiddenLayer(double[] dCdb1avg, double[,] dCdw1avg)
        {
            for (int j = 0; j < n1; j++)
            {
                HiddenLayer.Biases[j] = HiddenLayer.Biases[j] - ETA * dCdb1avg[j];

                for (int k = 0; k < n0; k++)
                    HiddenLayer.Weights[j, k] = HiddenLayer.Weights[j, k] - ETA * dCdw1avg[j, k];
            }
        }

        private void UpdateOutputLayer(double[] dCdb2avg, double[,] dCdw2avg)
        {
            for (int i = 0; i < n2; i++)
            {
                OutputLayer.Biases[i] = OutputLayer.Biases[i] - ETA * dCdb2avg[i];

                for (int j = 0; j < n1; j++)
                    OutputLayer.Weights[i, j] = OutputLayer.Weights[i, j] - ETA * dCdw2avg[i, j];
            }
        }
        #endregion


    }
}
