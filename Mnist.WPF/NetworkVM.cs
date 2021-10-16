using Mnist.Logic;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace Mnist.WPF
{
    public class NetworkVM : INotifyPropertyChanged
    {
        public double ETA
        {
            get => _network.ETA;
            set
            {
                if(_network.ETA != value)
                {
                    _network.ETA = value;

                    OnPropertyChanged(nameof(ETA));
                }
            }
        }

        public int MiniBatchSize
        {
            get => _network.MiniBatchSize;
            set
            {
                if (_network.MiniBatchSize != value)
                {
                    _network.MiniBatchSize = value;

                    OnPropertyChanged(nameof(MiniBatchSize));
                }
            }
        }

        public int Epochs
        {
            get => _epochs;
            set
            {
                if (_epochs != value)
                {
                    _epochs = value;

                    OnPropertyChanged(nameof(Epochs));
                }
            }
        }

        private int _epochs;

        public int TrainingSize 
        { 
            get => _network.TrainingData.Values.Length;
        }

        public int TestSize
        {
            get => _network.TestData.Values.Length;
        }

        public ObservableCollection<string> Outputs { get; set; } = new ObservableCollection<string>();

        public string Cost
        {
            get => _cost;
            set
            {
                if (_cost != value)
                {
                    _cost = value;
                    OnPropertyChanged(nameof(Cost));
                }
            }
        }
        private string _cost;

        public int? CalculatedValue
        {
            get => _calculatedValue;
            set
            {
                if (_calculatedValue != value)
                {
                    _calculatedValue = value;
                    OnPropertyChanged(nameof(CalculatedValue));
                }
            }
        }
        private int? _calculatedValue;

        public int? Correct 
        {
            get => _correct;
            set
            {
                if (_correct != value)
                {
                    _correct = value;
                    OnPropertyChanged(nameof(Correct));
                }
            }
        }
        private int? _correct;
        public int? Wrong
        {
            get => _wrong;
            set
            {
                if (_wrong != value)
                {
                    _wrong = value;
                    OnPropertyChanged(nameof(Wrong));
                }
            }
        }
        private int? _wrong;

        public NetworkVM(NeuralNetwork network)
        {
            _network = network;

            for (int i = 0; i < _network.OutputLayer.NumberOfNodes; i++)
            {
                Outputs.Add(string.Empty);
            }
        }

        public event PropertyChangedEventHandler PropertyChanged;

        public void OnPropertyChanged([CallerMemberName] string propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        private readonly NeuralNetwork _network;
    }
}
