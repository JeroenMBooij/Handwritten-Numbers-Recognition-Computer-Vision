using Mnist.Logic;
using System;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;

namespace Mnist.WPF
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window, INotifyPropertyChanged
    {
        public NetworkVM NetworkVM { get; set; }

        public int SelectedMNIS
        {
            get => _selectedMNIS;
            set
            {
                if (_selectedMNIS != value)
                {
                    _selectedMNIS = value;
                    Paint_MNIS_Number();

                    OnPropertyChanged(nameof(SelectedMNIS));
                }
            }
        }
        private int _selectedMNIS;
        private int _direction = 1;


        public bool Disable { get; set; } = false;

        public int Number
        {
            get => _number;
            set
            {
                if (_number != value)
                {
                    _number = value;
                    OnPropertyChanged(nameof(Number));
                }
            }
        }
        private int _number;


        private NeuralNetwork _network;

        private Canvas _canvas;
        private ProgressBar _progressbar;

        //Amount of pixels in width and height.
        private const int dimDigit = 28;
        //Width and height of pixel.
        private const int dimPixel = 15;


        public MainWindow()
        {
            InitializeComponent();

            _canvas = (Canvas)FindName("canvas");
            _progressbar = (ProgressBar)FindName("myProgress");

            _network = new NeuralNetwork();
            NetworkVM = new NetworkVM(_network);

            _progressbar.Minimum = 0;
            _progressbar.Maximum = NetworkVM.TrainingSize;

            Paint_MNIS_Number();

            DataContext = this;

            BitmapEye = new BitmapImage();
            Loadimage(@"background.gif", BitmapEye);

        }

        public void TestNetwork_Click(object sender, RoutedEventArgs e)
        {
            if (Disable == false)
            {
                NetworkVM.Cost = _network.UpdateNetwork(SelectedMNIS).ToString();

                for (int i = 0; i < _network.OutputLayer.Outputs.Length; i++)
                    NetworkVM.Outputs[i] = _network.OutputLayer.Outputs[i].ToString("F20").TrimEnd('0');

                NetworkVM.CalculatedValue = Array.IndexOf(_network.OutputLayer.Outputs, _network.OutputLayer.Outputs.Max());
            }
            else
            {
                var dialog = new Dialog();
                dialog.Owner = this;
                dialog.ShowDialog();
            }
        }

        public void TestNetworkAll_Click(object sender, RoutedEventArgs e)
        {
            if (Disable == false)
            {
                UpdateProgressVisibility(Visibility.Visible);
                _progressbar.IsIndeterminate = true;
                Disable = true;

                int correct = 0;
                int wrong = 0;
                Task.Run(() =>
                {
                    for (int index = 0; index < NetworkVM.TestSize; index++)
                    {
                        _network.UpdateNetwork(index).ToString();

                        int calculatedValue = Array.IndexOf(_network.OutputLayer.Outputs, _network.OutputLayer.Outputs.Max());

                        int actualValue = _network.TestData.Values[index];

                        if (calculatedValue == actualValue)
                            correct++;
                        else
                            wrong++;
                    }

                    NetworkVM.Correct = correct;
                    NetworkVM.Wrong = wrong;

                    Dispatcher.BeginInvoke(new UpdateVisibilityCallback(UpdateProgressVisibility), Visibility.Collapsed);
                    Disable = false;

                });

            }
            else
            {
                var dialog = new Dialog();
                dialog.Owner = this;
                dialog.ShowDialog();
            }
        }

        public void TrainNetwork_Click(object sender, RoutedEventArgs e)
        {
            if (Disable == false)
            {
                Disable = true;
                _progressbar.IsIndeterminate = false;
                _progressbar.Maximum = NetworkVM.TrainingSize * NetworkVM.Epochs;

                UpdateProgressVisibility(Visibility.Visible);

                Task.Run(() =>
                {
                    for (int i = 0; i < NetworkVM.Epochs; i++)
                        _network.TrainNetworkMiniBatchWise(UpdateProgressBar);

                    Dispatcher.BeginInvoke(new UpdateVisibilityCallback(UpdateProgressVisibility), Visibility.Collapsed);

                    Disable = false;
                });
            }
            else
            {
                var dialog = new Dialog();
                dialog.Owner = this;
                dialog.ShowDialog();
            }
        }

        private void UpdateProgressBar(int progress)
        {
            Dispatcher.BeginInvoke(new Action(() =>
            {
                _progressbar.Value = progress;

                if (SelectedMNIS >= 9990)
                    _direction = -1;
                if (SelectedMNIS <= 10)
                    _direction = 1;

                SelectedMNIS += (10 * _direction);
            }));
        }

        private delegate void UpdateVisibilityCallback(Visibility visibility);
        private void UpdateProgressVisibility(Visibility visibility)
        {
            _progressbar.Visibility = visibility;
        }

        private void Paint_MNIS_Number()
        {
            _canvas.Children.Clear();
            if (SelectedMNIS >= _network.TestData.Values.Length)
                SelectedMNIS = _network.TestData.Values.Length - 1;
            if (SelectedMNIS < 0)
                SelectedMNIS = 0;

            for (int i = 0; i < dimDigit; i++)
            {
                for (int j = 0; j < dimDigit; j++)
                {

                    int grayScale = (int)(255.0 * (1.0 - _network.TestData.Inputs[SelectedMNIS, i + dimDigit * j]));

                    Rectangle rec = new Rectangle()
                    {
                        Width = dimPixel,
                        Height = dimPixel,
                        Fill = new SolidColorBrush(Color.FromRgb((byte)grayScale, (byte)grayScale, (byte)grayScale)),
                        Stroke = Brushes.Gray,
                        StrokeThickness = 0.5
                    };

                    _canvas.Children.Add(rec);
                    Canvas.SetLeft(rec, i * dimPixel);
                    Canvas.SetTop(rec, j * dimPixel);

                }
            }

            Number = _network.TestData.Values[SelectedMNIS];
        }


        #region PropertyChanged
        public event PropertyChangedEventHandler PropertyChanged;

        public void OnPropertyChanged([CallerMemberName] string propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
        #endregion


        #region Image
        public BitmapImage BitmapEye { get; set; }
        public void Loadimage(string imageFilePath, BitmapImage bitmap)
        {
            try
            {
                var path = $@"{Directory.GetCurrentDirectory()}\{imageFilePath}";
                var stream = File.OpenRead(path);
                bitmap.BeginInit();
                bitmap.CacheOption = BitmapCacheOption.OnLoad;
                bitmap.StreamSource = stream;
                bitmap.EndInit();
                stream.Close();
                stream.Dispose();
            }
            catch { }
        }
        #endregion
    }
}
