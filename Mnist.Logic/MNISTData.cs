using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Mnist.Logic
{
    public class MNISTData
    {
        //Alle handgeschreven cijfers in grijstinten (0.0..1.0) uitgedrukt (training-data). Dit zijn de activatiewaarden van de input-neuronen.
        public double[,] Inputs { get; set; }

        //Alle werkelijke waarden (0..9) van de handgeschreven cijfers (training-data).
        public int[] Values { get; set; }

        //Gewenste activatiewaarden van de output-neuronen voor alle handgeschreven cijfers (training-data). Slechts één output-neuron bevat de gewenste waarde 1.0, de rest bevat de gewenste waarde 0.0.
        public double[,] DesiredOutput { get; set; }    

        public MNISTData(string filePath, int nTraining, int n0, int L)
        {
            Inputs = new double[nTraining, n0];
            Values = new int[nTraining];
            DesiredOutput = new double[nTraining, L];

            Util.loadMnistDataFromFile(filePath, Values, Inputs, DesiredOutput);
        }

    }
}
