using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Mnist.Logic
{
    public static class Util
    {
        public static void loadMnistDataFromFile(string fileName, int[] v, double[,] x, double[,] y)
        {
            //Lees alle data uit het CSV-bestand:
            string[] csvData = System.IO.File.ReadAllLines(fileName);

            //Vul de array's met alle uitgesplitste data: 28 x 28 grijstinten en één werkelijke waarde per cijfer:
            for (int cijfer = 0; cijfer < v.Length; cijfer++)
            {
                //Splits de huidige regel in velden (eerste regel bevat de kopteksten dus overslaan, vandaar cijfer + 1):
                string[] grayScales = csvData[cijfer + 1].Split(',');

                //Het eerste veld van elke regel bevat de werkelijke waarde van het handgeschreven cijfer (0..9):
                v[cijfer] = int.Parse(grayScales[0]);

                //De volgende velden in elke regel bevatten de 28 x 28 grijstinten (0..255):
                for (int pixel = 0; pixel < x.GetLength(1); pixel++)
                {
                    //Zet de grijswaarde (0..255) om in een activatiewaarde (0.0..1.0).
                    x[cijfer, pixel] = int.Parse(grayScales[1 + pixel]) / 255.0;
                }

                //Bereid ook alvast de gewenste waarden voor de output-neuronen voor:
                for (int neuron = 0; neuron < y.GetLength(1); neuron++)
                {
                    //Een output-neuron krijgt de gewenste waarde 1.0 áls de werkelijke cijferwaarde (0..9) overeenkomt met het neuron-nummer (0..9).
                    //Alle overige output-neuronen krijgen de gewenste waarde 0.0:
                    y[cijfer, neuron] = (v[cijfer] == neuron) ? 1.0 : 0.0;
                }
            }
        }
    }
}
