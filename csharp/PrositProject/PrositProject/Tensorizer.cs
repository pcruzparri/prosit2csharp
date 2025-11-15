using Microsoft.ML.OnnxRuntime.Tensors;

namespace PrositProject
{
    public static class Tensorizer
    {
        private static Dictionary<string, int> AAToIndex = new Dictionary<string, int>
        {
            {"A", 1}, {"C", 2}, {"D", 3}, {"E", 4}, {"F", 5},
            {"G", 6}, {"H", 7}, {"I", 8}, {"K", 9}, {"L", 10},
            {"M", 11}, {"N", 12}, {"P", 13}, {"Q", 14}, {"R", 15},
            {"S", 16}, {"T", 17}, {"V", 18}, {"W", 19}, {"Y", 20},
            {"M(ox)", 21} // Fixed methionine oxidation
        };

        private static int MaxCharges = 6;

        public static DenseTensor<float> FeaturizePeptides(List<string> peptides, int maxLength = 30)
        {
            int batchSize = peptides.Count;
            var tensor = new DenseTensor<float>(new[] { batchSize, maxLength });
            for (int i = 0; i < batchSize; i++)
            {
                var peptide = peptides[i];
                for (int j = 0; j < maxLength; j++)
                {
                    if (j < peptide.Length)
                    {
                        var aa = peptide[j].ToString();
                        if (aa == "M" && j + 4 < peptide.Length && peptide.Substring(j, 5) == "M(ox)")
                        {
                            aa = "M(ox)";
                            j += 4;
                        }
                        tensor[i, j] = AAToIndex.ContainsKey(aa) ? (float)AAToIndex[aa] : (float)AAToIndex["M(ox)"];
                    }
                    else
                    {
                        tensor[i, j] = 0f; // padding
                    }
                }
            }
            return tensor;
        }

        public static DenseTensor<float> FeaturizeChargesOneHot(List<int> charges)
        {
            int batchSize = charges.Count;
            var tensor = new DenseTensor<float>(new[] { batchSize, MaxCharges });
            for (int i = 0; i < batchSize; i++)
            {
                tensor[i, charges[i] - 1] = 1.0f;
            }
            return tensor;
        }

        public static DenseTensor<float> FeaturizeCollisionEnergies(List<float> collisionEnergies)
        {
            int batchSize = collisionEnergies.Count;
            var tensor = new DenseTensor<float>(new[] { batchSize, 1 });
            for (int i = 0; i < batchSize; i++)
            {
                tensor[i, 0] = collisionEnergies[i] / 100.0f;
            }
            return tensor;
        }
    }
}
