using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace PrositProject
{
    public class SpectralPredictor
    {
        private InferenceSession _session;
        public SpectralPredictor(string modelPath)
        {
            _session = new InferenceSession(modelPath);
        }

        public Tensor<float> Predict(string dataPath)
        {
            var reader = new PrositInputsFileReader();
            reader.ReadCSV(dataPath);

            var peptideTensor = Tensorizer.FeaturizePeptides(reader.Peptides);
            var chargeTensor = Tensorizer.FeaturizeChargesOneHot(reader.PrecursorCharges);
            var ceTensor = Tensorizer.FeaturizeCollisionEnergies(reader.CollisionEnergies);

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("peptide_sequences", peptideTensor),
                NamedOnnxValue.CreateFromTensor("precursor_charges", chargeTensor),
                NamedOnnxValue.CreateFromTensor("normalized_collision_energies", ceTensor)
            };
            using (var results = _session.Run(inputs))
            {
                var output = results.First().AsTensor<float>();
                return output;
            }

        }

        public void Dispose()
        {
            _session.Dispose();
        }
    }
}
