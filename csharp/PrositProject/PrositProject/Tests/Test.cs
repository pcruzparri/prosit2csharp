using NUnit.Framework;

namespace PrositProject.Tests
{
    public class Tests
    {
        [Test]
        public void TestValidateHeaders_ValidHeaders_ReturnsTrue()
        {
            var cwd = System.IO.Directory.GetCurrentDirectory();
            var filePath = System.IO.Path.Combine(cwd, "Tests", "TestFiles", "ThreePeptidesInputsForSpectralPrediction.csv");

            var reader = new PrositInputsFileReader();
            var headers = System.IO.File.ReadLines(filePath).First().Split(',').ToList();
            Assert.That(reader.ValidateHeaders(headers), Is.True);
        }

        [Test]
        public void TestValidateHeaders_InvalidHeaders_ReturnsFalse()
        {
            var invalidHeaders = new List<string> { "invalid_header1", "invalid_header2" };
            var reader = new PrositInputsFileReader();
            bool isValid = reader.ValidateHeaders(invalidHeaders);
            Assert.That(isValid, Is.False);
        }

        [Test]
        public void TestPredictor_CanLoadModelAndPredict()
        {
            var cwd = System.IO.Directory.GetCurrentDirectory();
            var modelPath = System.IO.Path.Combine(cwd, "Models", "HLA_CID", "weight_192_0.16253_compatible.onnx");
            var dataPath = System.IO.Path.Combine(cwd, "Tests", "TestFiles", "ThreePeptidesInputsForSpectralPrediction.csv");
            var expectedOutputLength = 3 * 174; // 3 peptides * 174 fragment ions

            var predictor = new SpectralPredictor(modelPath);
            var results = predictor.Predict(dataPath);
            Assert.That(results, Is.Not.Null);
            Assert.That(results.Length, Is.EqualTo(expectedOutputLength));
            Assert.That(results.Dimensions[0], Is.EqualTo(3));
            Assert.That(results.Dimensions[1], Is.EqualTo(174));
        }
    }
}
