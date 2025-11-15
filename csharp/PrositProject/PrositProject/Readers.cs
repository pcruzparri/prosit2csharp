namespace PrositProject
{
    public class PrositInputsFileReader
    {
        public List<string> Peptides;
        public List<int> PrecursorCharges;
        public List<float> CollisionEnergies;
        public List<string> Headers;
        public List<string> ALLOWED_HEADERS = new List<string>
        {
            "peptide_sequence",
            "precursor_charge",
            "normalized_collision_energy"
        };
        public PrositInputsFileReader()
        {
            Peptides = new List<string>();
            PrecursorCharges = new List<int>();
            CollisionEnergies = new List<float>();
            Headers = new List<string>();
        }

        public void ReadCSV(string filePath)
        {
            var lines = System.IO.File.ReadAllLines(filePath);
            var parsedHeaders = lines.First().Split(',').ToList();
            if (!ValidateHeaders(parsedHeaders))
            {
                throw new Exception("Input file contains invalid headers.");
            }
            Headers = parsedHeaders;

            foreach (var line in lines.Skip(1))
            {
                var parts = line.Split(',');
                var peptideIndex = Headers.IndexOf(ALLOWED_HEADERS[0]);
                var chargeIndex = Headers.IndexOf(ALLOWED_HEADERS[1]);
                var ceIndex = Headers.IndexOf(ALLOWED_HEADERS[2]);

                Peptides.Add(parts[peptideIndex]);
                PrecursorCharges.Add(int.Parse(parts[chargeIndex]));
                CollisionEnergies.Add(float.Parse(parts[ceIndex]));
            }
        }

        public bool ValidateHeaders(List<string> headers)
        {
            bool isValid = ALLOWED_HEADERS.All(headers.Contains);
            return isValid;
        }
    }
}
