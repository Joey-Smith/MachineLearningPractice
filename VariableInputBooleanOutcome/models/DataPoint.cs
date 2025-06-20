namespace BinaryClassification.models
{
    public class DataPoint
    {
        public bool Label { get; set; }
        public float[] Features { get; set; }
        //public float SomeNumber { get; set; }
        //public string? SomeText { get; set; }
        public int VectorDimensions => Features?.Length ?? 0;
    }
}
