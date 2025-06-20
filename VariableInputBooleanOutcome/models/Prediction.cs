namespace BinaryClassification.models
{
    public class Prediction
    {
        // Original label.
        public bool Label { get; set; }
        // Predicted label from the trainer.
        public bool PredictedLabel { get; set; }
    }
}
