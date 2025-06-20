using BinaryClassification.models;
using Microsoft.ML;
using Microsoft.ML.Data;
using System.Diagnostics;

namespace BinaryClassification.logic
{
    public class LogisticRegression
    {
        static Stopwatch _stopwatch = new Stopwatch();

        private static SchemaDefinition GetSchemaDefinition(IEnumerable<DataPoint> dataPoints)
        {
            var inputSchemaDefinition = SchemaDefinition.Create(typeof(DataPoint), SchemaDefinition.Direction.Both);
            inputSchemaDefinition["Features"].ColumnType = new VectorDataViewType(NumberDataViewType.Single, dataPoints.FirstOrDefault().VectorDimensions);
            return inputSchemaDefinition;
        }

        public static void Run()
        {
            var mlContext = new MLContext(seed: 0);

            // Create a list of training data points.
            var trainingDataPoints = GenerateRandomDataPoints(5000);

            // Create a schema definition for the data points, so that ML.NET doesn't assume varvector.
            var schema = GetSchemaDefinition(trainingDataPoints);

            // Convert the list of data points to an IDataView object
            var trainingData = mlContext.Data.LoadFromEnumerable(trainingDataPoints, schema);
            //IDataView dataView = mlContext.Data.LoadFromTextFile<MyObject>(<PathToCSV>, hasHeader: true, separatorChar: ',');

            // Define the trainer.
            //var pipeline = mlContext.Transforms.Categorical.OneHotEncoding(inputColumnName: "SomeText", outputColumnName: "SomeTextEncoded")
            //    .Append(mlContext.Transforms.Concatenate("Features", "SomeTextEncoded", "SomeNumber"))
            //    .Append(mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression());
            var pipeline = mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression();

            // Train the model.
            var model = pipeline.Fit(trainingData);

            _stopwatch.Restart();

            // Create testing data. Use different random seed to make it different from training data.
            var testDataPoints = GenerateRandomDataPoints(500, seed: 123);

            var testData = mlContext.Data.LoadFromEnumerable(testDataPoints, schema);

            _stopwatch.Stop();

            // Run the model on test data set.
            var transformedTestData = model.Transform(testData);

            // Convert IDataView object to a list.
            var predictions = mlContext.Data
                .CreateEnumerable<Prediction>(transformedTestData,
                reuseRowObject: false).ToList();
            var data = mlContext.Data
                .CreateEnumerable<DataPoint>(transformedTestData,
                reuseRowObject: false).ToList();

            // Print 5 predictions.
            foreach (var p in predictions.Take(5))
                Console.WriteLine($"Label: {p.Label}, "
                    + $"Prediction: {p.PredictedLabel}");
            foreach (var p in data.Take(5))
                Console.WriteLine($"Label: {p.Label}, "
                    + $"# of Features: {p.Features.Length}");
            //foreach (var p in data.Take(5))
            //    Console.WriteLine($"Label: {p.Label}, "
            //        + $"SomeNumber: {p.SomeNumber}, " + $"SomeText: {p.SomeText}");

            // Expected output:
            //   Label: True, Prediction: True
            //   Label: False, Prediction: True
            //   Label: True, Prediction: True
            //   Label: True, Prediction: True
            //   Label: False, Prediction: False

            // Evaluate the overall metrics.
            var metrics = mlContext.BinaryClassification
                .Evaluate(transformedTestData);

            PrintMetrics(metrics);

            // Expected output:
            //   Accuracy: 0.88
            //   AUC: 0.96
            //   F1 Score: 0.87
            //   Negative Precision: 0.90
            //   Negative Recall: 0.87
            //   Positive Precision: 0.86
            //   Positive Recall: 0.89
            //   Log Loss: 0.38
            //   Log Loss Reduction: 0.62
            //   Entropy: 1.00
            //
            //   TEST POSITIVE RATIO:    0.4760 (238.0/(238.0+262.0))
            //   Confusion table
            //             ||======================
            //   PREDICTED || positive | negative | Recall
            //   TRUTH     ||======================
            //    positive ||      212 |       26 | 0.8908
            //    negative ||       35 |      227 | 0.8664
            //             ||======================
            //   Precision ||   0.8583 |   0.8972 |
        }

        private static IEnumerable<DataPoint> GenerateRandomDataPoints(int count,
            int seed = 0)

        {
            var random = new Random(seed);
            float randomFloat() => (float)random.NextDouble();
            for (int i = 0; i < count; i++)
            {
                var label = randomFloat() > 0.5f;
                //var someText = label ? "Dog" : DateTime.Now.AddDays(i).ToShortDateString();
                yield return new DataPoint
                {
                    Label = label,
                    // Create random features that are correlated with the label.
                    // For data points with false label, the feature values are
                    // slightly increased by adding a constant.
                    Features = Enumerable.Repeat(label, 5000)
                        .Select(x => x ? randomFloat() : randomFloat() +
                        0.1f).ToArray()
                    //SomeNumber = randomFloat() + 0.1f,
                    //SomeText = someText

                };
            }
        }

        // Class used to capture predictions.


        // Pretty-print BinaryClassificationMetrics objects.
        private static void PrintMetrics(BinaryClassificationMetrics metrics)
        {
            Console.WriteLine($"Elapsed time: {_stopwatch.Elapsed}");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:F2}");
            Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve:F2}");
            Console.WriteLine($"F1 Score: {metrics.F1Score:F2}");
            Console.WriteLine($"Negative Precision: " +
                $"{metrics.NegativePrecision:F2}");

            Console.WriteLine($"Negative Recall: {metrics.NegativeRecall:F2}");
            Console.WriteLine($"Positive Precision: " +
                $"{metrics.PositivePrecision:F2}");

            Console.WriteLine($"Positive Recall: {metrics.PositiveRecall:F2}\n");
            Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());
        }
    }
}