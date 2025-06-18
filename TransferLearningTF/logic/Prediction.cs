using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TransferLearningTF.models;
using TransferLearningTF.data;
using Microsoft.ML.Transforms;

namespace TransferLearningTF.logic
{
    public class Prediction
    {
        string _assetsPath, _imagesFolder, _trainTagsTsv, _testTagsTsv, _predictSingleImage, _inceptionTensorFlowModel;

        public Prediction()
        {
            _assetsPath = Path.Combine(Environment.CurrentDirectory, "assets");
            _imagesFolder = Path.Combine(_assetsPath, "images");
            _trainTagsTsv = Path.Combine(_imagesFolder, "tags.tsv");
            _testTagsTsv = Path.Combine(_imagesFolder, "test-tags.tsv");
            _predictSingleImage = Path.Combine(_imagesFolder, "p4.jpg");
            _inceptionTensorFlowModel = Path.Combine(_assetsPath, "inception", "tensorflow_inception_graph.pb");
        }

        public ITransformer GenerateModel(MLContext mlContext)
        {
            // Change the type of 'pipeline' to 'EstimatorChain<KeyToValueMappingTransformer>' for improved performance
            EstimatorChain<KeyToValueMappingTransformer> pipeline = mlContext.Transforms.LoadImages(outputColumnName: "input", imageFolder: _imagesFolder, inputColumnName: nameof(ImageData.ImagePath))
                .Append(mlContext.Transforms.ResizeImages(outputColumnName: "input", imageWidth: InceptionSettings.ImageWidth, imageHeight: InceptionSettings.ImageHeight, inputColumnName: "input"))
                .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "input", interleavePixelColors: InceptionSettings.ChannelsLast, offsetImage: InceptionSettings.Mean))
                .Append(mlContext.Model.LoadTensorFlowModel(_inceptionTensorFlowModel)
                .ScoreTensorFlowModel(outputColumnNames: new[] { "softmax2_pre_activation" }, inputColumnNames: new[] { "input" }, addBatchDimensionInput: true))
                .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "LabelKey", inputColumnName: "Label"))
                .Append(mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(labelColumnName: "LabelKey", featureColumnName: "softmax2_pre_activation"))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabelValue", "PredictedLabel"))
                .AppendCacheCheckpoint(mlContext);

            IDataView trainingData = mlContext.Data.LoadFromTextFile<ImageData>(path: _trainTagsTsv, hasHeader: false);
            ITransformer model = pipeline.Fit(trainingData);

            IDataView testData = mlContext.Data.LoadFromTextFile<ImageData>(path: _testTagsTsv, hasHeader: false);
            IDataView predictions = model.Transform(testData);

            // Create an IEnumerable for the predictions for displaying results
            IEnumerable<ImagePrediction> imagePredictionData = mlContext.Data.CreateEnumerable<ImagePrediction>(predictions, true);
            TransferLearningTF.logic.Display.DisplayResults(imagePredictionData);

            MulticlassClassificationMetrics metrics = mlContext.MulticlassClassification.Evaluate(predictions,
                    labelColumnName: "LabelKey",
                    predictedLabelColumnName: "PredictedLabel");

            Console.WriteLine($"LogLoss is: {metrics.LogLoss}");
            Console.WriteLine($"PerClassLogLoss is: {String.Join(" , ", metrics.PerClassLogLoss.Select(c => c.ToString()))}");

            return model;
        }

        public void ClassifySingleImage(MLContext mlContext, ITransformer model)
        {
            var imageData = new ImageData()
            {
                ImagePath = _predictSingleImage
            };

            // Make prediction function (input = ImageData, output = ImagePrediction)
            var predictor = mlContext.Model.CreatePredictionEngine<ImageData, ImagePrediction>(model);
            var prediction = predictor.Predict(imageData);

            Console.WriteLine($"Image: {Path.GetFileName(imageData.ImagePath)} predicted as: {prediction.PredictedLabelValue} with score: {prediction.Score?.Max()} ");
        }
    }
}
