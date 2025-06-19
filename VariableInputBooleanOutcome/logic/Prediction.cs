using Microsoft.ML;
using Microsoft.ML.Trainers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using BinaryClassification.models;

namespace BinaryClassification.logic
{
    public class Prediction
    {
        string _trainDataPath, _testDataPath, _modelPath;

        public Prediction()
        {
            _trainDataPath = Path.Combine(Environment.CurrentDirectory, "assets", "taxi-fare-train.csv");
            _testDataPath = Path.Combine(Environment.CurrentDirectory, "assets", "taxi-fare-test.csv");
            _modelPath = Path.Combine(Environment.CurrentDirectory, "assets", "Model.zip");
        }

        public ITransformer Train(MLContext mlContext)
        {
            IDataView dataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(_trainDataPath, hasHeader: true, separatorChar: ',');

            var pipeline = mlContext.Transforms
                .Categorical.OneHotEncoding(outputColumnName: "VendorIdEncoded", inputColumnName: "VendorId")
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "RateCodeEncoded", inputColumnName: "RateCode"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "PaymentTypeEncoded", inputColumnName: "PaymentType"))
                .Append(mlContext.Transforms.Concatenate("Features", "VendorIdEncoded", "RateCodeEncoded", "PassengerCount", "TripDistance", "PaymentTypeEncoded"))
                .Append(mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression(
                    new LbfgsLogisticRegressionBinaryTrainer.Options
                    {
                        LabelColumnName = "FareAmount",
                        FeatureColumnName = "Features",
                        MaximumNumberOfIterations = 100,
                        OptimizationTolerance = 1e-8f
                    }
                    ));

            //var pipeline = mlContext.Transforms
            //    .CopyColumns(outputColumnName: "Label", inputColumnName: "FareAmount")
            //    .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "VendorIdEncoded", inputColumnName: "VendorId"))
            //    .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "RateCodeEncoded", inputColumnName: "RateCode"))
            //    .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "PaymentTypeEncoded", inputColumnName: "PaymentType"))
            //    .Append(mlContext.Transforms.Concatenate("Features", "VendorIdEncoded", "RateCodeEncoded", "PassengerCount", "TripDistance", "PaymentTypeEncoded"))
            //    .Append(mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression(
            //        new LbfgsLogisticRegressionBinaryTrainer.Options
            //        {
            //            LabelColumnName = "FareAmount",
            //            FeatureColumnName = "Features",
            //            MaximumNumberOfIterations = 100,
            //            OptimizationTolerance = 1e-8f
            //        }
            //        ));

            var model = pipeline.Fit(dataView);

            return model;
        }

        public void Evaluate(MLContext mlContext, ITransformer model)
        {
            IDataView dataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(_testDataPath, hasHeader: true, separatorChar: ',');

            var predictions = model.Transform(dataView);
            var metrics = mlContext.BinaryClassification.Evaluate(predictions, "FareAmount", "Score");

            Console.WriteLine();
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Model quality metrics evaluation         ");
            Console.WriteLine($"*------------------------------------------------");

            Console.WriteLine($"*       Accuracy:      {metrics.Accuracy.ToString("F4")}");
            Console.WriteLine($"*       LogLossReduction:      {metrics.LogLossReduction}");
        }

        public void TestSinglePrediction(MLContext mlContext, ITransformer model)
        {
            var predictionFunction = mlContext.Model.CreatePredictionEngine<TaxiTrip, TaxiTripFarePrediction>(model);

            var taxiTripSample = new TaxiTrip()
            {
                VendorId = "CMT",
                RateCode = "1",
                PassengerCount = 1,
                TripTime = 1271,
                TripDistance = 3.8f,
                PaymentType = "CRD"
            };

            var prediction = predictionFunction.Predict(taxiTripSample);

            Console.WriteLine($"**********************************************************************");
            Console.WriteLine($"Predicted fare: {prediction.PredictedLabel}, actual fare: true");
            Console.WriteLine($"**********************************************************************");
        }
    }
}
