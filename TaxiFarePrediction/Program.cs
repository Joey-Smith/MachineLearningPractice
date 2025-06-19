using Microsoft.ML;
using TaxiFarePrediction;
using TaxiFarePrediction.logic;
using TaxiFarePrediction.models;

MLContext mlContext = new MLContext(seed: 0);
Prediction prediction = new Prediction();

var model = prediction.Train(mlContext);
prediction.Evaluate(mlContext, model);
prediction.TestSinglePrediction(mlContext, model);