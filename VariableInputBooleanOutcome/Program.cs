using Microsoft.ML;
using BinaryClassification;
using BinaryClassification.logic;
using BinaryClassification.models;

MLContext mlContext = new MLContext(seed: 0);
Prediction prediction = new Prediction();

var model = prediction.Train(mlContext);
prediction.Evaluate(mlContext, model);
prediction.TestSinglePrediction(mlContext, model);