using Microsoft.ML;
using Microsoft.ML.Data;
using TransferLearningTF.models;
using TransferLearningTF.logic;

MLContext mlContext = new MLContext();
Prediction prediction = new Prediction();

ITransformer model = prediction.GenerateModel(mlContext);
prediction.ClassifySingleImage(mlContext, model);