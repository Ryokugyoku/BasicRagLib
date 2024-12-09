namespace BasicRagLib.CntkModule;
using System;
using System.Collections.Generic;
using System.IO;
using CNTK;
public class CntkLearning
{

        // モデルファイルパスを受け取るコンストラクタ
    public CntkLearning(string modelFilePath)
    {
        // モデルを読み込む
        var model = Function.Load(modelFilePath, DeviceDescriptor.UseDefaultDevice());
    }
    /// <summary>
    /// CSVファイルのファイルパス、モデルファイルのファイルパスを指定してCNTKを使った学習を行う
    /// </summary>
    /// <param name="csvFilePath"></param>
    /// <param name="modelFilePath"></param>
    public CntkLearning(string csvFilePath, string modelFilePath)
    {
        // ここでCNTKを使った学習を行う
        var device = DeviceDescriptor.UseDefaultDevice();

        var data = LoadCsvData(csvFilePath);
        // データの前処理
        var features = data.Item1;
        var labels = data.Item2;



        Function model;
        if(File.Exists(modelFilePath)){
            model = Function.Load(modelFilePath, device);
        }else{
                        // 入力と出力の変数を定義
            var inputDim = features[0].Length;
            var outputDim = labels[0].Length;
            var input = Variable.InputVariable(new int[] { inputDim }, DataType.Float);
            var output = Variable.InputVariable(new int[] { outputDim }, DataType.Float);

            model =  CreateModel(input, outputDim, device);
        }
         

        // 損失関数と評価関数の定義
        var loss = CNTKLib.CrossEntropyWithSoftmax(model, model.Output);
        var evalError = CNTKLib.ClassificationError(model, model.Output);

        // トレーナーの定義
        var learningRate = new TrainingParameterScheduleDouble(0.01, 1);
        var trainer = Trainer.CreateTrainer(model, loss, evalError, new List<Learner> { Learner.SGDLearner(model.Parameters(), learningRate) });

        // トレーニングループ
        int minibatchSize = 64;
        int numMinibatchesToTrain = 1000;
        for (int i = 0; i < numMinibatchesToTrain; i++)
        {
            var minibatchData = GetMinibatch(features, labels, minibatchSize, device);
            trainer.TrainMinibatch(minibatchData, device);
            if (i % 100 == 0)
            {
                Console.WriteLine($"Minibatch {i}, Loss: {trainer.PreviousMinibatchLossAverage()}, Error: {trainer.PreviousMinibatchEvaluationAverage()}");
            }
        }

        model.Save(modelFilePath);
    }

    /// <summary>
    /// CSVファイルからデータを読み込む
    /// 最初のカラムがラベル、残りのカラムが特徴量として取り込む
    /// </summary>
    /// <param name="csvFilePath"></param>
    /// <returns></returns>
    private Tuple<float[][], float[][]> LoadCsvData(string csvFilePath)
    {
        var lines = File.ReadAllLines(csvFilePath);
        var features = new List<float[]>();
        var labels = new List<float[]>();

        foreach (var line in lines)
        {
            var values = line.Split(',');
            // 最初のカラムをラベルとして取り込む
            var label = new float[] { float.Parse(values[0]) };
            // 残りのカラムを特徴量として取り込む
            var feature = Array.ConvertAll(values[1..], float.Parse);
            features.Add(feature);
            labels.Add(label);
        }

        return new Tuple<float[][], float[][]>(features.ToArray(), labels.ToArray());
    }
    /// <summary>
    /// ミニバッチを作成し、CNTKのトレーニングに使用できるようにデータを背形する。
    /// </summary>
    /// <param name="features"></param>
    /// <param name="labels"></param>
    /// <param name="minibatchSize"></param>
    /// <param name="device"></param>
    /// <returns></returns>
    private Dictionary<Variable, Value> GetMinibatch(float[][] features, float[][] labels, int minibatchSize, DeviceDescriptor device)
    {
        var featureBatch = new List<float>();
        var labelBatch = new List<float>();

        for (int i = 0; i < minibatchSize; i++)
        {
            featureBatch.AddRange(features[i]);
            labelBatch.AddRange(labels[i]);
        }

        var featureValue = Value.CreateBatch(new int[] { features[0].Length }, featureBatch, device);
        var labelValue = Value.CreateBatch(new int[] { labels[0].Length }, labelBatch, device);

        return new Dictionary<Variable, Value>
        {
            { Variable.InputVariable(new int[] { features[0].Length }, DataType.Float), featureValue },
            { Variable.InputVariable(new int[] { labels[0].Length }, DataType.Float), labelValue }
        };
    }

    /// <summary>
    /// ニューラルモデルを定義する処理。
    /// 隠れ層、出力層の定義を行う。
    /// </summary>
    /// <param name="input"></param>
    /// <param name="outputDim"></param>
    /// <param name="device"></param>
    /// <returns></returns>
    private Function CreateModel(Variable input, int outputDim, DeviceDescriptor device)
    {
        
        var hiddenLayer = Dense(input, 50, device, CNTKLib.ReLU);
        var outputLayer = Dense(hiddenLayer, outputDim, device, CNTKLib.Softmax);
        return outputLayer;
    }

    /// <summary>
    /// 全結合層を定義する処理
    /// weightParam: 入力ユニット、出力ユニットの間の重みを表します。
    /// biasParam: 出力ユニットのバイアスを表す。
    /// </summary>
    /// <param name="input"></param>
    /// <param name="outputDim"></param>
    /// <param name="device"></param>
    /// <param name="activation"></param>
    /// <returns></returns>
    private Function Dense(Variable input, int outputDim, DeviceDescriptor device, Func<Variable, Function> activation)
    {
        var inputDim = input.Shape[0];
        var weightParam = new Parameter(new int[] { outputDim, inputDim }, DataType.Float, CNTKLib.GlorotUniformInitializer(), device);
        var biasParam = new Parameter(new int[] { outputDim }, DataType.Float, 0, device);
        var fullyConnected = CNTKLib.Times(weightParam, input) + biasParam;
        return activation(fullyConnected);
    }
}
