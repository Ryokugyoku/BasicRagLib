using CNTK;

namespace BasicRagLib.CntkModule;
/// <summary>
/// モデルの作成・読み込みを行うクラスファイル
/// </summary>
class BaseModel{
    /// <summary>
    /// モデルファイルのオブジェクト
    /// このモデルファイルを参照して、諸々の操作を行う
    /// </summary>
    protected static Function? Model { get; set; } = null;

    /// <summary>
    /// モデルファイルパスを受け取るコンストラクタ
    /// モデルが存在しない場合、新規で作成を行う
    /// モデルがすでに読み込まれている場合は、処理を行わない
    /// </summary>
    /// <param name="modelFilePath">モデルファイルのパス</param>
    protected BaseModel(string modelFilePath){
        if(Model != null){
            return;
        }

        var device = DeviceDescriptor.UseDefaultDevice();
        
        if(File.Exists(modelFilePath)){
            // すでにモデルが存在する場合
            Model = Function.Load(modelFilePath, device);
        }else{
            Model = CreateNewModel(device);
            Model.Save(modelFilePath);
        }
        
    }

    /// <summary>
    /// 新しいモデルを作成する
    /// </summary>
    /// <param name="device">デバイス</param>
    /// <returns>作成されたモデル</returns>
    protected Function CreateNewModel(DeviceDescriptor device)
    {
        // 入力と出力の変数を定義
        var inputDim = 100; // 例として100次元の入力
        var outputDim = 10; // 例として10次元の出力
        var input = Variable.InputVariable(new int[] { inputDim }, DataType.Float);
        var output = Variable.InputVariable(new int[] { outputDim }, DataType.Float);

        // モデルの定義
        var hiddenLayer = Dense(input, 50, device, CNTKLib.ReLU);
        var outputLayer = Dense(hiddenLayer, outputDim, device, CNTKLib.Softmax);
        return outputLayer;
    }

     /// <summary>
    /// 全結合層を定義する処理
    /// </summary>
    /// <param name="input">入力変数</param>
    /// <param name="outputDim">出力次元</param>
    /// <param name="device">デバイス</param>
    /// <param name="activation">活性化関数</param>
    /// <returns>全結合層</returns>
    protected Function Dense(Variable input, int outputDim, DeviceDescriptor device, Func<Variable, Function> activation)
    {
        var inputDim = input.Shape[0];
        var weightParam = new Parameter(new int[] { outputDim, inputDim }, DataType.Float, CNTKLib.GlorotUniformInitializer(), device);
        var biasParam = new Parameter(new int[] { outputDim }, DataType.Float, 0, device);
        var fullyConnected = CNTKLib.Times(weightParam, input) + biasParam;
        return activation(fullyConnected);
    }
}