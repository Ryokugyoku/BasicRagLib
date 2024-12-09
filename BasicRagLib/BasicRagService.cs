using BasicRagLib.CntkModule;

namespace BasicRagLib;

/// <summary>
/// Ragを構築するためのサービス
/// </summary>
public class BasicRagService
{

    BasicRagService(string modelFilePath)
    {

    }

    /// <summary>
    /// CSVファイルのファイルパス、モデルファイルのファイルパスを指定してRagを構築する
    /// </summary>
    /// <param name="csvFilePath"> CSVファイルが格納されたファイルパス</param>
    /// <param name="modelFilePath"> 学習したモデルを格納するファイル</param>
    BasicRagService(string csvFilePath, string modelFilePath)
    {
        // ここでRagを構築する
        CntkLearning cntkLearning = new CntkLearning(csvFilePath, modelFilePath);

    }
}
