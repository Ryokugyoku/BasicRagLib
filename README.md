# BasicRagLib Project

# 要件定義書 (Requirements Definition Document)

## 概要 (Overview)
OpenAIなどと組み合わせて、独自の検索拡張生成AIを実装を簡単にするためのライブラリ

利用している機械学習ライブラリ:
[Cntk](https://ja.wikipedia.org/wiki/Microsoft_Cognitive_Toolkit)

## 目的 (Purpose)
このプロジェクトの目的は、開発者が簡単にカスタム検索拡張AIを構築できるようにすることです。これにより、特定のニーズに合わせた検索機能を提供することができます。

## 機能要件 (Functional Requirements)
1. **検索機能の拡張**:
   - ユーザーが用意したCSVファイルを元に、学習させ、OpenAIにあるPromptに対して捕捉情報などを注入できるようにする。
   
2. **Future**:
    - DBから学習できるようにする
    - GraphRAGの実装　※優先

