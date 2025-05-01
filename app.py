import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import chi2_contingency
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import re
from collections import Counter
import itertools
import os
from anthropic import Anthropic

# ページ設定
st.set_page_config(
    page_title="採用動画アンケート結果分析",
    page_icon="📊",
    layout="wide"
)

# Anthropic APIキーの設定
try:
    api_key = st.secrets["ANTHROPIC_API_KEY"]
    
    # Anthropicライブラリをインポート
    import anthropic
    
    # 新しいAPI形式でクライアント初期化
    client = anthropic.Anthropic(api_key=api_key)
    
except Exception as e:
    st.error(f"Anthropic APIキーの設定中にエラーが発生しました: {str(e)}")
    st.stop()

# 日本語フォントの設定
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic']

def analyze_free_text_with_anthropic(text_series):
    """Anthropic APIを使用した自由記述の分析（構造化された出力形式）"""
    results = []
    
    # テキストを結合して分析用のプロンプトを作成
    combined_text = ' '.join(text_series.dropna())
    
    # プロンプトテンプレート
    prompt_template = """
    以下の自由記述アンケート回答を分析してください。回答者の意見、感想、提案を体系的に整理し、以下の形式で結果をまとめてください。

    【分析対象】
    {text}

    【分析結果】
    1. 主要テーマ（5つまで、重要度順）:
       - テーマ1: [テーマ名] - [簡潔な説明]
       - テーマ2: [テーマ名] - [簡潔な説明]
       ...

    2. 頻出キーワード（10個まで、出現頻度順）:
       - キーワード1: [出現回数]
       - キーワード2: [出現回数]
       ...

    3. 感情分析:
       - ポジティブな意見: [割合]%
       - ネガティブな意見: [割合]%
       - 中立的な意見: [割合]%
       - 主なポジティブポイント: [簡潔に箇条書き]
       - 主なネガティブポイント: [簡潔に箇条書き]

    4. カテゴリー別の意見（重要な順に5つまで）:
       - カテゴリー1: 
         * 主な意見: [簡潔に要約]
         * 具体的な提案: [あれば記載]
       - カテゴリー2:
         * 主な意見: [簡潔に要約]
         * 具体的な提案: [あれば記載]
       ...

    5. 特筆すべき少数意見（3つまで）:
       - [意見1の要約]
       - [意見2の要約]
       - [意見3の要約]

    6. 総合分析（200字以内）:
       [アンケート全体の傾向、主な発見、示唆される対応策などを簡潔に記載]

    7. アクションアイテム（優先度順に3つまで）:
       - [具体的な行動提案1]
       - [具体的な行動提案2]
       - [具体的な行動提案3]
    """
    
    # プロンプトにテキストを埋め込む
    formatted_prompt = prompt_template.format(text=combined_text)
    
    try:
        # Anthropic APIを使用して分析を実行
        message = client.messages.create(
            model="claude-3-5-haiku-latest",
            max_tokens=2000,  # 十分な出力トークン数を確保
            messages=[
                {
                    "role": "user",
                    "content": formatted_prompt
                }
            ]
        )
        
        # レスポンス処理
        if message and hasattr(message, 'content') and message.content:
            analysis_result = message.content[0].text
            results.append(analysis_result)
        else:
            results.append("分析結果を取得できませんでした。")
        
    except Exception as e:
        st.error(f"Anthropic APIを使用した分析中にエラーが発生しました: {str(e)}")
        results.append(f"分析に失敗しました。エラー: {str(e)}")
    
    return results

def analyze_free_text(df, text_columns):
    """自由記述の分析（Anthropic APIのみ）"""
    results = {}
    
    for column in text_columns:
        # Anthropic APIを使用した分析
        anthropic_analysis = analyze_free_text_with_anthropic(df[column])
        
        results[column] = {
            'anthropic_analysis': anthropic_analysis
        }
    
    return results

def process_ranked_attributes(df, question):
    """属性データ用：順位付き複数回答の処理（全体分布も返す）"""
    answers = []
    for response in df[question]:
        if pd.isna(response):
            continue
        items = [item.strip() for item in str(response).split('、')]
        
        # 質問タイプの判定
        if "（上位3つまで）" in question:
            # 上位3つまでの質問は順位付きで処理
            for rank, item in enumerate(items, 1):
                answers.append((item, rank))
        elif "（複数選択可）" in question:
            # 複数選択可の質問は順位なしで処理
            for item in items:
                answers.append((item, None))
    
    if not answers:
        return {}
    
    result_df = pd.DataFrame(answers, columns=['回答', '順位'])
    rank_distributions = {}
    
    if "（上位3つまで）" in question:
        # 上位3つまでの質問の場合、順位ごとの分布を計算
        max_rank = result_df['順位'].max()
        for rank in range(1, max_rank + 1):
            rank_answers = result_df[result_df['順位'] == rank]['回答']
            if not rank_answers.empty:
                rank_distributions[f"{rank}位"] = rank_answers.value_counts()
    else:
        # 複数選択可の質問の場合、全体の分布のみを計算
        all_counts = result_df['回答'].value_counts()
        rank_distributions['全体'] = all_counts
    
    return rank_distributions

def analyze_attributes(df, attributes):
    """属性データの分析（通常属性＋複数回答属性の順位分布）"""
    stats = {}
    ranked_distributions = {}
    for attr in attributes:
        if "（上位3つまで）" in attr or "（複数選択可）" in attr:
            # 複数回答属性は順位ごとの分布を計算
            ranked_distributions[attr] = process_ranked_attributes(df, attr)
        else:
            value_counts = df[attr].value_counts()
            stats[attr] = {
                'count': df[attr].count(),
                'unique': df[attr].nunique(),
                'top': value_counts.index[0] if not value_counts.empty else None,
                'freq': value_counts.iloc[0] if not value_counts.empty else 0,
                'distribution': value_counts.to_dict()
            }
    return stats, ranked_distributions

def analyze_yes_no_questions(df, yes_no_questions, attributes):
    """2択質問の分析"""
    # 1. 回答分布の集計
    response_dist = {}
    for question in yes_no_questions:
        if "（上位3つまで）" in question or "（複数選択可）" in question:
            # 複数回答の質問は特別処理
            ranked_answers = process_ranked_attributes(df, question)
            if not ranked_answers:
                response_dist[question] = {}
            else:
                response_dist[question] = ranked_answers
        else:
            # 通常の2択質問
            response_dist[question] = df[question].value_counts(normalize=True)
    
    return response_dist

def visualize_analysis(df, attributes, yes_no_questions, text_columns):
    """分析結果の可視化"""
    # 属性データの分析
    stats, attribute_ranked = analyze_attributes(df, attributes)
    # 2択質問の分析
    response_dist = analyze_yes_no_questions(df, yes_no_questions, attributes)
    # 自由記述の分析
    text_analysis = analyze_free_text(df, text_columns)
    return {
        'stats': stats,
        'attribute_ranked': attribute_ranked,
        'response_dist': response_dist,
        'text_analysis': text_analysis
    }

# タイトル
st.title("採用動画アンケート結果分析")

# ファイルアップロード
uploaded_file = st.file_uploader("CSVファイルをアップロード", type=['csv'])

if uploaded_file is not None:
    # CSVファイルの読み込み
    df = pd.read_csv(uploaded_file)
    
    # 質問の分類
    attributes = [col for col in df.columns if '▪︎' in col]
    yes_no_questions = [col for col in df.columns if '・' in col]
    text_columns = [col for col in df.columns if '#' in col]
    
    # 分析の実行
    analysis_results = visualize_analysis(df, attributes, yes_no_questions, text_columns)
    
    # タブの作成
    tab_attributes, tab_yes_no, tab_text, tab_summary, tab_cross = st.tabs([
        "1. 属性分析",
        "2. 2択質問分析",
        "3. 自由記述分析",
        "4. 総合分析",
        "5. クロス分析"
    ])
    
    # 1. 属性分析タブ
    with tab_attributes:
        st.markdown("### 1. 属性分析")
        # 3列のレイアウトを作成
        cols = st.columns(3)
        col_index = 0
        
        for attr in attributes:
            if "（上位3つまで）" in attr or "（複数選択可）" in attr:
                # 複数回答の質問は質問タイプに応じて処理
                rank_distributions = analysis_results['attribute_ranked'].get(attr, {})
                
                if "（上位3つまで）" in attr:
                    # 上位3つまでの質問は順位ごとに独立した図を表示
                    rank_keys = [k for k in rank_distributions.keys() if k != '全体']
                    for rank in sorted(rank_keys, key=lambda x: int(x.replace('位','')) if x.endswith('位') else 999):
                        with cols[col_index % 3]:
                            rank_dist = rank_distributions[rank]
                            # 回答数でソート
                            sorted_dist = rank_dist.sort_values(ascending=False)
                            st.markdown(f"###### {attr} - {rank}")
                            fig = px.pie(
                                values=sorted_dist.values,
                                names=sorted_dist.index,
                                width=400,
                                height=400
                            )
                            fig.update_layout(
                                uniformtext_minsize=12,
                                uniformtext_mode='hide'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        col_index += 1
                else:
                    # 複数選択可の質問は全体の分布のみを表示
                    with cols[col_index % 3]:
                        all_dist = rank_distributions.get('全体', pd.Series())
                        # 回答数でソート
                        sorted_dist = all_dist.sort_values(ascending=False)
                        st.markdown(f"###### {attr}")
                        fig = px.pie(
                            values=sorted_dist.values,
                            names=sorted_dist.index,
                            width=400,
                            height=400
                        )
                        fig.update_layout(
                            uniformtext_minsize=12,
                            uniformtext_mode='hide'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    col_index += 1
            else:
                # 通常の属性は1つの図を表示
                with cols[col_index % 3]:
                    stat = analysis_results['stats'][attr]
                    # 回答数でソート
                    sorted_dist = pd.Series(stat['distribution']).sort_values(ascending=False)
                    st.markdown(f"###### {attr}")
                    fig = px.pie(
                        values=sorted_dist.values,
                        names=sorted_dist.index,
                        width=400,
                        height=400
                    )
                    fig.update_layout(
                        uniformtext_minsize=12,
                        uniformtext_mode='hide'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                col_index += 1
    
    # 2. 2択質問分析タブ
    with tab_yes_no:
        st.markdown("### 2. 2択質問分析")
        
        # 回答分布の表示
        st.markdown("#### 回答分布")
        # 3列のレイアウトを作成
        cols = st.columns(3)
        col_index = 0
        
        for question, dist in analysis_results['response_dist'].items():
            if "（上位3つまで）" in question or "（複数選択可）" in question:
                # 複数回答の質問は質問タイプに応じて処理
                if "（上位3つまで）" in question:
                    # 上位3つまでの質問は順位ごとの円グラフ
                    for rank, rank_dist in dist.items():
                        with cols[col_index % 3]:
                            # 回答数でソート
                            sorted_dist = rank_dist.sort_values(ascending=False)
                            st.markdown(f"###### {question} - {rank}")
                            fig = px.pie(
                                values=sorted_dist.values,
                                names=sorted_dist.index,
                                width=400,
                                height=400
                            )
                            fig.update_layout(
                                uniformtext_minsize=12,
                                uniformtext_mode='hide'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        col_index += 1
                else:
                    # 複数選択可の質問は全体の分布のみを表示
                    with cols[col_index % 3]:
                        all_dist = dist.get('全体', pd.Series())
                        # 回答数でソート
                        sorted_dist = all_dist.sort_values(ascending=False)
                        st.markdown(f"###### {question}")
                        fig = px.pie(
                            values=sorted_dist.values,
                            names=sorted_dist.index,
                            width=400,
                            height=400
                        )
                        fig.update_layout(
                            uniformtext_minsize=12,
                            uniformtext_mode='hide'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    col_index += 1
            else:
                # 通常の2択質問
                with cols[col_index % 3]:
                    # 回答数でソート
                    sorted_dist = dist.sort_values(ascending=False)
                    st.markdown(f"###### {question}")
                    fig = px.pie(
                        values=sorted_dist.values,
                        names=sorted_dist.index,
                        width=400,
                        height=400
                    )
                    fig.update_layout(
                        uniformtext_minsize=12,
                        uniformtext_mode='hide'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                col_index += 1
    
    # 3. 自由記述分析タブ
    with tab_text:
        st.markdown("### 3. 自由記述分析")
        
        # 質問ごとのサブタブを作成
        if analysis_results['text_analysis']:
            # サブタブのリストを作成
            text_tabs = st.tabs([f"質問{i+1}: {column.split('#')[1] if '#' in column else column}" 
                               for i, column in enumerate(analysis_results['text_analysis'].keys())])
            
            # 各サブタブに分析結果を表示
            for tab, (column, analysis) in zip(text_tabs, analysis_results['text_analysis'].items()):
                with tab:
                    # 元の質問文を表示
                    st.markdown(f"**質問文**: {column}")
                    st.markdown("---")
                    
                    # Anthropicによる分析結果の表示
                    st.markdown("#### AI分析結果")
                    for result in analysis['anthropic_analysis']:
                        st.markdown(result)
                        
    # 4. 総合分析タブ
    with tab_summary:
        st.markdown("### 4. 総合分析")
        
        # 主成分分析
        st.markdown("#### 主成分分析")
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(df[numeric_columns])
            fig = px.scatter(
                x=pca_result[:, 0],
                y=pca_result[:, 1],
                title="主成分分析結果"
            )
            st.plotly_chart(fig)
        
        # クラスター分析
        st.markdown("#### クラスター分析")
        if len(numeric_columns) > 0:
            kmeans = KMeans(n_clusters=3)
            clusters = kmeans.fit_predict(df[numeric_columns])
            fig = px.scatter(
                x=pca_result[:, 0],
                y=pca_result[:, 1],
                color=clusters,
                title="クラスター分析結果"
            )
            st.plotly_chart(fig)

    # 5. クロス分析タブ
    with tab_cross:
        def display_cross_analysis(df):
            st.markdown("### 5. クロス分析")

            tabs = st.tabs([
                "性別 × 興味のある業界",
                "性別 × 興味のある職種",
                "学年 × 興味のある業界",
                "学年 × 興味のある職種",
                "出身地 × 興味のある業界",
                "出身地 × 興味のある職種",
                "学年 × 憧れている業界"
            ])

            # 定義: 各クロス集計タブの情報
            cross_info = [
                ("▪︎ 性別", "▪︎ 興味のある業界"),
                ("▪︎ 性別", "▪︎ 興味のある職種"),
                ("▪︎ 学年", "▪︎ 興味のある業界"),
                ("▪︎ 学年", "▪︎ 興味のある職種"),
                ("▪︎ 出身地", "▪︎ 興味のある業界"),
                ("▪︎ 出身地", "▪︎ 興味のある職種"),
                ("▪︎ 学年", "▪︎ 憧れている業界")
            ]

            for tab, (row_attr, col_attr) in zip(tabs, cross_info):
                with tab:
                    st.markdown(f"#### {row_attr} × {col_attr} のクロス集計")
                    try:
                        ct = pd.crosstab(df[row_attr], df[col_attr])
                        ct = ct.loc[:, ct.sum().sort_values(ascending=False).index]  # カラムを頻度順に

                        fig = px.bar(
                            ct.T,
                            barmode="stack",
                            title=f"{row_attr} × {col_attr}"
                        )
                        fig.update_layout(xaxis_title=col_attr, yaxis_title="人数")
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"エラーが発生しました: {str(e)}")

        # クロス分析の実行
        display_cross_analysis(df)    