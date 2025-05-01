import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import chi2_contingency
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import itertools
import os
from openai import OpenAI

# ページ設定
st.set_page_config(
    page_title="採用動画アンケート結果分析",
    page_icon="📊",
    layout="wide"
)

# OpenAI APIキーの設定
try:
    api_key = st.secrets["OPENAI_API_KEY"]
    client = OpenAI(api_key=api_key)
except Exception as e:
    st.error(f"OpenAI APIキーの設定中にエラーが発生しました: {str(e)}")
    st.stop()

# 日本語フォントの設定
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic']

def analyze_free_text_with_openai(text_series):
    """OpenAI APIを使用した自由記述の分析（構造化された出力形式）"""
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
        # OpenAI APIを使用して分析を実行
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "あなたはアンケート分析の専門家です。"},
                {"role": "user", "content": formatted_prompt}
            ],
            temperature=0.7
        )
        
        # レスポンス処理
        if response and response.choices:
            analysis_result = response.choices[0].message.content
            results.append(analysis_result)
        else:
            results.append("分析結果を取得できませんでした。")
        
    except Exception as e:
        st.error(f"OpenAI APIを使用した分析中にエラーが発生しました: {str(e)}")
        results.append(f"分析に失敗しました。エラー: {str(e)}")
    
    return results

def analyze_free_text(df, text_columns):
    """自由記述の分析（OpenAI APIのみ）"""
    results = {}
    
    for column in text_columns:
        # OpenAI APIを使用した分析
        openai_analysis = analyze_free_text_with_openai(df[column])
        
        results[column] = {
            'openai_analysis': openai_analysis
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

def analyze_all_survey_responses(df, text_columns, attributes, yes_no_questions):
    """アンケート全体の戦略分析（定量・定性分析）"""
    # 1. 定量分析の準備
    quantitative_analysis = {}
    
    # 属性データの分析
    for attr in attributes:
        if "（上位3つまで）" in attr or "（複数選択可）" in attr:
            # 複数回答の処理
            responses = []
            for response in df[attr].dropna():
                items = [item.strip() for item in str(response).split('、')]
                responses.extend(items)
            value_counts = pd.Series(responses).value_counts()
            quantitative_analysis[attr] = {
                'type': 'multiple_choice',
                'total_responses': len(responses),
                'unique_responses': len(value_counts),
                'top_responses': value_counts.head(5).to_dict(),
                'distribution': value_counts.to_dict()
            }
        else:
            # 単一回答の処理
            value_counts = df[attr].value_counts()
            quantitative_analysis[attr] = {
                'type': 'single_choice',
                'total_responses': df[attr].count(),
                'unique_responses': df[attr].nunique(),
                'top_response': value_counts.index[0] if not value_counts.empty else None,
                'distribution': value_counts.to_dict()
            }
    
    # 2択質問の分析
    for question in yes_no_questions:
        value_counts = df[question].value_counts(normalize=True)
        quantitative_analysis[question] = {
            'type': 'binary',
            'total_responses': df[question].count(),
            'distribution': value_counts.to_dict(),
            'positive_ratio': value_counts.get('はい', 0)
        }
    
    # 2. 定性分析（自由記述）
    all_responses = []
    for column in text_columns:
        responses = df[column].dropna()
        all_responses.extend(responses)
    
    combined_text = ' '.join(all_responses)
    
    # プロンプトテンプレート
    prompt_template = """
    以下の採用動画アンケートの分析結果を総合的に評価し、定量・定性の両面から戦略的な示唆をまとめてください。

    【定量分析結果】
    {quantitative_summary}

    【自由記述回答】
    {text}

    # 出力形式
    ##### 回答者属性
       [主要な属性分布と特徴を要約]

    ##### データから見る強み:
       - [強み1]: [定量的な裏付けとともに説明]
       - [強み2]: [定量的な裏付けとともに説明]
       - [強み3]: [定量的な裏付けとともに説明]

    ##### データから見る課題:
       - [課題1]: [定量的な裏付けと想定される影響]
       - [課題2]: [定量的な裏付けと想定される影響]
       - [課題3]: [定量的な裏付けと想定される影響]

    ##### アクションプラン:
        短期的な改善施策 (1-3ヶ月):
            - [データに基づく具体的な施策1]
            - [データに基づく具体的な施策2]
            - [データに基づく具体的な施策3]
    
        中期的な戦略施策 (3-6ヶ月):
            - [データに基づく具体的な施策1]
            - [データに基づく具体的な施策2]
            - [データに基づく具体的な施策3]
    
        長期的な戦略転換 (6ヶ月以上):
            - [データに基づく具体的な施策1]
            - [データに基づく具体的な施策2]
            - [データに基づく具体的な施策3]

    """
    
    # 定量分析のサマリーを作成
    quantitative_summary = []
    
    # 属性データのサマリー
    for attr, data in quantitative_analysis.items():
        if data['type'] == 'multiple_choice':
            summary = f"{attr}の上位回答: " + ", ".join([f"{k}({v}件)" for k, v in data['top_responses'].items()])
        elif data['type'] == 'single_choice':
            summary = f"{attr}の最頻値: {data['top_response']} ({data['distribution'][data['top_response']]}件)"
        else:  # binary
            summary = f"{attr}の肯定的回答率: {data.get('positive_ratio', 0)*100:.1f}%"
        quantitative_summary.append(summary)
    
    # プロンプトにデータを埋め込む
    formatted_prompt = prompt_template.format(
        quantitative_summary="\n".join(quantitative_summary),
        text=combined_text
    )
    
    try:
        # OpenAI APIを使用して分析を実行
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "あなたはアンケート分析と採用戦略の専門家です。"},
                {"role": "user", "content": formatted_prompt}
            ],
            temperature=0.7
        )
        
        # レスポンス処理
        if response and response.choices:
            return {
                'quantitative_analysis': quantitative_analysis,
                'strategic_analysis': response.choices[0].message.content
            }
        else:
            return {
                'quantitative_analysis': quantitative_analysis,
                'strategic_analysis': "分析結果を取得できませんでした。"
            }
        
    except Exception as e:
        st.error(f"OpenAI APIを使用した分析中にエラーが発生しました: {str(e)}")
        return {
            'quantitative_analysis': quantitative_analysis,
            'strategic_analysis': f"分析に失敗しました。エラー: {str(e)}"
        }

def visualize_analysis(df, attributes, yes_no_questions, text_columns):
    """分析結果の可視化"""
    # 属性データの分析
    stats, attribute_ranked = analyze_attributes(df, attributes)
    # 2択質問の分析
    response_dist = analyze_yes_no_questions(df, yes_no_questions, attributes)
    # 自由記述の分析
    text_analysis = analyze_free_text(df, text_columns)
    # 全体の戦略分析（定量・定性）
    comprehensive_analysis = analyze_all_survey_responses(df, text_columns, attributes, yes_no_questions)
    
    return {
        'stats': stats,
        'attribute_ranked': attribute_ranked,
        'response_dist': response_dist,
        'text_analysis': text_analysis,
        'comprehensive_analysis': comprehensive_analysis
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
    tab_attributes, tab_yes_no, tab_text, tab_cross, tab_strategic = st.tabs([
        "1. 属性分析",
        "2. 2択質問分析",
        "3. 自由記述分析",
        "4. クロス分析",
        "5. 総合分析"
    ])
    
    # 1. 属性分析タブ
    with tab_attributes:
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
        # 質問ごとのサブタブを作成
        if analysis_results['text_analysis']:
            # サブタブのリストを作成
            text_tabs = st.tabs([f"質問{i+1}: {column.split('#')[1] if '#' in column else column}" 
                               for i, column in enumerate(analysis_results['text_analysis'].keys())])
            
            # 各サブタブに分析結果を表示
            for tab, (column, analysis) in zip(text_tabs, analysis_results['text_analysis'].items()):
                with tab:                    
                    # OpenAIによる分析結果の表示
                    for result in analysis['openai_analysis']:
                        st.markdown(result)
                        
    # 4. クロス分析タブ
    with tab_cross:
        def display_cross_analysis(df):
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

    # 5. 戦略分析タブ
    with tab_strategic:
        if 'comprehensive_analysis' in analysis_results:
            # 戦略分析結果の表示
            st.markdown(analysis_results['comprehensive_analysis']['strategic_analysis'])
        else:
            st.warning("分析結果が見つかりません。")    