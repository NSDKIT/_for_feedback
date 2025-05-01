import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import chi2_contingency
import networkx as nx
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import re
from collections import Counter
import itertools
# import openai

# # OpenAI APIキーの設定
# try:
#     openai.api_key = st.secrets["OPENAI_API_KEY"]
# except Exception as e:
#     st.error(f"OpenAI APIキーの設定中にエラーが発生しました: {str(e)}")
#     st.stop()

# ページ設定
st.set_page_config(
    page_title="採用動画アンケート結果分析",
    page_icon="📊",
    layout="wide"
)

# 日本語フォントの設定
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

def preprocess_text(text_series):
    """テキストデータの前処理"""
    # NaNを空文字列に変換
    text_series = text_series.fillna('')
    
    # 全角英数字を半角に変換
    text_series = text_series.str.translate(str.maketrans(
        'ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ０１２３４５６７８９',
        'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
    ))
    
    # 不要な文字を削除
    text_series = text_series.str.replace(r'[^\w\s]', '', regex=True)
    
    return text_series

def extract_themes(text_series, n_themes=5):
    """テーマの抽出"""
    # テキストを単語に分割
    words = ' '.join(text_series).split()
    
    # 単語の出現頻度をカウント
    word_counts = Counter(words)
    
    # 上位n_themes個のテーマを抽出
    themes = [word for word, count in word_counts.most_common(n_themes)]
    
    return themes

def build_co_occurrence_network(text_series, window_size=2):
    """共起ネットワークの構築"""
    # テキストを単語に分割
    words = ' '.join(text_series).split()
    
    # 共起関係をカウント
    co_occurrence = {}
    for i in range(len(words)):
        for j in range(i+1, min(i+window_size+1, len(words))):
            pair = tuple(sorted([words[i], words[j]]))
            co_occurrence[pair] = co_occurrence.get(pair, 0) + 1
    
    return co_occurrence

def analyze_attributes(df, attributes):
    """属性データの分析（通常属性＋複数回答属性の順位分布）"""
    stats = {}
    ranked_distributions = {}
    for attr in attributes:
        if attr in MULTIPLE_CHOICE_QUESTIONS:
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
    # クロス集計
    cross_tabs = {}
    for attr1, attr2 in itertools.combinations(attributes, 2):
        cross_tabs[f"{attr1}_vs_{attr2}"] = pd.crosstab(df[attr1], df[attr2])
    return stats, cross_tabs, ranked_distributions

def process_ranked_attributes(df, question):
    """属性データ用：順位付き複数回答の処理（全体分布も返す）"""
    answers = []
    for response in df[question]:
        if pd.isna(response):
            continue
        items = [item.strip() for item in str(response).split('、')]
        for rank, item in enumerate(items, 1):
            answers.append((item, rank))
    if not answers:
        return {}
    result_df = pd.DataFrame(answers, columns=['回答', '順位'])
    rank_distributions = {}
    max_rank = result_df['順位'].max()
    # 各順位ごとの分布
    for rank in range(1, max_rank + 1):
        rank_answers = result_df[result_df['順位'] == rank]['回答']
        if not rank_answers.empty:
            rank_distributions[f"{rank}位"] = rank_answers.value_counts()
    # 全体分布
    all_counts = result_df['回答'].value_counts()
    rank_distributions['全体'] = all_counts
    return rank_distributions

# 複数回答の質問リスト
MULTIPLE_CHOICE_QUESTIONS = [
    "▪︎ 企業を選ぶ際に重視するポイント",
    "▪︎ 生き生き働いていると感じる状態",
    "▪︎ 働きがいを感じるとき",
    "▪︎ 就活情報源"
]

def analyze_yes_no_questions(df, yes_no_questions, attributes):
    """2択質問の分析"""
    # 1. 回答分布の集計
    response_dist = {}
    for question in yes_no_questions:
        if question in MULTIPLE_CHOICE_QUESTIONS:
            # 複数回答の質問は特別処理
            ranked_answers = process_ranked_attributes(df, question)
            if not ranked_answers:
                response_dist[question] = {}
            else:
                response_dist[question] = ranked_answers
        else:
            # 通常の2択質問
            response_dist[question] = df[question].value_counts(normalize=True)
    
    # 2. 属性別の傾向分析
    trend_analysis = {}
    for question in yes_no_questions:
        for attribute in attributes:
            trend_analysis[f"{question}_by_{attribute}"] = pd.crosstab(
                df[attribute], 
                df[question], 
                normalize='index'
            )
    
    # 3. カイ二乗検定
    chi2_results = {}
    for question in yes_no_questions:
        for attribute in attributes:
            contingency_table = pd.crosstab(df[attribute], df[question])
            chi2, p, dof, expected = chi2_contingency(contingency_table)
            chi2_results[f"{question}_vs_{attribute}"] = {
                'chi2': chi2,
                'p_value': p,
                'dof': dof
            }
    
    return response_dist, trend_analysis, chi2_results

def analyze_free_text(df, text_columns):
    """自由記述の分析"""
    results = {}
    
    for column in text_columns:
        # テキストの前処理
        processed_text = preprocess_text(df[column])
        
        # テーマの抽出
        themes = extract_themes(processed_text)
        
        # 共起ネットワークの構築
        co_occurrence = build_co_occurrence_network(processed_text)
        
        # ワードクラウドの生成
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            font_path=None  # システムのデフォルトフォントを使用
        ).generate(' '.join(processed_text))
        
        results[column] = {
            'themes': themes,
            'co_occurrence': co_occurrence,
            'wordcloud': wordcloud
        }
    
    return results

def visualize_analysis(df, attributes, yes_no_questions, text_columns):
    """分析結果の可視化"""
    # 属性データの分析
    stats, cross_tabs, attribute_ranked = analyze_attributes(df, attributes)
    # 2択質問の分析
    response_dist, trend_analysis, chi2_results = analyze_yes_no_questions(
        df, yes_no_questions, attributes
    )
    # 自由記述の分析
    text_analysis = analyze_free_text(df, text_columns)
    return {
        'stats': stats,
        'cross_tabs': cross_tabs,
        'attribute_ranked': attribute_ranked,
        'response_dist': response_dist,
        'trend_analysis': trend_analysis,
        'chi2_results': chi2_results,
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
    yes_no_questions = [col for col in df.columns if '⚫︎' in col]
    text_columns = [col for col in df.columns if '・' in col]
    
    # 分析の実行
    analysis_results = visualize_analysis(df, attributes, yes_no_questions, text_columns)
    
    # タブの作成
    tab_attributes, tab_yes_no, tab_text, tab_summary = st.tabs([
        "1. 属性分析",
        "2. 2択質問分析",
        "3. 自由記述分析",
        "4. 総合分析"
    ])
    
    # 1. 属性分析タブ
    with tab_attributes:
        st.markdown("### 1. 属性分析")
        subtab_dist, subtab_cross = st.tabs(["属性ごとの分布（円グラフ）", "クロス集計"])
        with subtab_dist:
            # 3列のレイアウトを作成
            cols = st.columns(3)
            col_index = 0
            
            for attr in attributes:
                if attr in MULTIPLE_CHOICE_QUESTIONS:
                    # 複数回答の質問は順位ごとに独立した図を表示
                    rank_distributions = analysis_results['attribute_ranked'].get(attr, {})
                    rank_keys = [k for k in rank_distributions.keys() if k != '全体']
                    
                    for rank in sorted(rank_keys, key=lambda x: int(x.replace('位','')) if x.endswith('位') else 999):
                        with cols[col_index % 3]:
                            rank_dist = rank_distributions[rank]
                            st.markdown(f"###### {attr} - {rank}")
                            fig = px.pie(
                                values=rank_dist.values,
                                names=rank_dist.index,
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
                        st.markdown(f"###### {attr}")
                        fig = px.pie(
                            values=list(stat['distribution'].values()),
                            names=list(stat['distribution'].keys()),
                            width=400,
                            height=400
                        )
                        fig.update_layout(
                            uniformtext_minsize=12,
                            uniformtext_mode='hide'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    col_index += 1
        with subtab_cross:
            for key, cross_tab in analysis_results['cross_tabs'].items():
                st.markdown(f"##### {key}")
                st.write(cross_tab)
    
    # 2. 2択質問分析タブ
    with tab_yes_no:
        st.markdown("### 2. 2択質問分析")
        
        # 回答分布の表示
        st.markdown("#### 回答分布")
        for question, dist in analysis_results['response_dist'].items():
            if question in MULTIPLE_CHOICE_QUESTIONS:
                # 複数回答の質問は順位ごとの円グラフ
                st.markdown(f"##### {question}")
                for rank, rank_dist in dist.items():
                    st.markdown(f"###### {rank}")
                    fig = px.pie(
                        values=rank_dist.values,
                        names=rank_dist.index,
                        width=400,
                        height=400
                    )
                    fig.update_layout(
                        uniformtext_minsize=12,
                        uniformtext_mode='hide'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                # 通常の2択質問
                fig = px.pie(
                    values=dist.values,
                    names=dist.index,
                    title=question,
                    width=400,
                    height=400
                )
                fig.update_layout(
                    uniformtext_minsize=12,
                    uniformtext_mode='hide'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # 傾向分析の表示
        st.markdown("#### 傾向分析")
        for key, trend in analysis_results['trend_analysis'].items():
            st.markdown(f"##### {key}")
            fig = px.imshow(
                trend,
                text_auto=".2f",
                aspect="auto"
            )
            st.plotly_chart(fig)
        
        # カイ二乗検定結果の表示
        st.markdown("#### 統計的有意差の検定")
        for key, result in analysis_results['chi2_results'].items():
            st.markdown(f"##### {key}")
            st.write(f"カイ二乗値: {result['chi2']:.2f}")
            st.write(f"p値: {result['p_value']:.4f}")
            st.write(f"自由度: {result['dof']}")
    
    # 3. 自由記述分析タブ
    with tab_text:
        st.markdown("### 3. 自由記述分析")
        
        for column, analysis in analysis_results['text_analysis'].items():
            st.markdown(f"#### {column}")
            
            # テーマの表示
            st.markdown("##### 主要テーマ")
            for theme in analysis['themes']:
                st.write(f"- {theme}")
            
            # ワードクラウドの表示
            st.markdown("##### ワードクラウド")
            plt.figure(figsize=(10, 5))
            plt.imshow(analysis['wordcloud'], interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)
    
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