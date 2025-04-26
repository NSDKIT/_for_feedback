import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import chi2_contingency
import networkx as nx
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import japanize_matplotlib
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import re
from collections import Counter
import itertools

# OpenAI APIキーの設定
try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
except Exception as e:
    st.error(f"OpenAI APIキーの設定中にエラーが発生しました: {str(e)}")
    st.stop()

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
    """属性データの分析"""
    # 1. 基本統計量の計算
    stats = df[attributes].describe()
    
    # 2. クロス集計
    cross_tabs = {}
    for attr1, attr2 in itertools.combinations(attributes, 2):
        cross_tabs[f"{attr1}_vs_{attr2}"] = pd.crosstab(df[attr1], df[attr2])
    
    # 3. 相関分析
    correlation = df[attributes].corr()
    
    return stats, cross_tabs, correlation

def analyze_yes_no_questions(df, yes_no_questions, attributes):
    """2択質問の分析"""
    # 1. 回答分布の集計
    response_dist = {}
    for question in yes_no_questions:
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
    # 1. 属性データの分析
    stats, cross_tabs, correlation = analyze_attributes(df, attributes)
    
    # 2. 2択質問の分析
    response_dist, trend_analysis, chi2_results = analyze_yes_no_questions(
        df, yes_no_questions, attributes
    )
    
    # 3. 自由記述の分析
    text_analysis = analyze_free_text(df, text_columns)
    
    return {
        'stats': stats,
        'cross_tabs': cross_tabs,
        'correlation': correlation,
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
        
        # 基本統計量の表示
        st.markdown("#### 基本統計量")
        st.write(analysis_results['stats'])
        
        # 相関分析の表示
        st.markdown("#### 相関分析")
        fig = px.imshow(
            analysis_results['correlation'],
            text_auto=".2f",
            aspect="auto"
        )
        st.plotly_chart(fig)
        
        # クロス集計の表示
        st.markdown("#### クロス集計")
        for key, cross_tab in analysis_results['cross_tabs'].items():
            st.markdown(f"##### {key}")
            st.write(cross_tab)
    
    # 2. 2択質問分析タブ
    with tab_yes_no:
        st.markdown("### 2. 2択質問分析")
        
        # 回答分布の表示
        st.markdown("#### 回答分布")
        for question, dist in analysis_results['response_dist'].items():
            fig = px.pie(
                values=dist.values,
                names=dist.index,
                title=question
            )
            st.plotly_chart(fig)
        
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