import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import numpy as np
import openai
import os
import json

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

# タイトル
st.title("採用動画アンケート結果分析")

# ファイルアップロード
uploaded_file = st.file_uploader("CSVファイルをアップロード", type=['csv'])

def analyze_with_openai(text, prompt):
    """OpenAI APIを使用してテキストを分析する"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "あなたは採用動画の分析の専門家です。与えられたデータから、客観的で具体的な分析を行ってください。"},
                {"role": "user", "content": f"{prompt}\n\n分析対象:\n{text}"}
            ],
            temperature=0.3,  # より決定論的な応答に
            max_tokens=2000   # トークン数を増やす
        )
        return response.choices[0].message['content']
    except Exception as e:
        st.error(f"OpenAI APIの呼び出し中にエラーが発生しました: {str(e)}")
        return None

def extract_questions(df):
    """CSVファイルから質問項目を抽出する"""
    # 属性情報の列名（▪︎を含む列）
    attribute_columns = [col for col in df.columns if '▪︎' in col]
    
    # はい/いいえ質問の列名を抽出（文頭が⚫︎で始まる列）
    yes_no_columns = [col for col in df.columns if col.startswith('⚫︎')]
    
    # その他の列を抽出
    other_columns = [col for col in df.columns if col not in attribute_columns and col not in yes_no_columns]
    
    # 自由記述回答の列名を抽出（その他の列から選択）
    free_answer_columns = [col for col in other_columns if df[col].dtype == 'object']
    
    return {
        'attributes': attribute_columns,
        'yes_no': yes_no_columns,
        'free_answers': free_answer_columns
    }

def process_multiple_answers(df, column):
    """複数回答を処理する"""
    # 回答を文字列として取得
    answers = df[column].astype(str)
    
    # カンマで分割して個別の回答に分解
    all_answers = []
    for answer in answers:
        # カンマで分割し、空白を削除
        split_answers = [a.strip() for a in answer.split(',')]
        all_answers.extend(split_answers)
    
    # 回答の集計
    answer_counts = pd.Series(all_answers).value_counts()
    
    return answer_counts

if uploaded_file is not None:
    # CSVファイルの読み込み
    df = pd.read_csv(uploaded_file)
    
    # 質問項目の抽出
    questions = extract_questions(df)
    
    # タブの作成
    tab_attributes, tab_yes_no, tab_free_answers, tab_analysis = st.tabs([
        "1. 回答者属性", "2. 2択質問", "3. 自由記述", "4. 総合分析"
    ])
    
    # 1. 回答者属性タブ
    with tab_attributes:
        st.markdown("### 1. 回答者属性")
        st.markdown("回答者の基本的な属性（学年、性別、学部系統）の分布を分析します。")
        
        # 属性タブの下のサブタブ
        subtab_overview, subtab_details = st.tabs([
            "1-1. 属性別分布", 
            "1-2. クロス分析"
        ])
        
        with subtab_overview:
            st.markdown("#### 1-1. 属性別分布")
            st.markdown("各属性（学年、性別、学部系統）の回答者分布を円グラフで表示します。")
            
            # 3列のコンテナを作成
            cols = st.columns(3)
            
            # 各属性を3列で表示
            for i, attr in enumerate(questions['attributes']):
                with cols[i % 3]:
                    # 複数回答の処理
                    counts = process_multiple_answers(df, attr)
                    fig = px.pie(
                        values=counts.values,
                        names=counts.index,
                        title=f'{attr}分布'
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"pie_chart_{attr}")
        
        with subtab_details:
            st.markdown("#### 1-2. クロス分析")
            st.markdown("属性間の関係性を詳細に分析します。")
            
            # 3列のコンテナを作成
            cols = st.columns(3)
            
            # 各属性を3列で表示
            for i, attr in enumerate(questions['attributes']):
                with cols[i % 3]:
                    st.markdown(f"**{attr}**")
                    # 複数回答の処理
                    counts = process_multiple_answers(df, attr)
                    st.write(counts)
                    st.write("---")
    
    # 2. 2択質問タブ
    with tab_yes_no:
        st.markdown("### 2. 2択質問の分析")
        st.markdown("「はい/いいえ」で回答する質問の結果を分析します。")
        
        # 2択質問タブの下のサブタブ
        subtab_charts, subtab_trends = st.tabs([
            "2-1. 回答分布", 
            "2-2. 属性別傾向"
        ])
        
        with subtab_charts:
            st.markdown("#### 2-1. 回答分布")
            st.markdown("各質問に対する「はい/いいえ」の回答分布を円グラフで表示します。")
            
            # 3列のコンテナを作成
            cols = st.columns(3)
            
            # 各質問を3列で表示
            for i, question in enumerate(questions['yes_no']):
                with cols[i % 3]:
                    yes_count = (df[question] == 'はい').sum()
                    no_count = (df[question] == 'いいえ').sum()
                    total = yes_count + no_count
                    yes_percentage = (yes_count / total) * 100
                    
                    fig = go.Figure(data=[go.Pie(
                        labels=['はい', 'いいえ'],
                        values=[yes_count, no_count],
                        marker_colors=['#4CAF50', '#F44336']
                    )])
                    
                    fig.update_layout(
                        title=f"{question}<br>はい: {yes_percentage:.1f}%",
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True, key=f"pie_chart_{question}")
        
        with subtab_trends:
            st.markdown("#### 2-2. 属性別傾向")
            st.markdown("回答者の属性（学年、性別、学部系統）ごとの回答傾向を分析します。")
            
            # 質問ごとのサブタブを作成
            question_tabs = st.tabs([f"Q{i+1}: {q}" for i, q in enumerate(questions['yes_no'])])
            
            for i, (question, tab) in enumerate(zip(questions['yes_no'], question_tabs)):
                with tab:
                    st.markdown(f"##### {question}")
                    for attr in questions['attributes']:
                        st.markdown(f"**{attr}別の回答分布:**")
                        cross_tab = pd.crosstab(df[attr], df[question], normalize='index') * 100
                        st.write(cross_tab.round(1))
                        st.write("---")
    
    # 3. 自由記述タブ
    with tab_free_answers:
        st.markdown("### 3. 自由記述回答の分析")
        st.markdown("自由記述形式で回答された意見や感想を分析します。")
        
        # 自由記述回答の分析結果を保存
        free_text_analysis = {}
        
        # 複数回答の質問を特定
        multiple_choice_questions = [
            "企業を選ぶ際に重視するポイント",
            "イキイキ働いていると感じる状態",
            "働きがいを感じる時",
            "就活情報源"
        ]
        
        # 質問ごとのサブタブを作成
        question_tabs = st.tabs([f"Q{i+1}: {q}" for i, q in enumerate(questions['free_answers'])])
        
        for i, (field, tab) in enumerate(zip(questions['free_answers'], question_tabs)):
            with tab:
                # 回答の取得と前処理
                answers = df[field].dropna().tolist()
                total_responses = len(answers)
                
                if total_responses > 0:
                    # 複数回答の質問かどうかを判定
                    is_multiple_choice = any(q in field for q in multiple_choice_questions)
                    
                    if is_multiple_choice:
                        # 複数回答の処理
                        counts = process_multiple_answers(df, field)
                        
                        # 円グラフで表示
                        fig = px.pie(
                            values=counts.values,
                            names=counts.index,
                            title=f'{field}の分布'
                        )
                        st.plotly_chart(fig, use_container_width=True, key=f"pie_chart_{field}")
                        
                        # 集計結果を表で表示
                        st.markdown("**回答の集計:**")
                        st.write(counts)
                    else:
                        # 自由記述タブの下のサブタブ
                        subtab_grouped, subtab_raw = st.tabs([
                            f"3-1. 類似回答のグループ化 ({field})", 
                            f"3-2. 個別回答一覧 ({field})"
                        ])
                        
                        with subtab_grouped:
                            st.markdown(f"##### {field}")
                            st.markdown(f"**回答数: {total_responses}件**")
                            
                            # 類似回答のグループ化（AI活用）
                            grouping_prompt = """
                            以下の回答を意味的な類似性に基づいて3-5個のグループに分類してください。
                            各グループには、そのグループを代表する回答と、類似する他の回答を含めてください。
                            回答は原文のまま使用し、要約や言い換えは避けてください。

                            出力は以下のJSON形式で返してください：
                            {
                                "groups": [
                                    {
                                        "theme": "グループを表す短いテーマ",
                                        "representative": "最も代表的な回答（原文）",
                                        "similar_responses": ["類似回答1", "類似回答2", ...],
                                        "similar_count": 類似回答の数
                                    }
                                ]
                            }
                            """
                            
                            # 回答をAIに送信（長すぎる場合は分割して処理）
                            MAX_ANSWERS_PER_BATCH = 30  # バッチサイズを減らす
                            all_groups = []
                            
                            for j in range(0, len(answers), MAX_ANSWERS_PER_BATCH):
                                batch_answers = answers[j:j + MAX_ANSWERS_PER_BATCH]
                                analysis_text = "\n".join([f"- {answer}" for answer in batch_answers])
                                group_result = analyze_with_openai(analysis_text, grouping_prompt)
                                
                                if group_result:
                                    try:
                                        # JSONの前後の余分な文字列を削除
                                        json_str = group_result.strip()
                                        if not json_str.startswith('{'):
                                            json_str = json_str[json_str.find('{'):]
                                        if not json_str.endswith('}'):
                                            json_str = json_str[:json_str.rfind('}')+1]
                                        
                                        result_dict = json.loads(json_str)
                                        if 'groups' in result_dict:
                                            all_groups.extend(result_dict['groups'])
                                    except json.JSONDecodeError as e:
                                        st.error(f"回答の分類中にJSONパースエラーが発生しました: {str(e)}")
                                        continue
                                    except Exception as e:
                                        st.error(f"回答の分類中に予期せぬエラーが発生しました: {str(e)}")
                                        continue
                            
                            if all_groups:
                                # グループごとに表示
                                for group in all_groups:
                                    st.markdown(f"##### {group.get('theme', '未分類グループ')}")
                                    
                                    # 代表的な回答を表示
                                    st.markdown("**代表的な回答:**")
                                    st.write(group.get('representative', ''))
                                    
                                    # 類似回答を表示
                                    similar_responses = group.get('similar_responses', [])
                                    if similar_responses:
                                        similar_count = len(similar_responses)
                                        st.markdown(f"**類似する回答 ({similar_count}件):**")
                                        for response in similar_responses:
                                            st.write(f"- {response}")
                                
                                st.write("---")
                            
                            # 分類されなかった回答（ユニークな意見）
                            all_grouped_answers = set()
                            for group in all_groups:
                                all_grouped_answers.add(group.get('representative', ''))
                                all_grouped_answers.update(group.get('similar_responses', []))
                            
                            unique_answers = [ans for ans in answers if ans not in all_grouped_answers]
                            if unique_answers:
                                st.markdown("##### ユニークな回答")
                                sample_size = min(10, len(unique_answers))
                                sample_unique = np.random.choice(unique_answers, size=sample_size, replace=False)
                                for answer in sample_unique:
                                    st.write(f"- {answer}")
                        
                        with subtab_raw:
                            st.markdown(f"##### {field}")
                            st.markdown(f"**回答数: {total_responses}件**")
                            # 全ての回答をそのまま表示
                            for j, answer in enumerate(answers, 1):
                                st.write(f"{j}. {answer}")
                else:
                    st.info("このフィールドには回答がありません。")
                
                # 分析結果を保存（総合分析タブで使用）
                free_text_analysis[field] = {
                    'total': total_responses,
                    'all_answers': answers,
                    'grouped_answers': all_groups if total_responses > 0 else [],
                    'unique_answers': unique_answers if total_responses > 0 else []
                }
                
                st.write("---")
    
    # 4. 総合分析タブ
    with tab_analysis:
        st.markdown("### 4. 総合分析・改善提案")
        st.markdown("全ての回答データを統合的に分析し、改善案を提示します。")
        
        # 総合分析タブの下のサブタブ
        subtab_summary, subtab_strengths, subtab_challenges, subtab_actions = st.tabs([
            "4-1. 総合評価", 
            "4-2. 現状の強み", 
            "4-3. 主要な課題", 
            "4-4. 改善提案"
        ])
        
        with subtab_summary:
            st.markdown("#### 4-1. 総合評価")
            st.markdown("動画全体の評価を主要な指標に基づいて分析します。")
            
            # 動画の視聴率（サムネイルの目立ち度）
            thumbnail_question = next((q for q in questions['yes_no'] if 'サムネイル' in q), None)
            if thumbnail_question:
                thumbnail_yes = (df[thumbnail_question] == 'はい').sum()
                thumbnail_total = len(df)
                thumbnail_percentage = (thumbnail_yes / thumbnail_total) * 100
            else:
                thumbnail_percentage = 0
            
            # 冒頭の引き込み力
            intro_question = next((q for q in questions['yes_no'] if '冒頭' in q), None)
            if intro_question:
                intro_yes = (df[intro_question] == 'はい').sum()
                intro_percentage = (intro_yes / thumbnail_total) * 100
            else:
                intro_percentage = 0
            
            # 会社の雰囲気伝達
            atmosphere_question = next((q for q in questions['yes_no'] if '雰囲気' in q), None)
            if atmosphere_question:
                atmosphere_yes = (df[atmosphere_question] == 'はい').sum()
                atmosphere_percentage = (atmosphere_yes / thumbnail_total) * 100
            else:
                atmosphere_percentage = 0
            
            # 応募意向
            apply_question = next((q for q in questions['yes_no'] if '応募' in q), None)
            if apply_question:
                apply_yes = (df[apply_question] == 'はい').sum()
                apply_percentage = (apply_yes / thumbnail_total) * 100
            else:
                apply_percentage = 0
            
            # 総合評価のテキスト生成
            evaluation_text = f"""
            動画の視聴率: {thumbnail_percentage:.1f}%
            冒頭の引き込み力: {intro_percentage:.1f}%
            会社の雰囲気伝達: {atmosphere_percentage:.1f}%
            応募意向: {apply_percentage:.1f}%
            """
            
            # OpenAIによる総合評価の分析
            evaluation_prompt = """
            以下の数値データから、採用動画の総合評価を行ってください。
            具体的な数値に基づいて、強みと改善点を明確に示してください。
            """
            evaluation_result = analyze_with_openai(evaluation_text, evaluation_prompt)
            
            if evaluation_result:
                st.write(evaluation_result)
        
        with subtab_strengths:
            st.markdown("#### 4-2. 現状の強み")
            st.markdown("採用動画の特に効果的な要素と成功している点を分析します。")
            
            # 印象の分析から強みを抽出
            impression_field = next((f for f in questions['free_answers'] if '印象' in f), None)
            if impression_field and impression_field in free_text_analysis:
                impression_analysis = free_text_analysis[impression_field]
                top_impressions = [imp[0] for imp in impression_analysis['top_answers'][:3]]
                
                # 強みの分析テキスト生成
                strengths_text = f"""
                主な印象: {', '.join(top_impressions)}
                動画の評価指標:
                - 引き込み力: {intro_percentage:.1f}%
                - 雰囲気伝達: {atmosphere_percentage:.1f}%
                """
                
                # OpenAIによる強みの分析
                strengths_prompt = """
                以下のデータから、企業の強みを分析してください。
                特に採用動画を通じて伝わっている企業の特徴や魅力を具体的に指摘してください。
                """
                strengths_result = analyze_with_openai(strengths_text, strengths_prompt)
                
                if strengths_result:
                    st.write(strengths_result)
        
        with subtab_challenges:
            st.markdown("#### 4-3. 主要な課題")
            st.markdown("改善が必要な点や、視聴者が求める情報とのギャップを分析します。")
            
            # 欲しい情報の分析から課題を抽出
            desired_info_field = next((f for f in questions['free_answers'] if '情報' in f), None)
            if desired_info_field and desired_info_field in free_text_analysis:
                desired_info = free_text_analysis[desired_info_field]
                top_desired_info = [info[0] for info in desired_info['top_answers'][:3]]
                
                # 課題の分析テキスト生成
                challenges_text = f"""
                不足している情報: {', '.join(top_desired_info)}
                現在の指標:
                - 応募意向: {apply_percentage:.1f}%
                - 情報充実度: {atmosphere_percentage:.1f}%
                """
                
                # OpenAIによる課題の分析
                challenges_prompt = """
                以下のデータから、採用動画の主要な課題を分析してください。
                特に改善が必要な点や、学生が求める情報とのギャップを具体的に指摘してください。
                """
                challenges_result = analyze_with_openai(challenges_text, challenges_prompt)
                
                if challenges_result:
                    st.write(challenges_result)
        
        with subtab_actions:
            st.markdown("#### 4-4. 改善提案")
            st.markdown("短期・中期・長期の具体的な改善案とアクションプランを提示します。")
            
            if 'challenges_result' in locals():
                # 短期的な改善案
                st.markdown("#### 短期的な改善案（1-3ヶ月）")
                short_term_prompt = f"""
                以下の課題と指標に基づいて、短期的（1-3ヶ月）な改善案を提示してください。
                すぐに着手可能で、比較的早く効果が出る施策を優先的に提案してください。
                
                主な課題:
                {challenges_result}
                
                現在の指標:
                - 引き込み力: {intro_percentage:.1f}%
                - 応募意向: {apply_percentage:.1f}%
                """
                short_term_result = analyze_with_openai(short_term_prompt, "短期的な改善案を提案してください。")
                if short_term_result:
                    st.write(short_term_result)
                
                # 中期的な改善案
                st.markdown("#### 中期的な改善案（3-6ヶ月）")
                mid_term_prompt = f"""
                以下の課題と指標に基づいて、中期的（3-6ヶ月）な改善案を提示してください。
                準備に時間がかかるが、より本質的な改善につながる施策を提案してください。
                
                主な課題:
                {challenges_result}
                
                現在の指標:
                - 引き込み力: {intro_percentage:.1f}%
                - 応募意向: {apply_percentage:.1f}%
                """
                mid_term_result = analyze_with_openai(mid_term_prompt, "中期的な改善案を提案してください。")
                if mid_term_result:
                    st.write(mid_term_result)
                
                # 長期的な改善案
                st.markdown("#### 長期的な改善案（6ヶ月-1年）")
                long_term_prompt = f"""
                以下の課題と指標に基づいて、長期的（6ヶ月-1年）な改善案を提示してください。
                採用動画の根本的な改善や、採用戦略全体の見直しにつながる施策を提案してください。
                
                主な課題:
                {challenges_result}
                
                現在の指標:
                - 引き込み力: {intro_percentage:.1f}%
                - 応募意向: {apply_percentage:.1f}%
                """
                long_term_result = analyze_with_openai(long_term_prompt, "長期的な改善案を提案してください。")
                if long_term_result:
                    st.write(long_term_result) 