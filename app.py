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
    # 属性情報の列名
    attribute_columns = ['学年', '性別', '学部系統']
    
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

if uploaded_file is not None:
    # CSVファイルの読み込み
    df = pd.read_csv(uploaded_file)
    
    # 質問項目の抽出
    questions = extract_questions(df)
    
    # タブの作成
    tab_attributes, tab_yes_no, tab_free_answers, tab_analysis = st.tabs([
        "回答者属性", "2択質問", "自由記述", "総合分析"
    ])
    
    # 1. 回答者属性タブ
    with tab_attributes:
        st.header("回答者属性")
        col1, col2, col3 = st.columns(3)
        
        for i, attr in enumerate(questions['attributes']):
            col = [col1, col2, col3][i]
            with col:
                counts = df[attr].value_counts()
                fig = px.pie(
                    values=counts.values,
                    names=counts.index,
                    title=f'{attr}分布'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # 2. 2択質問タブ
    with tab_yes_no:
        st.header("2択質問の回答")
        cols = st.columns(2)
        for i, question in enumerate(questions['yes_no']):
            col = cols[i % 2]
            with col:
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
                
                st.plotly_chart(fig, use_container_width=True)
    
    # 3. 自由記述タブ
    with tab_free_answers:
        st.header("自由記述回答")
        
        # 自由記述回答の分析結果を保存
        free_text_analysis = {}
        
        for field in questions['free_answers']:
            st.subheader(field)
            
            # 回答の取得と前処理
            answers = df[field].dropna().tolist()
            total_responses = len(answers)
            
            if total_responses > 0:
                # 回答数の表示
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
                
                for i in range(0, len(answers), MAX_ANSWERS_PER_BATCH):
                    batch_answers = answers[i:i + MAX_ANSWERS_PER_BATCH]
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
        st.header("総合分析")
        
        # 総合評価
        st.subheader("1. 総合評価")
        
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
        
        # 現状の強み
        st.subheader("2. 現状の強み")
        
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
        
        # 主要な課題
        st.subheader("3. 主要な課題")
        
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
        
        # 改善提案とアクションプラン
        st.subheader("4. 改善提案とアクションプラン")
        
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