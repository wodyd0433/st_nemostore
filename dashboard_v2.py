import streamlit as st
import pandas as pd
import sqlite3
import os
import json
import re
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as px_go
import matplotlib.pyplot as plt
import koreanize_matplotlib

# 페이지 설정
st.set_page_config(page_title="Nemostore Pro EDA Dashboard", layout="wide")

# 프로젝트 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "nemo.db")

# ------------------------------------------------------------------------------
# 1. 데이터 로드 및 전처리 모듈
# ------------------------------------------------------------------------------

@st.cache_data(show_spinner="DB에서 데이터를 불러오는 중...")
def load_raw_data(limit=None):
    if not os.path.exists(DB_PATH):
        return pd.DataFrame(), pd.DataFrame()
    
    conn = sqlite3.connect(DB_PATH)
    items_query = "SELECT * FROM nemo_items"
    if limit:
        items_query += f" LIMIT {limit}"
    
    df_items = pd.read_sql(items_query, conn)
    df_agents = pd.read_sql("SELECT * FROM nemo_agents", conn)
    conn.close()
    
    return df_items, df_agents

def preprocess_data(df_items):
    if df_items.empty:
        return df_items
    
    df = df_items.copy()
    
    # 날짜 파싱
    date_cols = ['createdDateUtc', 'editedDateUtc', 'confirmedDateUtc', 'completionConfirmedDateUtc']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # 사진 개수 파생
    def count_json_list(x):
        try:
            return len(json.loads(x)) if x and x != 'null' else 0
        except:
            return 0
            
    df['photo_count_small'] = df['smallPhotoUrls'].apply(count_json_list)
    df['photo_count_origin'] = df['originPhotoUrls'].apply(count_json_list)
    df['has_photos'] = df['photo_count_small'] > 0
    
    # 금액/면적 기반 지표 (1단기 = 1,000원 -> 만 원 변환: x * 0.1)
    money_cols = ['deposit', 'monthlyRent', 'maintenanceFee', 'premium', 'sale', 'firstDeposit', 'firstMonthlyRent', 'firstPremium']
    for col in money_cols:
        if col in df.columns:
            df[col] = df[col] * 0.1
            
    # 파생변수 생성
    df['monthly_total_cost'] = df['monthlyRent'].fillna(0) + df['maintenanceFee'].fillna(0)
    df['rent_per_size'] = df['monthlyRent'] / df['size'].replace(0, pd.NA)
    df['total_cost_per_size'] = df['monthly_total_cost'] / df['size'].replace(0, pd.NA)
    df['has_premium'] = df['premium'] > 0
    df['is_basement'] = df['floor'] < 0
    
    # 역 정보 파싱 (예: "양재(서초구청)역, 도보 12분")
    def parse_station(val):
        if not val or pd.isna(val): return None, None
        match_min = re.search(r'도보\s*(\d+)분', str(val))
        minutes = int(match_min.group(1)) if match_min else None
        station = str(val).split(',')[0].replace('역', '').strip()
        return station, minutes

    station_data = df['nearSubwayStation'].apply(parse_station)
    df['station_name'] = [x[0] for x in station_data]
    df['walk_minutes'] = [x[1] for x in station_data]
    
    # 텍스트 파생
    df['title_len'] = df['title'].str.len().fillna(0)
    df['title_has_special'] = df['title'].str.contains(r'[★♥◆■●▶◀]', na=False)
    
    # 키워드 플래그
    keywords = ["무권리", "가성비", "인테리어", "주차", "룸", "대로변", "신축", "역세권"]
    for kw in keywords:
        df[f'kw_{kw}'] = df['title'].str.contains(kw, na=False)
        
    return df

# ------------------------------------------------------------------------------
# 2. UI 및 사이드바 필터
# ------------------------------------------------------------------------------

def main():
    st.title("🏙️ Nemostore Professional EDA Dashboard")
    st.caption("DB-only Explanatory Data Analysis | Built by Antigravity")
    
    # 데이터 로딩
    limit_opt = st.sidebar.number_input("조회 레코드 수 제한 (0은 전체)", 0, 10000, 0)
    df_raw, df_agents = load_raw_data(limit=limit_opt if limit_opt > 0 else None)
    
    if df_raw.empty:
        st.warning("DB에 데이터가 없습니다. 수집 스크립트를 먼저 실행하세요.")
        return
        
    df = preprocess_data(df_raw)
    
    # 사이드바 필터
    st.sidebar.header("🎯 Global Filters")
    
    # 업종 필터
    cat_large = st.sidebar.multiselect("업종 대분류", df['businessLargeCodeName'].unique())
    if cat_large:
        df = df[df['businessLargeCodeName'].isin(cat_large)]
        
    # 가격 슬라이더 (만 원 단위)
    rent_range = st.sidebar.slider("월세 범위 (만)", 0, int(df['monthlyRent'].max() or 1000), (0, 1000))
    df = df[(df['monthlyRent'] >= rent_range[0]) & (df['monthlyRent'] <= rent_range[1])]
    
    # 면적 슬라이더
    size_range = st.sidebar.slider("면적 범위 (㎡)", 0.0, float(df['size'].max() or 300.0), (0.0, 300.0))
    df = df[(df['size'] >= size_range[0]) & (df['size'] <= size_range[1])]
    
    # 기타 토글
    col_t1, col_t2 = st.sidebar.columns(2)
    with col_t1:
        f_premium = st.checkbox("권리금 있음")
    with col_t2:
        f_photo = st.checkbox("사진 있음")
        
    if f_premium: df = df[df['has_premium']]
    if f_photo: df = df[df['has_photos']]
    
    # 탭 구성
    tabs = st.tabs([
        "📊 Overview", "🧪 Data Quality", "📌 Univariate", "🔗 Bivariate", 
        "⏳ Time Analysis", "📝 Text/Subway EDA", "🏢 Agents", "🔎 Record Explorer"
    ])

    # --------------------------------------------------------------------------
    # 5.1 Overview
    # --------------------------------------------------------------------------
    with tabs[0]:
        st.header("Overview")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("총 매물 (필터후/전체)", f"{len(df)} / {len(df_raw)}")
        m2.metric("중앙값 월세", f"{df['monthlyRent'].median():.0f}만")
        m3.metric("평균 면적", f"{df['size'].mean():.1f}㎡")
        m4.metric("지하 비중", f"{(df['floor'] < 0).mean()*100:.1f}%")
        
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(df, x="monthlyRent", title="월세 분포 (Histogram)", nbins=50, color_discrete_sequence=['#ff4b4b'])
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            cat_counts = df['businessLargeCodeName'].value_counts().reset_index()
            fig = px.pie(cat_counts, values='count', names='businessLargeCodeName', title="업종 대분류 비중")
            st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------------------------------
    # 5.2 Data Quality
    # --------------------------------------------------------------------------
    with tabs[1]:
        st.header("Data Quality Analysis")
        null_counts = df_raw.isnull().mean() * 100
        dq_df = pd.DataFrame({
            "Column": null_counts.index,
            "Missing %": null_counts.values,
            "Unique Count": [df_raw[c].nunique() for c in null_counts.index],
            "Dtype": [str(df_raw[c].dtype) for c in null_counts.index]
        }).sort_values("Missing %", ascending=False)
        
        st.subheader("컬럼별 결측률 및 유니크 수")
        st.dataframe(dq_df, use_container_width=True)
        
        st.subheader("데이터 일관성 체크 (Consistency Warnings)")
        warnings = []
        if (df['floor'] > df['groundFloor']).any():
            warnings.append(f"⚠️ [층수 오류] 현재 층이 건물의 총 층수보다 높은 매물: {len(df[df['floor'] > df['groundFloor']])}건")
        if (df['monthlyRent'] == 0).any() and (df['deposit'] > 0).any():
            warnings.append(f"ℹ️ [순수 전세/보증금 매물] 월세가 0원인 매물: {len(df[df['monthlyRent'] == 0])}건")
        
        if warnings:
            for w in warnings: st.write(w)
        else:
            st.success("데이터 일관성에 큰 이슈가 발견되지 않았습니다.")

    # --------------------------------------------------------------------------
    # 5.3 Univariate
    # --------------------------------------------------------------------------
    with tabs[2]:
        st.header("단변량 분석 (Univariate)")
        target_col = st.selectbox("분석할 수치형 컬럼 선택", ['monthlyRent', 'deposit', 'maintenanceFee', 'monthly_total_cost', 'size', 'areaPrice', 'viewCount'])
        
        u_col1, u_col2 = st.columns([1, 2])
        with u_col1:
            st.write(df[target_col].describe())
        with u_col2:
            fig = px.box(df, y=target_col, title=f"{target_col} Box Plot", points="all")
            st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------------------------------
    # 5.4 Bivariate
    # --------------------------------------------------------------------------
    with tabs[3]:
        st.header("관계 분석 (Bivariate)")
        
        st.subheader("보증금 vs 월세 (산점도)")
        fig = px.scatter(df, x="deposit", y="monthlyRent", color="businessLargeCodeName", 
                         size="size", hover_data=['title'], title="Size-weighted Scatter: Deposit vs Rent")
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("상관관계 히트맵")
        numeric_df = df.select_dtypes(include=['number']).drop(columns=['id', 'number', 'articleType', 'state'], errors='ignore')
        corr = numeric_df.corr()
        fig = px.imshow(corr, text_auto=True, title="Correlation Matrix")
        st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------------------------------
    # 5.5 Time Analysis
    # --------------------------------------------------------------------------
    with tabs[4]:
        st.header("시계열 분석 (Time Analysis)")
        if not df['createdDateUtc'].isnull().all():
            df_time = df.set_index('createdDateUtc').resample('D').size().reset_index(name='count')
            fig = px.line(df_time, x='createdDateUtc', y='count', title="일별 신규 등록 매물 수")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("등록일(createdDateUtc) 데이터가 부족하여 시계열을 표시할 수 없습니다.")

    # --------------------------------------------------------------------------
    # 5.6 Text/Subway EDA
    # --------------------------------------------------------------------------
    with tabs[5]:
        st.header("Text & Subway EDA")
        t_col1, t_col2 = st.columns(2)
        
        with t_col1:
            st.subheader("주요 키워드 포함 비율")
            kw_cols = [c for c in df.columns if c.startswith('kw_')]
            kw_means = df[kw_cols].mean().sort_values(ascending=False) * 100
            fig = px.bar(x=kw_means.index.str.replace('kw_', ''), y=kw_means.values, title="Keyword Prevalence (%)")
            st.plotly_chart(fig, use_container_width=True)
            
        with t_col2:
            st.subheader("인근 지하철역 TOP 10")
            station_counts = df['station_name'].value_counts().head(10)
            fig = px.bar(station_counts, title="Top 10 Nearby Stations")
            st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------------------------------
    # 5.7 Agents
    # --------------------------------------------------------------------------
    with tabs[6]:
        st.header("중개사 분석 (Agents)")
        if not df_agents.empty:
            ag_col1, ag_col2 = st.columns(2)
            with ag_col1:
                st.subheader("중개사별 매물 보유량 분포")
                fig = px.histogram(df_agents, x="publicArticleCount", title="Agent Article Count Distribution")
                st.plotly_chart(fig, use_container_width=True)
            with ag_col2:
                st.subheader("중개사 상세 정보 샘플")
                st.dataframe(df_agents[['name', 'nameOfRepresentative', 'publicArticleCount']].head(10), use_container_width=True)
        else:
            st.info("수집된 중개사 정보가 없습니다.")

    # --------------------------------------------------------------------------
    # 5.8 Record Explorer
    # --------------------------------------------------------------------------
    with tabs[7]:
        st.header("Record Explorer")
        st.subheader("필터링된 상세 데이터")
        explore_cols = ['id', 'number', 'title', 'businessMiddleCodeName', 'deposit', 'monthlyRent', 'maintenanceFee', 'size', 'floor', 'station_name', 'walk_minutes']
        st.dataframe(df[explore_cols], use_container_width=True)
        
        csv = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("결과 CSV 다운로드", csv, "nemostore_eda_export.csv", "text/csv")

if __name__ == "__main__":
    main()
