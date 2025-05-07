import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import datetime

# 구글 시트에서 데이터 불러오기
@st.cache_data
def load_data():
    sheet_url = "https://docs.google.com/spreadsheets/d/1jWBjgEEZi2zyROyMKJ7hBMUwtC43fJ4O0-f9eh_Wo9M/gviz/tq?tqx=out:csv&sheet=Sheet1"
    df = pd.read_csv(sheet_url)
    df['날짜'] = pd.to_datetime(df['날짜'])
    return df

df = load_data()

st.title("📈 매출·신환·내원 예측 대시보드")

# 입력값과 타겟값 정의
features = ['디비', '플레이스순위', '울산치과검색량']
targets = {
    '매출': '매출',
    '전체신환': '전체신환',
    '내원': '내원'
}

# 사용자 선택
target_option = st.selectbox("예측 항목 선택", list(targets.keys()))
target_col = targets[target_option]

# 학습 데이터 분리
X = df[features]
y = df[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 학습
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# 예측
df['예측값'] = model.predict(X)
error = abs(df['예측값'] - y)
df['최소예측'] = df['예측값'] - error.std()
df['최대예측'] = df['예측값'] + error.std()

# 결과 테이블 출력
st.subheader(f"📊 {target_option} 예측 결과 (최근 7일)")
st.dataframe(df[['날짜', target_col, '예측값', '최소예측', '최대예측']].sort_values('날짜', ascending=False).head(7))

# 사용자 입력 예측기
st.subheader("🔍 직접 입력해서 예측해보기")

col1, col2, col3 = st.columns(3)
input_db = col1.number_input("디비", value=30)
input_rank = col2.number_input("플레이스 순위", value=3)
input_search = col3.number_input("울산치과 검색량", value=500)

if st.button("예측 실행"):
    input_data = pd.DataFrame([[input_db, input_rank, input_search]], columns=features)
    pred = model.predict(input_data)[0]
    st.success(f"✅ 예측된 {target_option}: {int(pred):,} (± 약 {int(error.std()):,})")

# 그래프
st.subheader(f"📈 {target_option} 실제값 vs 예측값 (전체)")
st.line_chart(df.set_index('날짜')[[target_col, '예측값']])
