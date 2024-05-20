import pandas as pd
import plotly.express as px
import streamlit as st
import asyncio
from dotenv import load_dotenv
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

#######################################
# PAGE SETUP
#######################################

st.set_page_config(page_title="Quản lý nhân sự", page_icon=":bar_chart:", layout="wide")
st.title("Thống kê nhân sự")

#######################################
# DATA LOADING
#######################################
input_xlsx = r'excel_file_example.xlsx'
output_csv = r'excel_file_example.csv'

@st.cache_data
def load_data(path: str):
    df = pd.read_excel(path, engine='openpyxl')
    df.to_csv(output_csv, index=False)
    return df

df = load_data(input_xlsx)

# Hàm tạo agent và chạy truy vấn của người dùng
async def run_agent(query):
    llm = ChatGoogleGenerativeAI(model="gemini-pro")
    agent = create_csv_agent(llm, output_csv, verbose=True)
    response = await agent.run(query)
    return response

user_query = st.text_input("Hỏi một câu về dữ liệu của bạn:")
if user_query:
    try:
        response = asyncio.run(run_agent(user_query))
        st.write("Phản hồi:", response)
    except Exception as e:
        st.error(f"Lỗi: {e}")

with st.expander("Xem trước dữ liệu"):
    st.dataframe(df, hide_index=True)

#######################################
# CÁC PHƯƠNG PHÁP HIỂN THỊ
#######################################

def plot_bottom_right():
    try:
        total_by_rank = df.groupby('Ngạch').size()

        fig = px.bar(
            x=total_by_rank.index,
            y=total_by_rank.values,
            title='Tổng số nhân viên theo Ngạch',
            labels={'x': 'Ngạch', 'y': 'Số lượng nhân viên'}
        )

        fig.update_layout(
            autosize=True,
            margin=dict(l=0, r=0, t=30, b=0)
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Đã xảy ra lỗi: {e}")

def plot_top_right():
    try:
        party_member_count = df['Đảng viên'].notnull().sum()
        non_party_member_count = len(df) - party_member_count
        party_member_counts = [party_member_count, non_party_member_count]
        party_labels = ['Đảng viên', 'Không phải Đảng viên']

        fig = px.pie(values=party_member_counts, names=party_labels,
                     title='Chia theo Đảng viên',
                     hole=0.4,
                     labels={'values': 'Số lượng', 'names': 'Tình trạng Đảng viên'})

        fig.update_traces(textposition='inside', textinfo='label+percent')
        fig.update_layout(
            autosize=True,
            margin=dict(l=0, r=0, t=30, b=0)
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Lỗi khi đọc file: {e}")

def plot_employee_count_by_department():
    data = pd.read_excel('excel_file_example.xlsx')
    employee_count_by_level = data
