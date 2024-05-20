import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_google_genai import ChatGoogleGenerativeAI

def main():
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

    llm = ChatGoogleGenerativeAI(model="gemini-pro")
    agent = create_csv_agent(llm, output_csv, verbose=True)

    user_query = st.text_input("Ask a question about your data:")
    if user_query:
        response = agent.run(user_query)
        st.write("Response:", response)

    with st.expander("Data Preview"):
        st.dataframe(df, hide_index=True)

    #######################################
    # VISUALIZATION METHODS
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
            st.error(f"Error reading the file: {e}")

    def plot_employee_count_by_department():
        data = pd.read_excel('excel_file_example.xlsx')
        employee_count_by_level = data['Trình độ'].value_counts()

        fig = px.bar(
            x=employee_count_by_level.values,
            y=employee_count_by_level.index,
            title='Tổng số nhân viên theo Trình độ',
            orientation='h',
            labels={'x': 'Số lượng người', 'y': 'Trình độ chuyên môn'}
        )

        fig.update_layout(
            autosize=True,
            margin=dict(l=0, r=0, t=30, b=0)
        )

        st.plotly_chart(fig, use_container_width=True)

    def plot_average_salary_by_department():
        data = pd.read_excel('excel_file_example.xlsx')
        employee_count_by_state = data['Quản lý nhà nước'].value_counts()

        fig = px.bar(
            x=employee_count_by_state.index,
            y=employee_count_by_state.values,
            title='Tổng số nhân viên theo Quản lý nhà nước',
            labels={'x': 'Trình độ QLNN', 'y': 'Số lượng người'}
        )

        fig.update_layout(
            autosize=True,
            margin=dict(l=0, r=0, t=30, b=0)
        )

        st.plotly_chart(fig, use_container_width=True)

    def plot_position_distribution():
        try:
            with_position_count = df['Chức vụ'].notnull().sum()
            without_position_count = len(df) - with_position_count
            position_counts = [with_position_count, without_position_count]
            position_labels = ['Có chức vụ', 'Không có chức vụ']

            fig = px.pie(values=position_counts, names=position_labels,
                         title='Phân phối chức vụ',
                         hole=0.4,
                         labels={'values': 'Số lượng', 'names': 'Tình trạng chức vụ'})

            fig.update_traces(textposition='inside', textinfo='label+percent')
            fig.update_layout(
                autosize=True,
                margin=dict(l=0, r=0, t=30, b=0)
            )

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error reading the file: {e}")

    #######################################
    # STREAMLIT LAYOUT
    #######################################

    top_left_column, top_right_column = st.columns((2, 1))
    bottom_left_column, bottom_right_column = st.columns(2)

    with top_left_column:
        column_1, column_2 = st.columns(2)

        with column_1:
            plot_employee_count_by_department()

        with column_2:
            plot_average_salary_by_department()

    with top_right_column:
        plot_top_right()

    with bottom_left_column:
        plot_position_distribution()

    with bottom_right_column:
        plot_bottom_right()

if __name__ == "__main__":
    main()
