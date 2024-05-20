"""
In an environment with streamlit, plotly and duckdb installed,
Run with `streamlit run streamlit_app.py`
"""
import random
import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_groq import ChatGroq
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_google_genai import ChatGoogleGenerativeAI
# bring in our GROQ_API_KEY
from dotenv import load_dotenv
load_dotenv()

#######################################
# PAGE SETUP
#######################################

st.set_page_config(page_title="Sales Dashboard", page_icon=":bar_chart:", layout="wide")

st.title("Sales Streamlit Dashboard")
# st.markdown("_Prototype v0.4.1_")

with st.sidebar:
    st.header("Configuration")
    uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is None:
    st.info(" Upload a file through config", icon="ℹ️")
    st.stop()

#######################################
# DATA LOADING
#######################################


@st.cache_data
def load_data(path: str):
    df = pd.read_excel(path)
    return df


df = load_data(uploaded_file)
all_months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

xls_file = r'excel_file_example.xlsx'
output_csv = r'excel_file_example.csv'

# Read the XLS file using pandas and openpyxl as the engine
data = pd.read_excel(uploaded_file, engine='openpyxl')

# Save the data as a CSV file
data.to_csv(output_csv, index=False)
if uploaded_file is not None:
    # df = pd.read_csv(output_csv)
    # st.write("Here are the first five rows of your file:")
    # st.write(df.head())
    # chat_model = ChatGroq(temperature=0, model_name="Llama3-8b-8192")
    # llm = ChatOpenAI(model="gpt-3.5-turbo")
    # agent_prompt_prefix = 'Your name is Jarvis and you are working with a pandas dataframe called "df".'

    # agent = create_pandas_dataframe_agent(
    #     llm, df, prefix=agent_prompt_prefix, verbose=True, agent_type=AgentType.OPENAI_FUNCTIONS
    # )
    
    # user_query = st.text_input("Ask a question about your data:")
    # if user_query:
    #     response = agent.invoke(user_query)
    #     st.write("Response:", response)

    llm = ChatGoogleGenerativeAI(model="gemini-pro")

    agent = create_csv_agent(llm, output_csv, verbose=True)
    user_query = st.text_input("Ask a question about your data:")
    if user_query:
        response = agent.run(user_query)
        st.write("Response:", response)

else:
    st.write("Please upload a CSV file to continue.")

with st.expander("Data Preview"):
    df = pd.read_csv(output_csv)
    st.write(df)
    # st.dataframe(
    #     df#,
    #     # column_config={"Year": st.column_config.NumberColumn(format="%d")},
    # )

#######################################
# VISUALIZATION METHODS
#######################################


def plot_metric(label, value, prefix="", suffix="", show_graph=False, color_graph=""):
    fig = go.Figure()

    fig.add_trace(
        go.Indicator(
            value=value,
            gauge={"axis": {"visible": False}},
            number={
                "prefix": prefix,
                "suffix": suffix,
                "font.size": 28,
            },
            title={
                "text": label,
                "font": {"size": 24},
            },
        )
    )

    if show_graph:
        fig.add_trace(
            go.Scatter(
                y=random.sample(range(0, 101), 30),
                hoverinfo="skip",
                fill="tozeroy",
                fillcolor=color_graph,
                line={
                    "color": color_graph,
                },
            )
        )

    fig.update_xaxes(visible=False, fixedrange=True)
    fig.update_yaxes(visible=False, fixedrange=True)
    fig.update_layout(
        # paper_bgcolor="lightgrey",
        margin=dict(t=30, b=0),
        showlegend=False,
        plot_bgcolor="white",
        height=100,
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_gauge(
    indicator_number, indicator_color, indicator_suffix, indicator_title, max_bound
):
    fig = go.Figure(
        go.Indicator(
            value=indicator_number,
            mode="gauge+number",
            domain={"x": [0, 1], "y": [0, 1]},
            number={
                "suffix": indicator_suffix,
                "font.size": 26,
            },
            gauge={
                "axis": {"range": [0, max_bound], "tickwidth": 1},
                "bar": {"color": indicator_color},
            },
            title={
                "text": indicator_title,
                "font": {"size": 28},
            },
        )
    )
    fig.update_layout(
        # paper_bgcolor="lightgrey",
        height=200,
        margin=dict(l=10, r=10, t=50, b=10, pad=8),
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_top_right():
    try:
        comparison_data = duckdb.sql(
            """
            WITH role_counts AS (
                SELECT
                    'Quản lý nhà nước',
                    SUM(CASE WHEN 'Quản lý nhà nước' = 'CV' THEN 1 ELSE 0 END) AS so_luong_CV,
                    SUM(CASE WHEN 'Quản lý nhà nước' = 'CVC' THEN 1 ELSE 0 END) AS so_luong_CVC
                FROM df
            )
            
            SELECT * FROM role_counts
            """
        ).df()

        # Kiểm tra nếu dữ liệu không có giá trị
        if comparison_data.empty:
            st.warning("No data available for the specified query.")
            return

        # Chuyển đổi dữ liệu để phù hợp với biểu đồ cột
        comparison_data = comparison_data.melt(id_vars=["Quản lý nhà nước"], var_name="role", value_name="count")

        fig = px.bar(
            comparison_data,
            x="role",
            y="count",
            color="role",
            barmode="group",
            text_auto=".2s",
            title="Comparison of CV and CVC in Quản lý nhà nước",
            height=400,
        )
        fig.update_traces(
            textfont_size=12, textangle=0, textposition="outside", cliponaxis=False
        )
        st.plotly_chart(fig, use_container_width=True)
    except duckdb.BinderException as e:
        st.error(f"SQL Error: {e}")
    except Exception as e:
        st.error(f"Unexpected Error: {e}")


def plot_bottom_left():
    sales_data = duckdb.sql(
        f"""
        WITH sales_data AS (
            SELECT 
            Scenario,{','.join(all_months)} 
            FROM df 
            WHERE Year='2023' 
            AND Account='Sales'
            AND business_unit='Software'
        )

        UNPIVOT sales_data 
        ON {','.join(all_months)}
        INTO
            NAME month
            VALUE sales
    """
    ).df()

    fig = px.line(
        sales_data,
        x="month",
        y="sales",
        color="Scenario",
        markers=True,
        text="sales",
        title="Monthly Budget vs Forecast 2023",
    )
    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, use_container_width=True)


def plot_bottom_right():
    sales_data = duckdb.sql(
        f"""
        WITH sales_data AS (
            UNPIVOT ( 
                SELECT 
                    Account,Year,{','.join([f'ABS({month}) AS {month}' for month in all_months])}
                    FROM df 
                    WHERE Scenario='Actuals'
                    AND Account!='Sales'
                ) 
            ON {','.join(all_months)}
            INTO
                NAME year
                VALUE sales
        ),

        aggregated_sales AS (
            SELECT
                Account,
                Year,
                SUM(sales) AS sales
            FROM sales_data
            GROUP BY Account, Year
        )
        
        SELECT * FROM aggregated_sales
    """
    ).df()

    fig = px.bar(
        sales_data,
        x="Year",
        y="sales",
        color="Account",
        title="Actual Yearly Sales Per Account",
    )
    st.plotly_chart(fig, use_container_width=True)


#######################################
# STREAMLIT LAYOUT
#######################################

top_left_column, top_right_column = st.columns((2, 1))
bottom_left_column, bottom_right_column = st.columns(2)

with top_left_column:
    column_1, column_2, column_3, column_4 = st.columns(4)

    with column_1:
        plot_metric(
            "Tổng số công chức",
            26,
            prefix="",
            suffix=" người",
            show_graph=False,
            color_graph="rgba(0, 104, 201, 0.2)",
        )
        plot_gauge(2, "#0068C9", "", "Thạc sĩ", 26)

    with column_2:
        plot_metric(
            "Số đảng viên",
            27,
            prefix="",
            suffix=" người",
            show_graph=False,
            color_graph="rgba(255, 43, 43, 0.2)",
        )
        plot_gauge(24, "#FF8700", " ", "Cử nhân", 26)

    with column_3:
        # Execute the SQL query
        ldql = duckdb.sql(
            f"""
            SELECT
                COUNT(*) AS So_nguoi_co_chuc_vu
            FROM
                df
            WHERE
                'Chức vụ' IS NOT NULL;
            """
        ).df()

        # Extract the number of people with non-null chức vụ
        so_nguoi_co_chuc_vu = ldql["So_nguoi_co_chuc_vu"].iloc[0]
        plot_metric("Lãnh đạo", so_nguoi_co_chuc_vu, prefix="", suffix=" người", show_graph=False)
        plot_gauge(7, "#FF2B2B", " ", "Cao cấp", 26)
        
    with column_4:
        plot_metric("Chuyên viên chính", 4, prefix="", suffix=" người", show_graph=False)
        plot_gauge(10, "#29B09D", " ", "Trung cấp", 4)

with top_right_column:
    plot_top_right()

with bottom_left_column:
    plot_bottom_left()

with bottom_right_column:
    plot_bottom_right()
