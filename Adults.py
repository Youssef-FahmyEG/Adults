import streamlit as st
import pandas as pd
import plotly.express as px

df = pd.read_csv("dataset_v1.csv")
st.set_page_config(layout = "wide", page_title = "📊 Adults Dashboard")
st.image("dataset_img.jpg" , use_container_width = True)
st.markdown("""
    <h1 style='text-align: center; font-size: 60px; margin-top: 20px;'>
        Adults Analysis
    </h1>
""", unsafe_allow_html = True)
st.sidebar.image("logo.png", use_container_width = True)

gender = st.sidebar.multiselect("Gender", df["sex"].unique())
race = st.sidebar.multiselect("Race", df["race"].unique())
country = st.sidebar.multiselect("Country", df["native-country"].unique())

filtered_df = df.copy()

if gender:
    filtered_df = filtered_df[filtered_df["sex"].isin(gender)]

if race:
    filtered_df = filtered_df[filtered_df["race"].isin(race)]

if country:
    filtered_df = filtered_df[filtered_df["native-country"].isin(country)]

tab1, tab2, tab3, tab4 = st.tabs([
    "Overview",
    "Visualization with Outliers",
    "Visualization without Outliers",
    "Summary & Recommendations"
])

with tab1:

    st.markdown("""
                This dataset is about adults aged 18 and over, selected as a random sample.
                It includes personal information such as **age**, **education**, **job**, and **hours worked per week** .The main goal of the dataset is to **predict whether a person earns more or less than $50K per year** based on these features.
""")

    st.markdown("<h2 style='text-align: center;'>🧾 Column Descriptions</h2>", unsafe_allow_html = True)

    column_descriptions = {
    "age": ["Age of the individual", "عمر الفرد"],
    "workclass": ["Type of employment", "نوع الوظيفة"],
    "fnlwgt": ["Final weight (used to represent the population)", "الوزن النهائي (يُستخدم لتمثيل السكان)"],
    "education": ["Level of education", "مستوى التعليم"],
    "education-num": ["Numerical representation of education level", "التمثيل الرقمي لمستوى التعليم"],
    "marital-status": ["Marital status of the individual", "الحالة الاجتماعية للفرد"],
    "occupation": ["Type of job", "نوع العمل"],
    "relationship": ["Relationship status within household", "الوضع العائلي داخل الأسرة"],
    "race": ["Race of the individual", "العرق"],
    "sex": ["Gender of the individual", "الجنس"],
    "capital-gain": ["Income from investment sources (capital gains)", "أرباح رأس المال"],
    "capital-loss": ["Capital losses (e.g., from investments)", "خسائر رأس المال"],
    "hours-per-week": ["Average hours worked per week", "متوسط عدد ساعات العمل في الأسبوع"],
    "native-country": ["Country ", "البلد"],
    "income": ["Income group (<=50K or >50K)", "تصنيف الدخل (أقل أو أكثر من 50 ألف دولار)"]
}

    desc_df = pd.DataFrame(
    [[f"**{col}**", eng, arb] for col, (eng, arb) in column_descriptions.items()],
    columns=["Column", "Description (English)", "الوصف (عربي)"])

    st.table(desc_df)

    st.error("⚠️ Note: This dataset is shown **after cleaning**.")
    st.markdown("<h3 style='text-align: center; font-weight: bold;'>📏 Shape of the Dataset</h3>", unsafe_allow_html=True)
    st.markdown(
    f"<p style='text-align: center; font-size: 18px;'>Rows: {filtered_df.shape[0]} &nbsp;&nbsp;&nbsp; Columns: {filtered_df.shape[1]}</p>",
    unsafe_allow_html = True
)

    st.markdown("<h3 style='text-align: center; font-weight: bold;'>📋 DataFrame</h3>", unsafe_allow_html=True)
    st.dataframe(filtered_df.head(5))

    st.markdown("<h3 style='text-align: center; font-weight: bold;'>📊 Statistical Summary (Numerical Columns)</h3>", unsafe_allow_html=True)
    st.dataframe(filtered_df.describe())

    st.markdown("<h3 style='text-align: center; font-weight: bold;'>📋 Categorical Summary</h3>", unsafe_allow_html=True)
    st.dataframe(filtered_df.describe(include = "O"))

    st.markdown("<h3 style='text-align: center; font-weight: bold;'>Correlation Matrix of Numerical Data</h3>", unsafe_allow_html=True)
    fig = px.imshow(df.select_dtypes(include="number").corr()
, text_auto=True)
    fig.update_layout(width = 800, height = 800)
    st.plotly_chart(fig)

with tab2:

  df_with_outlier = pd.read_csv("with_outliers_cleaned_adult_dataset.csv")

  filtered_outlier_df = df_with_outlier.copy()

  if gender:

    filtered_outlier_df = filtered_outlier_df[filtered_outlier_df["sex"].isin(gender)]

  if race:
    filtered_outlier_df = filtered_outlier_df[filtered_outlier_df["race"].isin(race)]

  if country:
    filtered_outlier_df = filtered_outlier_df[filtered_outlier_df["native-country"].isin(country)]

 # The distribution of income levels across different education levels?
  grouped = filtered_outlier_df.groupby("education")["income_classification"].value_counts().sort_values(ascending=False).reset_index(name="count")

  fig = px.bar(
        grouped,
        x="education",
        y="count",
        color="income_classification",
        barmode="group",
        title="Income Distribution by Education Level",
        labels={
            "count": "Number of Individuals",
            "education": "Education Level",
            "income_classification": "Income"
        }
    )

  fig.update_layout(
    title={
        'text': "Income Distribution by Education Level",
        'x': 0.5,
        'xanchor': 'center',
        'font': {

                 'size': 24,
            'family': 'Arial',
            'color': 'black'
        }
    },
    xaxis_tickangle=-45,
    xaxis_title="Education Level",
    yaxis_title="Count",
    legend_title="Income",
    height=800,
    width=800
)

  st.plotly_chart(fig)

  st.markdown('''
  The chart shows that most people with lower education levels like **High School** and **Some college** have low income.
  People with higher education like **Bachelors** and **Masters** are more likely to have high income.
              ''')

# What is the gender distribution among high-income earners?
  gender_distribution_high_income = filtered_outlier_df[filtered_outlier_df["income_classification"] == "High"]["sex"].value_counts()

  fig = px.pie(
    names=gender_distribution_high_income.index,
    values=gender_distribution_high_income.values)

  fig.update_traces(textinfo="percent+label")

  fig.update_layout(
    title={
        'text': "Gender Distribution (High Income)",
        'x': 0.5,
        'xanchor': 'center',
        'font': {
            'size': 24
        }
    },
    height=800,
    width=800
)


  st.plotly_chart(fig)

  st.markdown('''
  The chart shows that **males** represent the majority of high-income earners compared to **females**.
  ''')

  # Which occupations are most represented among those earning >$50K?
  top_occupations_high_income = filtered_outlier_df[filtered_outlier_df["income_classification"] == "High"]["occupation"].value_counts()

  fig = px.bar(
    x=top_occupations_high_income.values,
    y=top_occupations_high_income.index,
    orientation="h",
    title="Top Occupations (High Income)"
)

  fig.update_layout(
    title={
        'text': "Top Occupations (High Income)",
        'x': 0.5,
        'xanchor': 'center',
        'font': {
            'size': 24
        }
    },
    height=800,
    width=800,
    yaxis=dict(autorange="reversed")
)

  st.plotly_chart(fig)

  st.markdown('''
  The chart shows that the highest-paying occupations are **professional specialties** and **executive-managerial roles**, followed by **sales** and **craft-related jobs**.
              ''')

  # What is the correlation between weekly work hours and income?
  correlation = round(filtered_outlier_df["hours-per-week"].corr(filtered_outlier_df["income_numeric"]),2)

  fig = px.scatter(
    filtered_outlier_df,
    x="hours-per-week",
    y="income_numeric",
    title=f"Income vs Hours per Week (Correlation = {correlation})",
    labels={
        "hours-per-week": "Hours per Week",
        "income_numeric": "Income (0 = <=50K, 1 = >50K)"
    }
)

  fig.update_layout(
    title={
        'text': f"Income vs Hours per Week (Correlation = {correlation})",
        'x': 0.5,
        'xanchor': 'center',
        'font': {
            'size': 24
        }
    },
    height=800,
    width=800
)

  st.plotly_chart(fig)

  st.markdown('''
  The chart shows that there is a **weak positive correlation** between hours worked per week and income level''')

  # Which top 10 countries are most common among high-income individuals?

  top_countries_high_income = filtered_outlier_df[filtered_outlier_df["income_classification"] == "High"]["native-country"].value_counts().reset_index().head(10)

  fig = px.choropleth(
    top_countries_high_income,
    locations="native-country",
    locationmode="country names",
    color="count",
    color_continuous_scale="Blues",
    title="Top 10 Countries with High Income"
)

  fig.update_layout(
    height=800,
    width=800,
    title={
        'text': "Top 10 Countries with High Income",
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 24}
    }
)


  st.plotly_chart(fig)

  st.markdown('''
  The chart illustrates the distribution of **the top 10 high-income countries** on a world map, where the **blue** color gradient reflects the number of high-income individuals. **The United States** stands out as one of the leading countries.''')

  # Does marital status influence income level?

  income_by_marital_status = filtered_outlier_df.groupby("marital-status")["income_classification"].value_counts().sort_values(ascending = False).reset_index(name="count")

  fig = px.bar(
    income_by_marital_status,
    x="marital-status",
    y="count",
    color="income_classification",
    barmode="group",
    title="Income Classification by Marital Status"
)

  fig.update_layout(
    xaxis_title="Marital Status",
    yaxis_title="Count",
    height=800,
    width=800,
    title={
        'text': "Income Classification by Marital Status",
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 24}
    }
)
  st.plotly_chart(fig)

  st.markdown('''
  The chart shows that **never-married** and **married-civ-spouse** individuals represent the highest counts of **low-income** earners, while **married-civ-spouse** individuals also have a significant number of high-income earners.''')

  # How does education-num relate to income?

  avg_education_num_by_income = filtered_outlier_df.groupby("income_classification")["education-num"].mean()

  fig = px.pie(
    names=avg_education_num_by_income.index,
    values=avg_education_num_by_income.values,
    title="Average Education Level by Income",
    hole=0.5
)

  fig.update_traces(textinfo="percent+label")

  fig.update_layout(
    height=800,
    width=800,
    title={
        'text': "Average Education Level by Income",
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 24}
    }
)

  st.plotly_chart(fig)

  st.markdown('''
  The chart shows the average education level by income. **More than half** of the individuals with higher education fall into the **high-income group**, while the **rest** have **low income**.
              ''')

  # What is the average work hour per week per workclass?

  avg_hours_per_workclass = filtered_outlier_df.groupby("workclass")["hours-per-week"].mean().sort_values(ascending = False)

  fig = px.bar(
    x=avg_hours_per_workclass.values,
    y=avg_hours_per_workclass.index,
    orientation="h",
    title="Average Weekly Work Hours by Workclass"
)

  fig.update_layout(
    height=800,
    width=800,
    yaxis=dict(autorange="reversed"),
    title={
        'text': "Average Weekly Work Hours by Workclass",
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 24}
    }
)
  st.plotly_chart(fig)

  st.markdown('''
  The chart shows that **self-employed** individuals with **incorporated businesses work** the most hours per week. On the other hand, those who have **never worked** have the lowest average weekly work hours
              ''')

  # Which race groups are most likely to earn above $50K?

  race_high_income = filtered_outlier_df[filtered_outlier_df["income_classification"] == "High"]["race"].value_counts().sort_values(ascending = False).reset_index(name = "count")

  fig = px.bar(
    race_high_income,
    x="race",
    y="count",
    barmode="group",
    title="Income Classification by Race"
)

  fig.update_layout(
    xaxis_title="race",
    yaxis_title="Count",
    height=800,
    width=800,
    title={
        'text': "Income Classification by Race",
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 24}
    }
)
  st.plotly_chart(fig)

  st.markdown('''
  The chart shows that the number of income-classified individuals from the **White race** is significantly higher than all other racial groups. Other races like **Black**, **Asian-Pac-Islander**, **Amer-Indian-Eskimo**, and **Others** have much lower counts.''')

  # Which employment types (workclass) are most associated with high income?

  workclass_income_counts = filtered_outlier_df.groupby("workclass")["income_classification"].value_counts().sort_values(ascending = False).reset_index(name = "count")

  fig = px.bar(
    workclass_income_counts,
    x="workclass",
    y="count",
    color="income_classification",
    title="Income Distribution by Workclass",
    labels={
        "count": "Number of People",
        "workclass": "Workclass",
        "income_classification": "Income Classification"
    },
    barmode="group"
)

  fig.update_layout(
    height=800,
    width=800,
    xaxis_tickangle=-45,
    title={
        'text': "Income Distribution by Workclass",
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 24}
    }
)
  st.plotly_chart(fig)

  st.markdown('''
  The chart shows that showing the number of people in different workclass categories classified by income levels **(Low and High)**. The majority of people in the **"Private"** workclass have **low income**, with a notable number also in the high-income category. Other categories like **"Self-emp-not-inc"**, **"Local-gov"**, **"State-gov"**, **"Federal-gov"**, **"Self-emp-inc"**, **"Without-pay"** and **"Never-worked"** have fewer people with varying low and high income distributions.
              ''')

  # What are the top 5 most common occupations among females?
  top_female_occupations = filtered_outlier_df[filtered_outlier_df['sex'] == 'Female']['occupation'].value_counts().sort_values(ascending = False).head(5).reset_index(name = "count")

  blue_palette = ["#08306B", "#2171B5", "#4292C6", "#6BAED6", "#C6DBEF"]
  blue_shades = dict(zip(top_female_occupations["occupation"], blue_palette))

  fig = px.treemap(
    top_female_occupations,
    path=["occupation"],
    values="count",
    title="Top 5 Occupations Among Females",
    color="occupation",
    color_discrete_map=blue_shades
)

  fig.update_layout(
    height=800,
    width=800,
    title={
        'text': "Top 5 Occupations Among Females",
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 24}
    }
)
  st.plotly_chart(fig)

  st.markdown('''
  The chart shows that showing the distribution of the most common occupations. The largest areas are **"Adm-clerical"** and **"Prof-specialty"** followed by **"Other-service"**, **"Sales"**, and **"Exec-managerial"** indicating their relative prevalence. The **size** of each rectangle represents the proportion of females in each occupation.
              ''')

with tab3:

  df_without_outlier = pd.read_csv("cleaned_without_outlierv2.csv")

  filtered_without_outlier_df = df_without_outlier.copy()

  if gender:

    filtered_without_outlier_df = filtered_without_outlier_df[filtered_without_outlier_df["sex"].isin(gender)]

  if race:
    filtered_without_outlier_df = filtered_without_outlier_df[filtered_without_outlier_df["race"].isin(race)]

  if country:
    filtered_without_outlier_df = filtered_without_outlier_df[filtered_without_outlier_df["native-country"].isin(country)]

 # The distribution of income levels across different education levels?
  grouped = filtered_without_outlier_df.groupby("education")["income_classification"].value_counts().sort_values(ascending=False).reset_index(name="count")

  green_shades = {
    "Low": "#a1d99b",
    "High": "#006d2c"
}

  fig = px.bar(
    grouped,
    x="education",
    y="count",
    color="income_classification",
    barmode="group",
    title="Income Distribution by Education Level",
    labels={
        "count": "Number of Individuals",
        "education": "Education Level",
        "income_classification": "Income"
    },
    color_discrete_map=green_shades
)

  fig.update_layout(
    title={
        'text': "Income Distribution by Education Level",
        'x': 0.5,
        'xanchor': 'center',
        'font': {
            'size': 24,
            'family': 'Arial',
            'color': 'black'
        }
    },
    xaxis_tickangle=-45,
    xaxis_title="Education Level",
    yaxis_title="Count",
    legend_title="Income",
    height=800,
    width=800
)

  st.plotly_chart(fig, key="chart_1")

  #----------------------------------------

  gender_distribution_high_income = filtered_without_outlier_df[filtered_without_outlier_df["income_classification"] == "High"]["sex"].value_counts()

  gender_colors = {
    "Male": "#006d2c",
    "Female": "#a1d99b"
}
  fig = px.pie(
    names=gender_distribution_high_income.index,
    values=gender_distribution_high_income.values,
    color=gender_distribution_high_income.index,
    color_discrete_map=gender_colors

)

  fig.update_traces(textinfo="percent+label")
  fig.update_layout(
    title={
        'text': "Gender Distribution (High Income)",
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 24}
    },
    height=800,
    width=800
)
  st.plotly_chart(fig, key="chart_2")

  #----------------------------------------

  top_occupations_high_income = filtered_without_outlier_df[filtered_without_outlier_df["income_classification"] == "High"]["occupation"].value_counts()

  fig = px.bar(
    x=top_occupations_high_income.values,
    y=top_occupations_high_income.index,
    orientation="h",
    title="Top Occupations (High Income)")
  fig.update_layout(
    title={
        'text': "Top Occupations (High Income)",
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 24}
    },
    height=800,
    width=800,
    yaxis=dict(autorange="reversed")
)
  st.plotly_chart(fig, key="chart_3")

    #----------------------------------------

  correlation = round(filtered_without_outlier_df["hours-per-week"].corr(filtered_without_outlier_df["income_numeric"]), 2)
  fig = px.scatter(
    filtered_without_outlier_df,
    x="hours-per-week",
    y="income_numeric",
    title=f"Income vs Hours per Week (Correlation = {correlation})",
    color="income_classification",
    color_discrete_map=green_shades,
    labels={
        "hours-per-week": "Hours per Week",
        "income_numeric": "Income (0 = <=50K, 1 = >50K)"
    }
)
  fig.update_layout(
    title={
        'text': f"Income vs Hours per Week (Correlation = {correlation})",
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 24}
    },
    height=800,
    width=800
)
  st.plotly_chart(fig, key="chart_4")

  #----------------------------------------

  top_countries_high_income = filtered_without_outlier_df[filtered_without_outlier_df["income_classification"] == "High"]["native-country"].value_counts().reset_index()
  top_countries_high_income.columns = ["native-country", "count"]
  top_countries_high_income = top_countries_high_income.head(10)

  fig = px.choropleth(
    top_countries_high_income,
    locations="native-country",
    locationmode="country names",
    color="count",
    color_continuous_scale="Greens",
    title="Top 10 Countries with High Income"
)
  fig.update_layout(
    height=800,
    width=800,
    title={
        'text': "Top 10 Countries with High Income",
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 24}
    }
)
  st.plotly_chart(fig, key="chart_5")

  #----------------------------------------

  income_by_marital_status = filtered_without_outlier_df.groupby("marital-status")["income_classification"].value_counts().sort_values(ascending=False).reset_index(name="count")

  fig = px.bar(
    income_by_marital_status,
    x="marital-status",
    y="count",
    color="income_classification",
    barmode="group",
    title="Income Classification by Marital Status",
    color_discrete_map=green_shades
)
  fig.update_layout(
    xaxis_title="Marital Status",
    yaxis_title="Count",
    height=800,
    width=800,
    title={
        'text': "Income Classification by Marital Status",
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 24}
    }
)
  st.plotly_chart(fig, key="chart_6")

    #----------------------------------------

  green_shades1 = {
    "Low": "#a1d99b",
    "High": "#006d2c"
}
  avg_education_num_by_income = filtered_without_outlier_df.groupby("income_classification")["education-num"].mean()

  fig = px.pie(
    names=avg_education_num_by_income.index,
    values=avg_education_num_by_income.values,
    title="Average Education Level by Income",
    hole=0.5,
    color_discrete_map=green_shades1

)
  fig.update_traces(textinfo="percent+label")
  fig.update_layout(
    height=800,
    width=800,
    title={
        'text': "Average Education Level by Income",
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 24}
    }
)
  st.plotly_chart(fig, key="chart_7")

  #----------------------------------------

  avg_hours_per_workclass = filtered_without_outlier_df.groupby("workclass")["hours-per-week"].mean().sort_values(ascending=False)
  fig = px.bar(
    x=avg_hours_per_workclass.values,
    y=avg_hours_per_workclass.index,
    orientation="h",
    title="Average Weekly Work Hours by Workclass",
    color_discrete_map=green_shades1
)
  fig.update_layout(
    height=800,
    width=800,
    yaxis=dict(autorange="reversed"),
    title={
        'text': "Average Weekly Work Hours by Workclass",
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 24}
    }
)
  st.plotly_chart(fig, key="chart_8")

   #----------------------------------------

  race_colors = {
    "White": "#8dd3c7",
    "Black": "#ffffb3",
    "Asian-Pac-Islander": "#bebada",
    "Amer-Indian-Eskimo": "#fb8072",
    "Other": "#80b1d3"}

  race_high_income = filtered_without_outlier_df[filtered_without_outlier_df["income_classification"] == "High"]["race"].value_counts().sort_values(ascending=False).reset_index(name="count")

  fig = px.bar(
    race_high_income,
    x="race",
    y="count",
    barmode="group",
    title="Income Classification by Race"
)

  fig.update_layout(
    xaxis_title="race",
    yaxis_title="Count",
    height=800,
    width=800,
    title={
        'text': "Income Classification by Race",
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 24}
    }
)

  st.plotly_chart(fig, key="chart_9")

  #----------------------------------------

  workclass_income_counts = filtered_without_outlier_df.groupby("workclass")["income_classification"].value_counts().sort_values(ascending=False).reset_index(name="count")

  fig = px.bar(
    workclass_income_counts,
    x="workclass",
    y="count",
    color="income_classification",
    title="Income Distribution by Workclass",
    labels={
        "count": "Number of People",
        "workclass": "Workclass",
        "income_classification": "Income Classification"
    },
    barmode="group",
    color_discrete_map=green_shades
)
  fig.update_layout(
    height=800,
    width=800,
    xaxis_tickangle=-45,
    title={
        'text': "Income Distribution by Workclass",
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 24}
    }
)
  st.plotly_chart(fig, key="chart_10")

   #----------------------------------------

  green_palette = ['#3ca96b', '#5bc27c', '#81d68d', '#a8e6a1', '#d0f0c0']
  green_shades2 = dict(zip(top_female_occupations["occupation"], green_palette))

  top_female_occupations = filtered_without_outlier_df[filtered_without_outlier_df['sex'] == 'Female']['occupation'].value_counts().sort_values(ascending=False).head(5).reset_index(name="count")

  fig = px.treemap(
    top_female_occupations,
    path=["occupation"],
    values="count",
    color="occupation",
    color_discrete_map=green_shades2,
    title="Top 5 Occupations Among Females"
)
  fig.update_layout(
    height=800,
    width=800,
    title={
        'text': "Top 5 Occupations Among Females",
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 24}
    }
)
  st.plotly_chart(fig, key="chart_11")


with tab4 :

  st.markdown('''
<div style="text-align: center;">
  <h1><b>Summary</b></h1>
</div>

The dashboard analyzes adult dataset to predict income levels **(>50K or <=50K)** based on features like **age**, **education** , **occupation** , and **hours worked per week** . The dashboard includes interactive filters ( **gender** , **race** , **country** ) and visualizations across four tabs: **Overview** , **Visualization with Outliers** , **Visualization without Outliers** , and **Summary & Recommendations** . Key insights include:

<b>Education and Income:</b> Higher education levels ( **Bachelors** , **Masters** ) correlate with higher income, while lower education levels ( **High School** ) are associated with lower income.

<b>Gender Disparity:</b> Males dominate high-income groups compared to females.

<b>Occupations:</b> **Professional specialties** and **executive-managerial roles** are the most common among high-income earners.

<b>Work Hours:</b> A weak positive correlation exists between weekly work hours and income.

<b>Country Distribution:</b> **The United States** has the highest number of high-income individuals.

<b>Marital Status:</b> **Married** individuals (civil spouse) have a significant presence in both high- and low-income groups.

<b>Race:</b> **White** individuals dominate high-income groups with other races having lower representation.

<b>Workclass:</b> **Self-employed** individuals with incorporated businesses work the most hours, while **private sector** workers dominate both income categories.

<b>Female Occupations:</b> **Administrative** , **clerical** , and **professional specialties** are the most common occupations among **females** .

<b>Outliers:</b> After analyzing the data, the difference between data **with outlier** and **without outlier** indicate that outliers have little significant impact on the overall trends and distributions.
)''', unsafe_allow_html=True)

  st.markdown("""
<h2 style='text-align: center; font-size: 32px;'>Recommendations</h2>

- **Focus on Education:** Promote higher education programs, as advanced degrees strongly correlate with higher income.
- **Address Gender Inequality:** Develop policies to reduce income disparities, particularly supporting women in high-paying roles.
- **Target High-Income Occupations:** Encourage training in professional specialties and managerial roles to boost income potential.
- **Work-Life Balance:** Investigate the weak correlation between work hours and income to optimize productivity without excessive hours.
- **Diversity in Employment:** Increase representation of underrepresented races in high-income roles through targeted career development programs.
- **Support for Self-Employed:** Provide resources for self-employed individuals especially those in incorporated businesses to sustain high work hours and income.
""", unsafe_allow_html=True)

