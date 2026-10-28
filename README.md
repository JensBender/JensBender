![Profile-banner](images/profile-banner.gif)

# Hi there! I'm Jens

## 👋 About Me
PhD Researcher with 8+ years in advanced statistical modeling, now applying these skills to solve complex Data Science problems. Expertise in building robust, automated end-to-end ML solutions that translate raw data into actionable insights via deployed web apps and APIs.


## 🛠️ Skills

| Category                 | Skill    |
| ------------------------ | -------- |
| Programming              | [![Python][Python-badge]][Python-url] [![MySQL][MySQL-badge]][MySQL-url] |
| Data Manipulation        | [![NumPy][NumPy-badge]][NumPy-url] [![Pandas][Pandas-badge]][Pandas-url] |
| Data Visualization       | [![Matplotlib][Matplotlib-badge]][Matplotlib-url] [![Seaborn][Seaborn-badge]][Seaborn-url] [![Plotly][Plotly-badge]][Plotly-url] [![Power BI][PowerBI-badge]][PowerBI-url] |
| AI & Machine Learning    | [![scikit-learn][scikit-learn-badge]][scikit-learn-url] [![XGBoost][XGBoost-badge]][XGBoost-url] [![SHAP][SHAP-badge]][SHAP-url] [![TensorFlow][TensorFlow-badge]][TensorFlow-url] |
| Big Data                 | [![Spark][Spark-badge]][Spark-url] |
| Web Development          | [![FastAPI][FastAPI-badge]][FastAPI-url] [![Flask][Flask-badge]][Flask-url] [![Gradio][Gradio-badge]][Gradio-url] [![Pydantic][Pydantic-badge]][Pydantic-url] |
| Version Control          | [![Git][Git-badge]][Git-url] [![GitHub][GitHub-badge]][GitHub-url] |
| MLOps                    | [![MLflow][MLflow-badge]][MLflow-url] [![DVC][DVC-badge]][DVC-url] [![Hugging Face][HuggingFace-badge]][HuggingFace-url] |
| DevOps                   | [![Docker][Docker-badge]][Docker-url] [![Airflow][Airflow-badge]][Airflow-url] [![GitHub Actions][GitHubActions-badge]][GitHubActions-url] |
| Testing                  | [![pytest][Pytest-badge]][Pytest-url] [![Selenium][Selenium-badge]][Selenium-url] |
| Cloud                    | [![AWS][AWS-badge]][AWS-url] |


## 💻 Portfolio

### [Project 1: Medical Cost Planner](https://github.com/JensBender/medical-cost-prediction)
[![Python][Python-badge]][Python-url] [![scikit-learn][scikit-learn-badge]][scikit-learn-url] [![XGBoost][XGBoost-badge]][XGBoost-url] [![SHAP][SHAP-badge]][SHAP-url] [![DVC][DVC-badge]][DVC-url] [![MLflow][MLflow-badge]][MLflow-url]

**Active Development:** Model training and evaluation are complete. The web app and API are next.

Machine learning project to help U.S. adults plan for annual out-of-pocket healthcare costs using accessible demographic and health information. Trained on MEPS 2023 survey data, **XGBoost quantile regression** was selected as the final MVP model. It provides a **plan-around estimate, typical range, and safety cushion** to help users budget for uncertain costs.

On the held-out test set, the model passes all predefined performance thresholds for launch, with a survey-weighted **median absolute error (MdAE) of $240** for the plan-around estimate. Compared with population-wide or age-group estimates, it provides better predictions, especially for typical ranges and safety cushions.

**SHAP explanations** show which answers contribute most to the plan-around estimate. Cost comparison benchmarks help users compare their estimate with typical spending for U.S. adults and their age group, while medical inflation adjustment expresses amounts in current dollars.

**Example Prediction Output**  
High cost profile: 69 years, arthritis
>
> **Your Estimated Out-of-Pocket Costs for Next Year**
>
> - 💰 **Plan around:** $895
> - 📊 **Typical range:** $400–$2,025
> - 🛡️ **Safety cushion:** $3,852
>
> <details>
> <summary><strong>Which answers shaped your estimate?</strong> <i>(click to expand)</i></summary>
> <p>These answers made the largest contributions to your plan-around estimate:</p>
> <table>
> <tbody>
> <tr><td><strong>Arthritis:</strong> Yes</td><td align="right">↑ +$124</td></tr>
> <tr><td><strong>Joint pain:</strong> Yes</td><td align="right">↑ +$110</td></tr>
> <tr><td><strong>High blood pressure:</strong> Yes</td><td align="right">↑ +$94</td></tr>
> <tr><td><strong>Age:</strong> 69</td><td align="right">↑ +$92</td></tr>
> <tr><td><strong>Insurance:</strong> Any private</td><td align="right">↑ +$80</td></tr>
> </tbody>
> </table>
> </details>
>
> <details>
> <summary><strong>How you compare to others</strong> <i>(click to expand)</i></summary>
> <table>
> <tbody>
> <tr><td>Your plan-around estimate</td><td align="right">$895</td></tr>
> <tr><td>Typical American</td><td align="right">$268</td></tr>
> <tr><td>Typical for ages 65+</td><td align="right">$657</td></tr>
> </tbody>
> </table>
> </details>

<details>
<summary>ℹ️ <strong>Preprocessing Pipeline</strong> (click to expand)</summary>
<img src="images/medical_cost_pipeline.svg" alt="Preprocessing Pipeline">
</details>


### [Project 2: Loan Default Prediction](https://github.com/JensBender/loan-default-prediction) 
[![Python][Python-badge]][Python-url] [![Pandas][Pandas-badge]][Pandas-url] [![scikit-learn][scikit-learn-badge]][scikit-learn-url] [![FastAPI][FastAPI-badge]][FastAPI-url] [![pytest][Pytest-badge]][Pytest-url] [![Docker][Docker-badge]][Docker-url] [![Hugging Face][HuggingFace-badge]][HuggingFace-url]    
Built an end-to-end machine learning solution for predicting loan defaults using customer application data, enabling financial institutions to make data-driven lending decisions and better manage credit risk. The project includes:
- **Data Preprocessing**: Engineered new features (e.g., job stability, city tier, state default rate), handled duplicates, data types, missing values, and outliers, scaled numerical features, and encoded categorical features.
- **Exploratory Data Analysis**: Analyzed distributions and relationships using descriptive statistics, correlations, and visualizations.
- **Modeling**: Trained and evaluated eight baseline models (e.g., Logistic Regression, Random Forest, XGBoost) and tuned hyperparameters. Selected a Random Forest Classifier with an optimized decision threshold, achieving an AUC-PR of 0.59, recall of 0.79, and precision of 0.51 for the default class on the hold-out test set. Visualized feature importances and showed model prediction examples.
- **Deployment**: Served the full machine learning pipeline (preprocessing and model) as a web app using a FastAPI backend and Gradio frontend within a single Docker container, hosted on Hugging Face Spaces. Automated deployment via GitHub Actions to sync web app files to Hugging Face Spaces on every push.
- **Testing**: Implemented comprehensive unit, integration, and end-to-end tests to validate individual components, their interactions, and entire user journeys.

**Model Pipeline:** [Hugging Face Hub](https://huggingface.co/JensBender/loan-default-prediction-pipeline)  
**Web App:** [Hugging Face Spaces](https://huggingface.co/spaces/JensBender/loan-default-prediction-app)

<img src="images/loan_default_web_app.png" alt="Web App">
<details>
<summary>ℹ️ <strong>Model Performance</strong> (click to expand)</summary>
<img src="images/precision_recall_curves_tuned.png" alt="Precision-Recall Curves">
</details>

### [Project 3: YouTube ETL & Analytics](https://github.com/JensBender/youtube-channel-analytics)
[![Python][Python-badge]][Python-url] [![MySQL][MySQL-badge]][MySQL-url] [![Airflow][Airflow-badge]][Airflow-url] [![Docker][Docker-badge]][Docker-url] [![AWS][AWS-badge]][AWS-url] [![Hugging Face][HuggingFace-badge]][HuggingFace-url] [![Power BI][PowerBI-badge]][PowerBI-url]  
To empower YouTube content creators and marketers with actionable insights into their channel's performance, especially in comparison to related channels, I developed a comprehensive **ETL pipeline** and designed an interactive **Power BI report**. This project involved:

- **Data Extraction**: Utilized the YouTube API to gather extensive data from three selected channels, including videos and comments.
- **Data Transformation**: Performed sentiment analysis on video comments via API requests to a RoBERTa sentiment analysis model, which I deployed using Gradio on a private Hugging Face Space.
- **Data Loading**: Stored the transformed data in a MySQL database hosted on AWS.
- **Automation**: Managed the ETL workflow using Apache Airflow, Docker, and AWS.
- **Data Visualization**: Designed an interactive Power BI report to deliver insights into channel performance, featuring key metrics and comparative analysis. 

This project enables YouTube content creators to easily monitor and evaluate their channel's performance relative to their peers, allowing for more informed decision-making and strategic planning.

<img src="images/etl_pipeline.svg" alt="YouTube ETL Pipeline">
<details>
<summary>ℹ️ <strong>Power BI Dashboard</strong> (click to expand)</summary>
<img src="images/powerbi_comments.PNG" alt="PowerBI Comments">
</details>

### [Project 4: Hate Speech Detection](https://github.com/JensBender/hate-speech-detection)
[![Python][Python-badge]][Python-url] [![TensorFlow][TensorFlow-badge]][TensorFlow-url] [![scikit-learn][scikit-learn-badge]][scikit-learn-url] [![NumPy][NumPy-badge]][NumPy-url] [![Pandas][Pandas-badge]][Pandas-url] [![Matplotlib][Matplotlib-badge]][Matplotlib-url] [![Flask][Flask-badge]][Flask-url]  
- **Motivation**: Develop a hate speech detector for social media comments. 
- **Data**: Utilized the [ETHOS Hate Speech Detection Dataset](https://github.com/intelligence-csd-auth-gr/Ethos-Hate-Speech-Dataset).
- **Models**: Trained and evaluated the performance of three deep learning models using TensorFlow and scikit-learn. The fine-tuned BERT model demonstrated superior performance (78.0% accuracy) compared to the SimpleRNN (66.3%) and LSTM (70.7%) models.  
- **Deployment**: Prepared the fine-tuned BERT model for production by integrating it into a web application and an API endpoint using the Flask web framework.

| Fine-tuned BERT: Confusion Matrix | Model Deployment | 
| ------------------ | ------------------ | 
| ![BERT-confusion-matrix](images/bert_confusion_matrix.png) | <img src="images/hate_speech_model_deployment.PNG" style="width: 275px;"> |


### More Projects
- **[Rental Price Prediction](https://github.com/JensBender/rental-price-prediction):** Predicting Singapore rental prices using XGBoost. Includes web scraping of properties, feature engineering via Google Maps API, and a Dockerized Flask application for deployment.
- **[Machine Learning Template](https://github.com/JensBender/machine-learning-template):** A versatile, ready-to-use machine learning template for tabular data. Streamlines EDA, data preprocessing, and modeling for regression and classification.
- **[ChatGPT Cover Letter Generator](https://github.com/JensBender/chatgpt-cover-letter-generator):** LLM-driven cover letter generator. Transforms job postings into tailored applications using the OpenAI API and Web Scraping.


## 🏅 Certifications & Courses

**ML in Production** | DeepLearning.AI | May 2025  
*Skills: MLOps · Data-Centric ML Lifecycle · Error Analysis*

**Hugging Face** | DeepLearning.AI | July 2024  
*Skills: Transformers · Multimodal AI · Gradio · HF Hub & Spaces*

**Data Engineering Foundations** | DeepLearning.AI | June 2024  
*Skills: Requirements Gathering · Tech Specs*

**IBM Data Engineering** | IBM | May 2024  
*Skills: ETL · Bash · Apache Airflow (DAGs)*

**Advanced SQL: MySQL for Ecommerce & Web Analytics** | Udemy | February 2024 | [🔗 Certificate](https://www.udemy.com/certificate/UC-ac04dd78-4589-4b2e-a863-7722cd78ec2f/)  
*Skills: MySQL · Business Intelligence · Data Analysis · Subqueries · Temporary Tables*

**A/B Testing** | Google | February 2024  
*Skills: Experimental Design · Metric Selection · Power Analysis · Sanity Checks*

**AWS Certified Cloud Practitioner** | AWS | January 2024 | [🔗 Certificate](https://www.credly.com/badges/3287f8a9-0dcd-48d2-afc3-c255faf027bc/public_url)  
*Skills: Amazon Web Services (AWS) · Cloud Concepts · Security & Compliance · Billing & Pricing*

**Ultimate AWS Certified Cloud Practitioner CLF-C02** | Udemy | January 2024 | [🔗 Certificate](https://www.udemy.com/certificate/UC-2090637d-9845-42f3-9f7b-97195874331a/)  
*Skills: Amazon Web Services (AWS) · Identity and Access Management (IAM) · Elastic Compute Cloud (EC2) · Simple Storage Service (S3) · Virtual Private Cloud (VPC) · CloudWatch · Database & Analytics*

**Spark and Python for Big Data with PySpark** | Udemy | January 2024 | [🔗 Certificate](https://www.udemy.com/certificate/UC-27da6f52-bc5f-4e72-bc5b-c2cd488566b0/)  
*Skills: Apache Spark · PySpark · Spark DataFrames · MLlib · Amazon Web Services (AWS) · Databricks*

**Microsoft Power BI Data Analyst** | Udemy | November 2023 | [🔗 Certificate](https://www.udemy.com/certificate/UC-eb56c820-8c91-4e03-8c57-efdc8c570c6b/)  
*Skills: Power BI · Power Query · Data Analysis Expressions (DAX) · Data Modeling · Interactive Dashboards*

**Product Management for AI** | 365 Data Science | November 2023  
*Skills: AI Strategy · UX · Stakeholder Management*

**LLM App Development** | OpenAI & DeepLearning.AI | June 2023  
*Skills: OpenAI API · Prompt Engineering*

**Deep Learning** | alfatraining Bildungszentrum GmbH | April 2023  
*Skills: TensorFlow · Neural Networks · Convolutional Neural Networks (CNN) · Computer Vision · Recurrent Neural Networks (RNN) · Long Short-Term Memory (LSTM) · Natural Language Processing (NLP) · Time Series Analysis*

**Machine Learning by Stanford University & DeepLearning.AI** | Coursera | April 2023 | [🔗 Certificate](https://coursera.org/share/1c62950a6100b0426d454b652e77498c)  
*Skills: Linear & Logistic Regression · Neural Networks · Recommender Systems · Reinforcement Learning*

**Python for Machine Learning & Data Science Masterclass** | Udemy | March 2023 | [🔗 Certificate](https://www.udemy.com/certificate/UC-4de79ac0-2282-45c9-93e1-a7cb6f812592/)  
*Skills: scikit-learn · Pandas · NumPy · Matplotlib · Seaborn · Random Forest · Gradient Boosting · Support Vector Machines (SVM) · DBSCAN*

**Machine Learning** | alfatraining Bildungszentrum GmbH | February 2023  
*Skills: Regression · K-Nearest Neighbors (KNN) · Decision Trees · Random Forest · Support Vector Machines (SVM) · Clustering · Principal Component Analysis (PCA) · Feature Engineering · Model Evaluation*

**The Ultimate MySQL Bootcamp: Go from SQL Beginner to Expert** | Udemy | December 2022 | [🔗 Certificate](https://www.udemy.com/certificate/UC-e324e4f7-95ba-4894-b8e0-65229ff5e2dc)  
*Skills: MySQL · Database Schemas · SQL Joins · Aggregate Functions · Window Functions*

**The Git & GitHub Bootcamp** | Udemy | September 2022  
*Skills: Git · GitHub · Version Control · Branching · Merging · Pull Requests*

**100 Days of Code: The Complete Python Pro Bootcamp** | Udemy | April 2022 – November 2022  
*Skills: Python · OOP · Flask · Web Scraping (Beautiful Soup, Selenium) · APIs (Requests) · Automation*


## 👨‍💻 GitHub Statistics
<p align="left">
  <a href="https://github.com/JensBender">
    <img src="https://github-readme-stats-eight-theta.vercel.app/api/top-langs?username=JensBender&layout=compact&theme=radical" alt="Top Languages" height="180px" />
  </a>
</p>


## ©️ Credits
<sup><small>Profile banner GIF based on the video by [RDNE Stock project](https://www.pexels.com/video/business-analytics-presentation-7947451/) from Pexels</small></sup>


<!-- MARKDOWN LINKS -->
[Airflow-badge]: https://img.shields.io/badge/Apache%20Airflow-017CEE?style=for-the-badge&logo=Apache%20Airflow&logoColor=white
[Airflow-url]: https://airflow.apache.org/
[AWS-badge]: https://img.shields.io/badge/AWS-%23FF9900?style=for-the-badge&logo=amazonaws&logoColor=white
[AWS-url]: https://aws.amazon.com/
[ChatGPT-badge]: https://img.shields.io/badge/chatGPT-74aa9c?style=for-the-badge&logo=openai&logoColor=white
[ChatGPT-url]: https://chatgpt.com/
[Docker-badge]: https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white
[Docker-url]: https://www.docker.com/
[DVC-badge]: https://img.shields.io/badge/DVC-13ADC7?style=for-the-badge&logo=dvc&logoColor=white
[DVC-url]: https://dvc.org/
[FastAPI-badge]: https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white
[FastAPI-url]: https://fastapi.tiangolo.com/
[Flask-badge]: https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white
[Flask-url]: https://flask.palletsprojects.com/en/2.3.x/
[Git-badge]: https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white
[Git-url]: https://git-scm.com/
[GitHub-badge]: https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white
[GitHub-url]: https://github.com/
[GitHubActions-badge]: https://img.shields.io/badge/github%20actions-%236E5494.svg?style=for-the-badge&logo=githubactions&logoColor=white
[GitHubActions-url]: https://github.com/features/actions
[Gradio-badge]: https://img.shields.io/badge/Gradio-fc7404?style=for-the-badge&logo=gradio&logoColor=white
[Gradio-url]: https://gradio.app
[HuggingFace-badge]: https://img.shields.io/badge/Hugging%20Face-ffcc00?style=for-the-badge&logo=huggingface&logoColor=black
[HuggingFace-url]: https://huggingface.co/
[Matplotlib-badge]: https://img.shields.io/badge/Matplotlib-%23DDDDDD?style=for-the-badge&logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxODAiIGhlaWdodD0iMTgwIiBzdHJva2U9ImdyYXkiPgo8ZyBzdHJva2Utd2lkdGg9IjIiIGZpbGw9IiNGRkYiPgo8Y2lyY2xlIGN4PSI5MCIgY3k9IjkwIiByPSI4OCIvPgo8Y2lyY2xlIGN4PSI5MCIgY3k9IjkwIiByPSI2NiIvPgo8Y2lyY2xlIGN4PSI5MCIgY3k9IjkwIiByPSI0NCIvPgo8Y2lyY2xlIGN4PSI5MCIgY3k9IjkwIiByPSIyMiIvPgo8cGF0aCBkPSJtOTAsMnYxNzZtNjItMjYtMTI0LTEyNG0xMjQsMC0xMjQsMTI0bTE1MC02MkgyIi8+CjwvZz48ZyBvcGFjaXR5PSIuOCI+CjxwYXRoIGZpbGw9IiM0NEMiIGQ9Im05MCw5MGgxOGExOCwxOCAwIDAsMCAwLTV6Ii8+CjxwYXRoIGZpbGw9IiNCQzMiIGQ9Im05MCw5MCAzNC00M2E1NSw1NSAwIDAsMC0xNS04eiIvPgo8cGF0aCBmaWxsPSIjRDkzIiBkPSJtOTAsOTAtMTYtNzJhNzQsNzQgMCAwLDAtMzEsMTV6Ii8+CjxwYXRoIGZpbGw9IiNEQjMiIGQ9Im05MCw5MC01OC0yOGE2NSw2NSAwIDAsMC01LDM5eiIvPgo8cGF0aCBmaWxsPSIjM0JCIiBkPSJtOTAsOTAtMzMsMTZhMzcsMzcgMCAwLDAgMiw1eiIvPgo8cGF0aCBmaWxsPSIjM0M5IiBkPSJtOTAsOTAtMTAsNDVhNDYsNDYgMCAwLDAgMTgsMHoiLz4KPHBhdGggZmlsbD0iI0Q3MyIgZD0ibTkwLDkwIDQ2LDU4YTc0LDc0IDAgMCwwIDEyLTEyeiIvPgo8L2c+PC9zdmc+
[Matplotlib-url]: https://matplotlib.org/
[MLflow-badge]: https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=MLflow&logoColor=white
[MLflow-url]: https://mlflow.org/
[MySQL-badge]: https://img.shields.io/badge/mysql-%2300f.svg?style=for-the-badge&logo=mysql&logoColor=white
[MySQL-url]: https://www.mysql.com/
[NumPy-badge]: https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white
[NumPy-url]: https://numpy.org/
[Pandas-badge]: https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white
[Pandas-url]: https://pandas.pydata.org/
[PowerBI-badge]: https://img.shields.io/badge/power_bi-F2C811?style=for-the-badge&logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iNjMwcHgiIGhlaWdodD0iNjMwcHgiIHZpZXdCb3g9IjAgMCA2MzAgNjMwIiB2ZXJzaW9uPSIxLjEiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgeG1sbnM6eGxpbms9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGxpbmsiPgogICAgPCEtLSBHZW5lcmF0b3I6IFNrZXRjaCA1My4yICg3MjY0MykgLSBodHRwczovL3NrZXRjaGFwcC5jb20gLS0+CiAgICA8dGl0bGU+UEJJIExvZ288L3RpdGxlPgogICAgPGRlc2M+Q3JlYXRlZCB3aXRoIFNrZXRjaC48L2Rlc2M+CiAgICA8ZGVmcz4KICAgICAgICA8bGluZWFyR3JhZGllbnQgeDE9IjUwJSIgeTE9IjAlIiB4Mj0iNTAlIiB5Mj0iMTAwJSIgaWQ9ImxpbmVhckdyYWRpZW50LTEiPgogICAgICAgICAgICA8c3RvcCBzdG9wLWNvbG9yPSIjRUJCQjE0IiBvZmZzZXQ9IjAlIj48L3N0b3A+CiAgICAgICAgICAgIDxzdG9wIHN0b3AtY29sb3I9IiNCMjU0MDAiIG9mZnNldD0iMTAwJSI+PC9zdG9wPgogICAgICAgIDwvbGluZWFyR3JhZGllbnQ+CiAgICAgICAgPGxpbmVhckdyYWRpZW50IHgxPSI1MCUiIHkxPSIwJSIgeDI9IjUwJSIgeTI9IjEwMCUiIGlkPSJsaW5lYXJHcmFkaWVudC0yIj4KICAgICAgICAgICAgPHN0b3Agc3RvcC1jb2xvcj0iI0Y5RTU4MyIgb2Zmc2V0PSIwJSI+PC9zdG9wPgogICAgICAgICAgICA8c3RvcCBzdG9wLWNvbG9yPSIjREU5ODAwIiBvZmZzZXQ9IjEwMCUiPjwvc3RvcD4KICAgICAgICA8L2xpbmVhckdyYWRpZW50PgogICAgICAgIDxwYXRoIGQ9Ik0zNDYsNjA0IEwzNDYsNjMwIEwzMjAsNjMwIEwxNTMsNjMwIEMxMzguNjQwNTk3LDYzMCAxMjcsNjE4LjM1OTQwMyAxMjcsNjA0IEwxMjcsMTgzIEMxMjcsMTY4LjY0MDU5NyAxMzguNjQwNTk3LDE1NyAxNTMsMTU3IEwzMjAsMTU3IEMzMzQuMzU5NDAzLDE1NyAzNDYsMTY4LjY0MDU5NyAzNDYsMTgzIEwzNDYsNjA0IFoiIGlkPSJwYXRoLTMiPjwvcGF0aD4KICAgICAgICA8ZmlsdGVyIHg9Ii05LjElIiB5PSItNi4zJSIgd2lkdGg9IjEzNi41JSIgaGVpZ2h0PSIxMTYuOSUiIGZpbHRlclVuaXRzPSJvYmplY3RCb3VuZGluZ0JveCIgaWQ9ImZpbHRlci00Ij4KICAgICAgICAgICAgPGZlT2Zmc2V0IGR4PSIyMCIgZHk9IjEwIiBpbj0iU291cmNlQWxwaGEiIHJlc3VsdD0ic2hhZG93T2Zmc2V0T3V0ZXIxIj48L2ZlT2Zmc2V0PgogICAgICAgICAgICA8ZmVHYXVzc2lhbkJsdXIgc3RkRGV2aWF0aW9uPSIxMCIgaW49InNoYWRvd09mZnNldE91dGVyMSIgcmVzdWx0PSJzaGFkb3dCbHVyT3V0ZXIxIj48L2ZlR2F1c3NpYW5CbHVyPgogICAgICAgICAgICA8ZmVDb2xvck1hdHJpeCB2YWx1ZXM9IjAgMCAwIDAgMCAgIDAgMCAwIDAgMCAgIDAgMCAwIDAgMCAgMCAwIDAgMC4wNTMwMjExOTc2IDAiIHR5cGU9Im1hdHJpeCIgaW49InNoYWRvd0JsdXJPdXRlcjEiPjwvZmVDb2xvck1hdHJpeD4KICAgICAgICA8L2ZpbHRlcj4KICAgICAgICA8bGluZWFyR3JhZGllbnQgeDE9IjUwJSIgeTE9IjAlIiB4Mj0iNTAlIiB5Mj0iMTAwJSIgaWQ9ImxpbmVhckdyYWRpZW50LTUiPgogICAgICAgICAgICA8c3RvcCBzdG9wLWNvbG9yPSIjRjlFNjhCIiBvZmZzZXQ9IjAlIj48L3N0b3A+CiAgICAgICAgICAgIDxzdG9wIHN0b3AtY29sb3I9IiNGM0NEMzIiIG9mZnNldD0iMTAwJSI+PC9zdG9wPgogICAgICAgIDwvbGluZWFyR3JhZGllbnQ+CiAgICA8L2RlZnM+CiAgICA8ZyBpZD0iUEJJLUxvZ28iIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJHcm91cCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoNzcuNTAwMDAwLCAwLjAwMDAwMCkiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiBmaWxsPSJ1cmwoI2xpbmVhckdyYWRpZW50LTEpIiB4PSIyNTYiIHk9IjAiIHdpZHRoPSIyMTkiIGhlaWdodD0iNjMwIiByeD0iMjYiPjwvcmVjdD4KICAgICAgICAgICAgPGcgaWQ9IkNvbWJpbmVkLVNoYXBlIj4KICAgICAgICAgICAgICAgIDx1c2UgZmlsbD0iYmxhY2siIGZpbGwtb3BhY2l0eT0iMSIgZmlsdGVyPSJ1cmwoI2ZpbHRlci00KSIgeGxpbms6aHJlZj0iI3BhdGgtMyI+PC91c2U+CiAgICAgICAgICAgICAgICA8dXNlIGZpbGw9InVybCgjbGluZWFyR3JhZGllbnQtMikiIGZpbGwtcnVsZT0iZXZlbm9kZCIgeGxpbms6aHJlZj0iI3BhdGgtMyI+PC91c2U+CiAgICAgICAgICAgIDwvZz4KICAgICAgICAgICAgPHBhdGggZD0iTTIxOSw2MDQgTDIxOSw2MzAgTDE5Myw2MzAgTDI2LDYzMCBDMTEuNjQwNTk2NSw2MzAgMS43NTg1MTk3NWUtMTUsNjE4LjM1OTQwMyAwLDYwNCBMMCwzNDEgQy0xLjc1ODUxOTc1ZS0xNSwzMjYuNjQwNTk3IDExLjY0MDU5NjUsMzE1IDI2LDMxNSBMMTkzLDMxNSBDMjA3LjM1OTQwMywzMTUgMjE5LDMyNi42NDA1OTcgMjE5LDM0MSBMMjE5LDYwNCBaIiBpZD0iQ29tYmluZWQtU2hhcGUiIGZpbGw9InVybCgjbGluZWFyR3JhZGllbnQtNSkiPjwvcGF0aD4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==
[PowerBI-url]: https://powerbi.microsoft.com/en-us/
[Plotly-badge]: https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white
[Plotly-url]: https://plotly.com/python/
[Pydantic-badge]: https://img.shields.io/badge/Pydantic-3776AB?style=for-the-badge&logo=pydantic&logoColor=white
[Pydantic-url]: https://docs.pydantic.dev/
[Python-badge]: https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[Pytest-badge]: https://img.shields.io/badge/pytest-%23F0F0F0?style=for-the-badge&logo=pytest&logoColor=2f9fe3
[Pytest-url]: https://docs.pytest.org/
[Python-url]: https://www.python.org/
[scikit-learn-badge]: https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white
[scikit-learn-url]: https://scikit-learn.org/stable/
<!-- Seaborn logo: https://github.com/mwaskom/seaborn/blob/master/doc/_static/logo-mark-darkbg.svg (PNG export). -->
[Seaborn-badge]: https://img.shields.io/badge/seaborn-%230C4A89.svg?style=for-the-badge&logo=data:image/png;base64,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
[Seaborn-url]: https://seaborn.pydata.org/
[Selenium-badge]: https://img.shields.io/badge/selenium-%43B02A.svg?style=for-the-badge&logo=selenium&logoColor=white
[Selenium-url]: https://www.selenium.dev/
[Spark-badge]: https://img.shields.io/badge/Apache%20Spark-E25A1C.svg?style=for-the-badge&logo=Apache-Spark&logoColor=white
[Spark-url]: https://spark.apache.org/
[TensorFlow-badge]: https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white
[TensorFlow-url]: https://www.tensorflow.org/
<!-- XGBoost rocket: https://xgboost.ai/images/logo/xgboost-logo.svg (recolored white). -->
[XGBoost-badge]: https://img.shields.io/badge/XGBoost-0068C7?style=for-the-badge&logo=data:image/svg%2Bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9Ii0zNC43MDMyNSAtMzQuNzAzMjUgNDg1Ljg0NTUgNDg1Ljg0NTUiIGZpbGw9IiNmZmYiPjxwYXRoIGQ9Im0yOC44OCwzMzkuNDU5Yy0yLjU1OSwwLTUuMTE5LC0wLjk3Ny03LjA3MSwtMi45MjktMy45MDUsLTMuOTA1LTMuOTA1LC0xMC4yMzcgMCwtMTQuMTQzbDIwLjU0LC0yMC41NGMzLjkwNSwtMy45MDQgMTAuMjM3LC0zLjkwNCAxNC4xNDMsMCAzLjkwNSwzLjkwNSAzLjkwNSwxMC4yMzcgMCwxNC4xNDNsLTIwLjU0LDIwLjU0Yy0xLjk1MywxLjk1Mi00LjUxMiwyLjkyOS03LjA3MiwyLjkyOXoiLz48cGF0aCBkPSJtMTAsNDE2LjQzOWMtMi41NiwwLTUuMTE5LC0wLjk3Ny03LjA3MiwtMi45My0zLjkwNSwtMy45MDUtMy45MDQsLTEwLjIzNyAxMGUtNCwtMTQuMTQybDY4LjQ3LC02OC40NmMzLjkwNSwtMy45MDQgMTAuMjM3LC0zLjkwNCAxNC4xNDIsMC4wMDEgMy45MDUsMy45MDUgMy45MDQsMTAuMjM3LTAuMDAyLDE0LjE0MmwtNjguNDcsNjguNDZjLTEuOTUxLDEuOTUzLTQuNTEsMi45MjktNy4wNjksMi45Mjl6Ii8+PHBhdGggZD0ibTczLjI5LDQxMS4yNTljLTIuNTYsMC01LjExOCwtMC45NzctNy4wNzEsLTIuOTI5LTMuOTA1LC0zLjkwNS0zLjkwNSwtMTAuMjM3IDAsLTE0LjE0M2wzNC4yMywtMzQuMjI5YzMuOTA1LC0zLjkwNCAxMC4yMzcsLTMuOTAzIDE0LjE0MiwwIDMuOTA1LDMuOTA1IDMuOTA1LDEwLjIzNyAwLDE0LjE0M2wtMzQuMjMsMzQuMjI5Yy0xLjk1MiwxLjk1Mi00LjUxMiwyLjkyOS03LjA3MSwyLjkyOXoiLz48cGF0aCBkPSJtMjA4LjUyLDMzNi4yMzljLTIuNTYsMC01LjExOCwtMC45NzctNy4wNzEsLTIuOTI5TDgzLjEzOSwyMTVjLTMuOTA1LC0zLjkwNS0zLjkwNSwtMTAuMjM3IDAsLTE0LjE0MyAzLjkwNSwtMy45MDQgMTAuMjM3LC0zLjkwNCAxNC4xNDMsMGwxMTguMzEsMTE4LjMxMWMzLjkwNSwzLjkwNSAzLjkwNSwxMC4yMzcgMCwxNC4xNDMtMS45NTMsMS45NTItNC41MTMsMi45MjgtNy4wNzIsMi45Mjh6Ii8+PHBhdGggZD0ibTI1OS41OCwyMTguNTM0Yy0xNi40NzQsMC0zMS45NTksLTYuNDE2LTQzLjYwNCwtMTguMDY2LTExLjY0NiwtMTEuNjQxLTE4LjA2MiwtMjcuMTI2LTE4LjA2MiwtNDMuNiAwLC0xNi40NzQgNi40MTYsLTMxLjk1OSAxOC4wNjUsLTQzLjYwNCAxMS42NDEsLTExLjY0NiAyNy4xMjYsLTE4LjA2MiA0My42LC0xOC4wNjIgMTYuNDc0LDAgMzEuOTU5LDYuNDE2IDQzLjYwNCwxOC4wNjUgMTEuNjQ1LDExLjY0MSAxOC4wNjEsMjcuMTI2IDE4LjA2MSw0My42IDAsMTYuNDcyLTYuNDE1LDMxLjk1Ni0xOC4wNjEsNDMuNnYwYy0xMGUtNCwwLjAwMi0xMGUtNCwxMGUtNC0wLjAwNCwwLjAwNC0xMS42NDMsMTEuNjQ4LTI3LjEyNywxOC4wNjMtNDMuNTk5LDE4LjA2M3ptMCwtMTAzLjMzYy0xMS4xMywwLTIxLjU5Miw0LjMzNC0yOS40NTcsMTIuMjA0LTcuODc0LDcuODY5LTEyLjIwOCwxOC4zMzEtMTIuMjA4LDI5LjQ2MSAwLDExLjEzIDQuMzM0LDIxLjU5MiAxMi4yMDQsMjkuNDU3IDcuODY5LDcuODc0IDE4LjMzMSwxMi4yMDggMjkuNDYxLDEyLjIwOCAxMS4xMywwIDIxLjU5MiwtNC4zMzQgMjkuNDU3LC0xMi4yMDQgMC4wMDIsLTAuMDAxIDAuMDAzLC0wLjAwMiAwLjAwNCwtMC4wMDQgNy44NywtNy44NjUgMTIuMjA0LC0xOC4zMjcgMTIuMjA0LC0yOS40NTcgMCwtMTEuMTMtNC4zMzQsLTIxLjU5Mi0xMi4yMDQsLTI5LjQ1Ny03Ljg2OSwtNy44NzQtMTguMzMxLC0xMi4yMDgtMjkuNDYxLC0xMi4yMDh6Ii8+PHBhdGggZD0ibTg5LjI5MSwyOTYuMzFjLTEuODEsMC0zLjY0MiwtMC40OS01LjI5LC0xLjUyMS00LjY4NCwtMi45MjYtNi4xMDgsLTkuMDk2LTMuMTgyLC0xMy43NzlsMzAuNDksLTQ4LjhjMi45MjcsLTQuNjg0IDkuMDk3LC02LjExIDEzLjc4LC0zLjE4MiA0LjY4NCwyLjkyNiA2LjEwOCw5LjA5NiAzLjE4MiwxMy43NzlsLTMwLjQ5LDQ4LjhjLTEuODk3LDMuMDM2LTUuMTU2LDQuNzAzLTguNDksNC43MDN6Ii8+PHBhdGggZD0ibTEwOS43MiwzMTYuNzM5Yy0yLjU1OSwwLTUuMTE4LC0wLjk3Ny03LjA3MSwtMi45MjktMy45MDUsLTMuOTA1LTMuOTA2LC0xMC4yMzctMC4wMDEsLTE0LjE0M2wzOS42NSwtMzkuNjVjMy45MDUsLTMuOTA0IDEwLjIzNywtMy45MDQgMTQuMTQyLDAgMy45MDUsMy45MDUgMy45MDYsMTAuMjM3IDAuMDAxLDE0LjE0MmwtMzkuNjUsMzkuNjVjLTEuOTUyLDEuOTU0LTQuNTEyLDIuOTMtNy4wNzEsMi45M3oiLz48cGF0aCBkPSJtMTMwLjE2LDMzNy4xNmMtMy4zMzQsMC02LjU5MywtMS42NjYtOC40OSwtNC43MDItMi45MjYsLTQuNjg0LTEuNTAxLC0xMC44NTQgMy4xODIsLTEzLjc3OWw0OC44LC0zMC40OWM0LjY4MywtMi45MjkgMTAuODUzLC0xLjUwMyAxMy43OCwzLjE4MiAyLjkyNiw0LjY4NCAxLjUwMSwxMC44NTMtMy4xODIsMTMuNzc5bC00OC44LDMwLjQ5Yy0xLjY0OSwxLjAzLTMuNDgsMS41Mi01LjI5LDEuNTJ6Ii8+PHBhdGggZD0ibTE3Ny4zNTYsMTE2LjUwOWMtMi4zMDcsMC00LjYyNSwtMC43OTQtNi41MTIsLTIuNDE1LTQuMTg5LC0zLjU5OS00LjY2OCwtOS45MTItMS4wNjksLTE0LjEwMmwzMy43MSwtMzkuMjRjMy41OTgsLTQuMTg4IDkuOTExLC00LjY2OCAxNC4xMDIsLTEuMDY4IDQuMTg5LDMuNTk5IDQuNjY4LDkuOTEyIDEuMDY4LDE0LjEwMWwtMzMuNzEsMzkuMjRjLTEuOTc3LDIuMzAyLTQuNzc1LDMuNDg0LTcuNTg5LDMuNDg0eiIvPjxwYXRoIGQ9Im0xNTguMjY1LDM1Mi43ODdjLTEwLjQ0OCwwLTIwLjcyMywtNC4wODUtMjguMzQsLTExLjcxMmwtNi41ODIsLTYuNTgyYy0wLjA5MywtMC4wODYtMC4xODQsLTAuMTczLTAuMjczLC0wLjI2M0w3NS4zNzYsMjg2LjUzNUM2NC4zODQsMjc1LjUyOSA2MC43NTMsMjU5LjAwNCA2Ni4xMTcsMjQ0LjQyNmwxNC43MSwtMzkuOTUyYzAuNDEzLC0xLjEyIDEuMDIyLC0yLjE1NyAxLjc5OSwtMy4wNjFsODcuMTQsLTEwMS40MmMzLjYwMSwtNC4xODggOS45MTMsLTQuNjY3IDE0LjEwMiwtMS4wNjggNC4xODksMy42IDQuNjY3LDkuOTEzIDEuMDY4LDE0LjEwMmwtODUuOTY1LDEwMC4wNS0xNC4wODYsMzguMjU3Yy0yLjY4Miw3LjI4OS0wLjg2NCwxNS41NTYgNC42MzIsMjEuMDU5bDQ3LjQzMiw0Ny40MzNjMC4wOTIsMC4wODYgMC4xODQsMC4xNzMgMC4yNzMsMC4yNjNsNi44NSw2Ljg1YzUuNDk3LDUuNTA0IDEzLjc1Niw3LjMxOCAyMS4wNDgsNC42M0wyMDMuMzcyLDMxNy40OCAzNDIuNjc0LDE5Ny44MDVjNC4xOTEsLTMuNiAxMC41MDQsLTMuMTE5IDE0LjEwMiwxLjA2OCAzLjYsNC4xODkgMy4xMjEsMTAuNTAzLTEuMDY4LDE0LjEwMkwyMTUuMDM2LDMzMy44MjRjLTAuOTA0LDAuNzc3LTEuOTQsMS4zODctMy4wNTksMS43OTlsLTM5Ljk0MSwxNC43MWMtNC40NzksMS42NTItOS4xNDMsMi40NTQtMTMuNzcxLDIuNDU0eiIvPjxwYXRoIGQ9Im0zNDkuMTksMjE1LjM4OWMtMi41NTksMC01LjExOCwtMC45NzctNy4wNzEsLTIuOTI5LTMuOTA1LC0zLjkwNS0zLjkwNSwtMTAuMjM3IDAsLTE0LjE0MyAxOS44ODUsLTE5Ljg4NCAzNC42NDIsLTQzLjMxNSA0My44NjMsLTY5LjY0NCAxMS43MzYsLTMzLjUxMiAxMy42MjYsLTY5LjI1IDUuNTM2LC0xMDMuNzMzLTM0LjQ4LC04LjA4OS03MC4yMjEsLTYuMTk5LTEwMy43MzMsNS41MzYtMjYuMzI5LDkuMjIxLTQ5Ljc2MSwyMy45NzktNjkuNjQ1LDQzLjg2My0zLjkwNSwzLjkwNC0xMC4yMzYsMy45MDUtMTQuMTQzLDAtMy45MDUsLTMuOTA1LTMuOTA1LC0xMC4yMzcgMCwtMTQuMTQzIDIyLjAyNSwtMjIuMDI0IDQ3Ljk5MSwtMzguMzc1IDc3LjE3NiwtNDguNTk2IDM5LjE1OCwtMTMuNzExIDgxLjA1OCwtMTUuMjkgMTIxLjE3MSwtNC41NjEgMy40NTQsMC45MjQgNi4xNTIsMy42MjIgNy4wNzYsNy4wNzYgMTAuNzI4LDQwLjExNCA5LjE1MSw4Mi4wMTQtNC41NjMsMTIxLjE3LTEwLjIyMSwyOS4xODUtMjYuNTcxLDU1LjE1LTQ4LjU5Niw3Ny4xNzUtMS45NTIsMS45NTItNC41MTEsMi45MjktNy4wNzEsMi45Mjl6Ii8+PHBhdGggZD0ibTM5NS40MSwxNDEuOThjLTIuNTYsMC01LjExOCwtMC45NzctNy4wNzEsLTIuOTI5TDI3Ny40MDksMjguMTJjLTMuOTA1LC0zLjkwNS0zLjkwNSwtMTAuMjM3IDAsLTE0LjE0MyAzLjkwOCwtMy45MDUgMTAuMjM4LC0zLjkwMyAxNC4xNDMsMGwxMTAuOTMsMTEwLjkzMWMzLjkwNSwzLjkwNSAzLjkwNSwxMC4yMzcgMCwxNC4xNDMtMS45NTQsMS45NTItNC41MTMsMi45MjktNy4wNzIsMi45Mjl6Ii8+PHBhdGggZD0ibTkwLjIyLDIxNy45MjljLTAuODMyLDAtMS42NywtMC4xMDQtMi40NzcsLTAuMzA5bC02Ni4zMywtMTYuODhjLTMuMDM3LC0wLjc3My01LjUzNywtMi45MjYtNi43NTEsLTUuODE0LTEuMjE1LC0yLjg4OS0xLjAwNSwtNi4xODEgMC41NjYsLTguODkyIDcuNzc4LC0xMy40MTggMTcuMzU1LC0yNS44NiAyOC40NjcsLTM2Ljk4MiAzNS4yODEsLTM1LjI4MSA4NC4xMTksLTU0LjQ0NSAxMzMuOTg4LC01Mi41MzcgNS4zNjksMC4xNzYgOS42NzEsNC41ODMgOS42NzEsOS45OTQgMCw1LjUyMi00LjQ3MiwxMC05Ljk5NSwxMGgtMC4wMWMtMC4xMjcsMC0wLjI1NCwtMC4wMDItMC4zODEsLTAuMDA3LTQ0LjMzOCwtMS42OTktODcuNzY1LDE1LjMyNS0xMTkuMTI3LDQ2LjY4OC02LjY4NCw2LjY4OS0xMi43NDIsMTMuOTE0LTE4LjEwMSwyMS41NzZsNTIuNzMsMTMuNDE5YzQuNDM1LDEuMDI0IDcuNzQ1LDQuOTk4IDcuNzQ1LDkuNzQzIDAsNS41MjMtNC40NzIsMTAuMDAxLTkuOTk1LDEwLjAwMXoiLz48cGF0aCBkPSJtMjI1LjQxLDQwMi41NzljLTEuMzE1LDAtMi42MzMsLTAuMjU5LTMuODc2LC0wLjc4Mi0yLjg5LC0xLjIxNS01LjA0MiwtMy43MTQtNS44MTUsLTYuNzVsLTE2Ljg5MSwtNjYuMzRjLTEuMzYzLC01LjM1MyAxLjg3MiwtMTAuNzk2IDcuMjI0LC0xMi4xNTggNS4zNDksLTEuMzY2IDEwLjc5NSwxLjg3MSAxMi4xNTgsNy4yMjNsMTMuNDgsNTIuOTQ4YzcuNjYzLC01LjM1OSAxNC44ODksLTExLjQxOSAyMS41ODEsLTE4LjEwNCAzMS4zNiwtMzEuMzYgNDguMzc4LC03NC43ODUgNDYuNjg0LC0xMTkuMTM2LTAuMjEsLTUuNTE5IDQuMDkzLC0xMC4xNjMgOS42MTEsLTEwLjM3NCA1LjUwOSwtMC4yMzMgMTAuMTY0LDQuMDkzIDEwLjM3NSw5LjYxMSAxLjkwMyw0OS44OTctMTcuMjQzLDk4Ljc1NS01Mi41MzIsMTM0LjA0NC0xMS4xMjQsMTEuMTEzLTIzLjU2NywyMC42OTEtMzYuOTg2LDI4LjQ3LTEuNTQyLDAuODk1LTMuMjc1LDEuMzQ4LTUuMDEzLDEuMzQ4eiIvPjwvc3ZnPg==
[XGBoost-url]: https://xgboost.readthedocs.io/
<!-- SHAP symbol: https://github.com/shap/shap/blob/master/docs/artwork/shap_logo_scratch.ai -->
[SHAP-badge]: https://img.shields.io/badge/SHAP-343131?style=for-the-badge&logo=data:image/svg%2Bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA1NC44OTY4IDY5LjU0MDEiPjxnIGZpbGw9IiNmZmYiIHRyYW5zZm9ybT0ibWF0cml4KDEgMCAwIC0xIC0zNjYuNTA0OSA0NTguNDk1OSkiPjxwYXRoIHRyYW5zZm9ybT0idHJhbnNsYXRlKDQxNy4zNTE3IDQ0Ny4yNDY5KSIgZD0iTTAgMCBMMCAwIEMyLjIzNiAwIDQuMDUgLTEuODEzIDQuMDUgLTQuMDUgTDQuMDUgLTEyLjMxMyBDNC4wNSAtMTQuNTUgMi4yMzYgLTE2LjM2MyAwIC0xNi4zNjMgQy0yLjIzNyAtMTYuMzYzIC00LjA1IC0xNC41NSAtNC4wNSAtMTIuMzEzIEwtNC4wNSAtNC4wNSBDLTQuMDUgLTEuODEzIC0yLjIzNyAwIDAgMCIvPjxwYXRoIHRyYW5zZm9ybT0idHJhbnNsYXRlKDQxMy4zMDE5IDQyMy43NjY0KSIgZD0iTTAgMCBDMCAyLjIzNyAxLjgxMiA0LjA1IDQuMDUgNC4wNSBDNi4yODYgNC4wNSA4LjEgMi4yMzcgOC4xIDAgTDguMSAtMTkuNTEyIEM4LjEgLTIxLjc0OSA2LjI4NiAtMjMuNTYyIDQuMDUgLTIzLjU2MiBDMS44MTIgLTIzLjU2MiAwIC0yMS43NDkgMCAtMTkuNTEyIEgiLz48cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgzODIuMjU0MSA0MTguNjEyMSkiIGQ9Ik0wIDAgTDAgMCBDMi4yMzYgMCA0LjA1IC0xLjgxMyA0LjA1IC00LjA1IEw0LjA1IC0yMS41MTcgQzQuMDUgLTIzLjc1MyAyLjIzNiAtMjUuNTY2IDAgLTI1LjU2NiBDLTIuMjM3IC0yNS41NjYgLTQuMDUgLTIzLjc1MyAtNC4wNSAtMjEuNTE3IEwtNC4wNSAtNC4wNSBDLTQuMDUgLTEuODEzIC0yLjIzNyAwIDAgMCIvPjxwYXRoIHRyYW5zZm9ybT0idHJhbnNsYXRlKDQwNS42NTI1IDQzNS45OTczKSIgZD0iTTAgMCBMMCAwIEMyLjIzNiAwIDQuMDUgLTEuODEzIDQuMDUgLTQuMDUgTDQuMDUgLTM1LjgzNCBDNC4wNSAtMzguMDcgMi4yMzYgLTM5Ljg4NCAwIC0zOS44ODQgQy0yLjIzNyAtMzkuODg0IC00LjA1IC0zOC4wNyAtNC4wNSAtMzUuODM0IEwtNC4wNSAtNC4wNSBDLTQuMDUgLTEuODEzIC0yLjIzNyAwIDAgMCIvPjxwYXRoIHRyYW5zZm9ybT0idHJhbnNsYXRlKDM4Mi4yNTQxIDQ1Ni40NTA1KSIgZD0iTTAgMCBMMCAwIEMyLjIzNiAwIDQuMDUgLTEuODEzIDQuMDUgLTQuMDQ5IEw0LjA1IC0zMC43MjEgQzQuMDUgLTMyLjk1NyAyLjIzNiAtMzQuNzcgMCAtMzQuNzcgQy0yLjIzNyAtMzQuNzcgLTQuMDUgLTMyLjk1NyAtNC4wNSAtMzAuNzIxIEwtNC4wNSAtNC4wNDkgQy00LjA1IC0xLjgxMyAtMi4yMzcgMCAwIDAiLz48cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSg0MDUuNjUyNSA0NTUuNDI4KSIgZD0iTTAgMCBMMCAwIEMyLjIzNiAwIDQuMDUgLTEuODEzIDQuMDUgLTQuMDUgTDQuMDUgLTEyLjMxMyBDNC4wNSAtMTQuNTQ5IDIuMjM2IC0xNi4zNjMgMCAtMTYuMzYzIEMtMi4yMzcgLTE2LjM2MyAtNC4wNSAtMTQuNTQ5IC00LjA1IC0xMi4zMTMgTC00LjA1IC00LjA1IEMtNC4wNSAtMS44MTMgLTIuMjM3IDAgMCAwIi8+PHBhdGggdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMzkzLjk1MzMgNDI1Ljc3MDgpIiBkPSJNMCAwIEwwIDAgQzIuMjM2IDAgNC4wNSAtMS44MTMgNC4wNSAtNC4wNSBMNC4wNSAtMzIuNzY2IEM0LjA1IC0zNS4wMDIgMi4yMzYgLTM2LjgxNSAwIC0zNi44MTUgQy0yLjIzNyAtMzYuODE1IC00LjA1IC0zNS4wMDIgLTQuMDUgLTMyLjc2NiBMLTQuMDUgLTQuMDUgQy00LjA1IC0xLjgxMyAtMi4yMzcgMCAwIDAiLz48cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgzOTMuOTUzMyA0NTguNDk1OSkiIGQ9Ik0wIDAgTDAgMCBDMi4yMzYgMCA0LjA1IC0xLjgxMyA0LjA1IC00LjA1IEw0LjA1IC0yNS42MDcgQzQuMDUgLTI3Ljg0NCAyLjIzNiAtMjkuNjU3IDAgLTI5LjY1NyBDLTIuMjM3IC0yOS42NTcgLTQuMDUgLTI3Ljg0NCAtNC4wNSAtMjUuNjA3IEwtNC4wNSAtNC4wNSBDLTQuMDUgLTEuODEzIC0yLjIzNyAwIDAgMCIvPjxwYXRoIHRyYW5zZm9ybT0idHJhbnNsYXRlKDM3MC41NTQ5IDQwMC4yMDQ0KSIgZD0iTTAgMCBMMCAwIEMtMi4yMzcgMCAtNC4wNSAxLjgxMyAtNC4wNSA0LjA1IEwtNC4wNSAxMi4zMTMgQy00LjA1IDE0LjU0OSAtMi4yMzcgMTYuMzYzIDAgMTYuMzYzIEMyLjIzNiAxNi4zNjMgNC4wNSAxNC41NDkgNC4wNSAxMi4zMTMgTDQuMDUgNC4wNSBDNC4wNSAxLjgxMyAyLjIzNiAwIDAgMCIvPjxwYXRoIHRyYW5zZm9ybT0idHJhbnNsYXRlKDM3NC42MDQ3IDQyMy42ODQ4KSIgZD0iTTAgMCBDMCAtMi4yMzcgLTEuODEzIC00LjA1IC00LjA1IC00LjA1IEMtNi4yODcgLTQuMDUgLTguMSAtMi4yMzcgLTguMSAwIEwtOC4xIDE5LjUxMiBDLTguMSAyMS43NDkgLTYuMjg3IDIzLjU2MiAtNC4wNSAyMy41NjIgQy0xLjgxMyAyMy41NjIgMCAyMS43NDkgMCAxOS41MTIgSCIvPjwvZz48L3N2Zz4=
[SHAP-url]: https://shap.readthedocs.io/
