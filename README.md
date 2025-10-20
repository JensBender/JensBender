![Profile-banner](images/profile-banner.gif)

# Hi there! I'm Jens

<!-- ABOUT ME -->
## 👋 About Me
I'm an enthusiastic **data scientist** with over eight years of experience in data analysis, data visualization, and data storytelling. I enjoy solving challenging problems, harnessing the power of machine learning to derive valuable insights, and effectively communicating complex information.


<!-- SKILLS -->
## 🛠️ Skills

| Category                 | Skill    |
| ------------------------ | -------- |
| Programming              | [![Python][Python-badge]][Python-url] [![MySQL][MySQL-badge]][MySQL-url] |
| Data Manipulation        | [![NumPy][NumPy-badge]][NumPy-url] [![Pandas][Pandas-badge]][Pandas-url] |
| Data Visualization       | [![Matplotlib][Matplotlib-badge]][Matplotlib-url] [![Seaborn][Seaborn-badge]][Seaborn-url] [![Plotly][Plotly-badge]][Plotly-url] [![Power BI][PowerBI-badge]][PowerBI-url] |
| AI & Machine Learning    | [![scikit-learn][scikit-learn-badge]][scikit-learn-url] [![TensorFlow][TensorFlow-badge]][TensorFlow-url] [![Hugging Face][HuggingFace-badge]][HuggingFace-url] |
| Big Data                 | [![Spark][Spark-badge]][Spark-url] |
| Web Development          | [![FastAPI][FastAPI-badge]][FastAPI-url] [![Flask][Flask-badge]][Flask-url] [![Gradio][Gradio-badge]][Gradio-url] [![Pydantic][Pydantic-badge]][Pydantic-url] |
| Version Control          | [![Git][Git-badge]][Git-url] [![GitHub][GitHub-badge]][GitHub-url] |
| DevOps                   | [![Docker][Docker-badge]][Docker-url] [![Airflow][Airflow-badge]][Airflow-url] |
| Testing                  | [![pytest][Pytest-badge]][Pytest-url] [![Selenium][Selenium-badge]][Selenium-url] |
| Cloud                    | [![AWS][AWS-badge]][AWS-url] |
| Development Environments | [![Jupyter Notebook][JupyterNotebook-badge]][JupyterNotebook-url] [![VS Code][VSCode-badge]][VSCode-url] [![PyCharm][PyCharm-badge]][PyCharm-url] [![Spyder][Cursor-badge]][Cursor-url] |


<!-- PORTFOLIO -->
## 💻 Portfolio

### [Project 1: ETL Pipeline for YouTube Channel Analytics](https://github.com/JensBender/youtube-channel-analytics)
[![Python][Python-badge]][Python-url] [![MySQL][MySQL-badge]][MySQL-url] [![Airflow][Airflow-badge]][Airflow-url] [![Docker][Docker-badge]][Docker-url] [![AWS][AWS-badge]][AWS-url] [![Hugging Face][HuggingFace-badge]][HuggingFace-url] [![Power BI][PowerBI-badge]][PowerBI-url]  
To empower YouTube content creators and marketers with actionable insights into their channel's performance, especially in comparison to related channels, I developed a comprehensive **ETL pipeline** and designed an interactive **Power BI report**. This project involved:

- **Data Extraction**: Utilized the YouTube API to gather extensive data from three selected channels, including videos and comments.
- **Data Transformation**: Performed sentiment analysis on video comments via API requests to a RoBERTa sentiment analysis model, which I deployed using Gradio on a private Hugging Face Space.
- **Data Loading**: Stored the transformed data in a MySQL database hosted on AWS.
- **Automation**: Managed the ETL workflow using Apache Airflow, Docker, and AWS.
- **Data Visualization**: Designed an interactive Power BI report to deliver insigths into channel performance, featuring key metrics and comparative analysis. 

This project enables YouTube content creators to easily monitor and evaluate their channel's performance relative to their peers, allowing for more informed decision-making and strategic planning.

<img src="images/powerbi_comments.PNG" alt="PowerBI Comments" style="width:80%;">

### [Project 2: Rental Price Prediction](https://github.com/JensBender/rental-price-prediction)
[![Python][Python-badge]][Python-url] [![NumPy][NumPy-badge]][NumPy-url] [![Pandas][Pandas-badge]][Pandas-url] [![Matplotlib][Matplotlib-badge]][Matplotlib-url] [![scikit-learn][scikit-learn-badge]][scikit-learn-url] [![Flask][Flask-badge]][Flask-url] [![Docker][Docker-badge]][Docker-url]  
- **Motivation**: Simplify the process of finding rental properties in Singapore's expensive real estate market by using machine learning to estimate rental prices. 
- **Data Collection**: Scraped 1680 property listings from an online property portal, including information on price, size, address, bedrooms, bathrooms and more.
- **Exploratory Data Analysis**: Visualized property locations on an interactive map, generated a word cloud to extract insights from property agent descriptions, and examined descriptive statistics, distributions, and correlations.  
- **Data Preprocessing**: Handled missing address data and engineered location-related features using the Google Maps API, extracted property features from agent descriptions and systematically evaluated multiple outlier handling methods. 
- **Model Training**: Trained five machine learning models with baseline configurations, selected an XGBoost regression model with optimized hyperparameters, and achieved a test dataset performance with an RMSE of 995, a MAPE of 0.13, and an R² of 0.90.
- **Model Deployment**: Created a web application for serving the XGBoost model using the Flask framework. Containerized this application using Docker and successfully deployed the Docker container on render.com.

<div style="display: flex;">
  <img src="images/map.png" style="width: 49%;"> 
  <img src="images/feature_importance.png" style="width: 49%;">
</div>

### [Project 3: Hate Speech Detection](https://github.com/JensBender/hate-speech-detection)
[![Python][Python-badge]][Python-url] [![TensorFlow][TensorFlow-badge]][TensorFlow-url] [![scikit-learn][scikit-learn-badge]][scikit-learn-url] [![NumPy][NumPy-badge]][NumPy-url] [![Pandas][Pandas-badge]][Pandas-url] [![Matplotlib][Matplotlib-badge]][Matplotlib-url] [![Flask][Flask-badge]][Flask-url]  
- **Motivation**: Develop a hate speech detector for social media comments. 
- **Data**: Utilized the [ETHOS Hate Speech Detection Dataset](https://github.com/intelligence-csd-auth-gr/Ethos-Hate-Speech-Dataset).
- **Models**: Trained and evaluated the performance of three deep learning models using TensorFlow and scikit-learn. The fine-tuned BERT model demonstrated superior performance (78.0% accuracy) compared to the SimpleRNN (66.3%) and LSTM (70.7%) models.  
- **Deployment**: Prepared the fine-tuned BERT model for production by integrating it into a web application and an API endpoint using the Flask web framework.

| Fine-tuned BERT: Confusion Matrix | Model Deployment | 
| ------------------ | ------------------ | 
| ![BERT-confusion-matrix](images/bert_confusion_matrix.png) | <img src="images/hate_speech_model_deployment.PNG" style="width: 275px;"> |


<!-- COURSE CERTIFICATES -->
## 🏅 Course Certificates

**Advanced SQL: MySQL for Ecommerce & Web Analytics**, Udemy, February 2024, [🔗 see certificate](https://www.udemy.com/certificate/UC-ac04dd78-4589-4b2e-a863-7722cd78ec2f/)  
Skills: MySQL · SQL

**AWS Certified Cloud Practitioner**, AWS, January 2024, [🔗 see certificate](https://www.credly.com/badges/3287f8a9-0dcd-48d2-afc3-c255faf027bc/public_url)  
Skills: Amazon Web Services (AWS) 

**Ultimate AWS Certified Cloud Practitioner CLF-C02**, Udemy, January 2024, [🔗 see certificate](https://www.udemy.com/certificate/UC-2090637d-9845-42f3-9f7b-97195874331a/)  
Skills: Amazon Web Services (AWS) 

**Spark and Python for Big Data with PySpark**, Udemy, January 2024, [🔗 see certificate](https://www.udemy.com/certificate/UC-27da6f52-bc5f-4e72-bc5b-c2cd488566b0/)  
Skills: Spark · PySpark · AWS · Python · Machine Learning · Linear Regression · Logistic Regression · Decision Trees · Random Forest · Gradient Boosting · k-means clustering · Recommender Systems · Natural Language Processing (NLP) 

**Microsoft Power BI Data Analyst**, Udemy, November 2023, [🔗 see certificate](https://www.udemy.com/certificate/UC-eb56c820-8c91-4e03-8c57-efdc8c570c6b/)  
Skills: Power BI

**Deep Learning**, alfatraining Bildungszentrum GmbH, April 2023  
Skills: TensorFlow · NumPy · Natural Language Processing (NLP) · Python · Deep Learning · Recurrent Neural Networks (RNN) · Neural Networks · Scikit-Learn · Reinforcement Learning · Transfer Learning · Convolutional Neural Networks (CNN) · Time Series Analysis

**Machine Learning by Stanford University & DeepLearning.AI**, Coursera, April 2023, [🔗 see certificate](https://coursera.org/share/1c62950a6100b0426d454b652e77498c)  
Skills: Decision Trees · Recommender Systems · Anomaly Detection · Python · Linear Regression · Neural Networks · Logistic Regression · Reinforcement Learning · Principal Component Analysis · k-means clustering

**Python for Machine Learning & Data Science Masterclass**, Udemy, March 2023, [🔗 see certificate](https://www.udemy.com/certificate/UC-4de79ac0-2282-45c9-93e1-a7cb6f812592/)  
Skills: Decision Trees · Support Vector Machine (SVM) · Matplotlib · Random Forest · Naive Bayes · NumPy · Seaborn · Hierarchical Clustering · Natural Language Processing (NLP) · Pandas · Python · Linear Regression · Scikit-Learn · Logistic Regression · Principal Component Analysis · Gradient Boosting · DBSCAN · k-means clustering · K-Nearest Neighbors (KNN)

**Machine Learning**, alfatraining Bildungszentrum GmbH, February 2023  
Skills: Decision Trees · Support Vector Machine (SVM) · Matplotlib · Naive Bayes · NumPy · Hierarchical Clustering · Pandas · Python · Linear Regression · Neural Networks · Scikit-Learn · Principal Component Analysis · DBSCAN · k-means clustering · K-Nearest Neighbors (KNN)

**The Ultimate MySQL Bootcamp: Go from SQL Beginner to Expert**, Udemy, December 2022, [🔗 see certificate](https://www.udemy.com/certificate/UC-e324e4f7-95ba-4894-b8e0-65229ff5e2dc)  
Skills: MySQL · SQL


<!-- GITHUB STATISTICS -->
## 👨‍💻 GitHub Statistics
[![Top Languages](https://github-readme-stats.vercel.app/api/top-langs/?username=JensBender&layout=compact)](https://github.com/JensBender)


<!-- CREDITS -->
## ©️ Credits
<sup><small>Profile banner GIF based on the video by [RDNE Stock project](https://www.pexels.com/video/business-analytics-presentation-7947451/) from Pexels</small></sup>


<!-- MARKDOWN LINKS -->
[Airflow-badge]: https://img.shields.io/badge/Apache%20Airflow-017CEE?style=for-the-badge&logo=Apache%20Airflow&logoColor=white
[Airflow-url]: https://airflow.apache.org/
[AWS-badge]: https://img.shields.io/badge/AWS-%23FF9900.svg?style=for-the-badge&logo=data:image/svg+xml;base64,PCFET0NUWVBFIHN2ZyBQVUJMSUMgIi0vL1czQy8vRFREIFNWRyAxLjEvL0VOIiAiaHR0cDovL3d3dy53My5vcmcvR3JhcGhpY3MvU1ZHLzEuMS9EVEQvc3ZnMTEuZHRkIj4KCjwhLS0gVXBsb2FkZWQgdG86IFNWRyBSZXBvLCB3d3cuc3ZncmVwby5jb20sIFRyYW5zZm9ybWVkIGJ5OiBTVkcgUmVwbyBNaXhlciBUb29scyAtLT4KPHN2ZyB3aWR0aD0iNjRweCIgaGVpZ2h0PSI2NHB4IiB2aWV3Qm94PSIwIDAgMTYgMTYiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgZmlsbD0ibm9uZSI+Cgo8ZyBpZD0iU1ZHUmVwb19iZ0NhcnJpZXIiIHN0cm9rZS13aWR0aD0iMCIvPgoKPGcgaWQ9IlNWR1JlcG9fdHJhY2VyQ2FycmllciIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIi8+Cgo8ZyBpZD0iU1ZHUmVwb19pY29uQ2FycmllciI+IDxnIGZpbGw9IiNmZmZmZmYiPiA8cGF0aCBkPSJNNC41MSA3LjY4N2MwIC4xOTcuMDIuMzU3LjA1OC40NzUuMDQyLjExNy4wOTYuMjQ1LjE3LjM4NGEuMjMzLjIzMyAwIDAxLjAzNy4xMjNjMCAuMDUzLS4wMzIuMTA3LS4xLjE2bC0uMzM2LjIyNGEuMjU1LjI1NSAwIDAxLS4xMzguMDQ4Yy0uMDU0IDAtLjEwNy0uMDI2LS4xNi0uMDc0YTEuNjUyIDEuNjUyIDAgMDEtLjE5Mi0uMjUxIDQuMTM3IDQuMTM3IDAgMDEtLjE2NC0uMzE1Yy0uNDE2LjQ5MS0uOTM3LjczNy0xLjU2NS43MzctLjQ0NyAwLS44MDQtLjEyOS0xLjA2NC0uMzg1LS4yNjEtLjI1Ni0uMzk0LS41OTgtLjM5NC0xLjAyNSAwLS40NTQuMTYtLjgyMi40ODQtMS4xLjMyNS0uMjc4Ljc1Ni0uNDE2IDEuMzA0LS40MTYuMTggMCAuMzY3LjAxNi41NjQuMDQyLjE5Ny4wMjcuNC4wNy42MTIuMTE4di0uMzljMC0uNDA2LS4wODUtLjY4OS0uMjUtLjg1NC0uMTctLjE2Ni0uNDU4LS4yNDYtLjg2OC0uMjQ2LS4xODYgMC0uMzc3LjAyMi0uNTc0LjA3YTQuMjMgNC4yMyAwIDAwLS41NzUuMTgxIDEuNTI1IDEuNTI1IDAgMDEtLjE4Ni4wNy4zMjYuMzI2IDAgMDEtLjA4NS4wMTZjLS4wNzUgMC0uMTEyLS4wNTQtLjExMi0uMTY2di0uMjYyYzAtLjA4NS4wMS0uMTUuMDM3LS4xODZhLjM5OS4zOTkgMCAwMS4xNS0uMTEzYy4xODUtLjA5Ni40MDktLjE3Ni42Ny0uMjQuMjYtLjA3LjUzNy0uMTAxLjgzLS4xMDEuNjMzIDAgMS4wOTYuMTQ0IDEuMzk0LjQzMi4yOTMuMjg4LjQ0Mi43MjYuNDQyIDEuMzE0djEuNzNoLjAxem0tMi4xNjEuODExYy4xNzUgMCAuMzU2LS4wMzIuNTQ4LS4wOTYuMTkyLS4wNjQuMzYyLS4xODIuNTA1LS4zNDJhLjg0OC44NDggMCAwMC4xODEtLjM0MWMuMDMyLS4xMjkuMDU0LS4yODMuMDU0LS40NjVWNy4wM2E0LjQzIDQuNDMgMCAwMC0uNDktLjA5IDMuOTk2IDMuOTk2IDAgMDAtLjUtLjAzM2MtLjM1NyAwLS42MTcuMDctLjc5My4yMTQtLjE3Ni4xNDQtLjI2LjM0Ny0uMjYuNjE0IDAgLjI1LjA2My40MzcuMTk2LjU2Ni4xMjguMTMzLjMxNC4xOTcuNTU5LjE5N3ptNC4yNzMuNTc3Yy0uMDk2IDAtLjE2LS4wMTYtLjIwMi0uMDU0LS4wNDMtLjAzMi0uMDgtLjEwNi0uMTEyLS4yMDhsLTEuMjUtNC4xMjdhLjkzOC45MzggMCAwMS0uMDQ4LS4yMTRjMC0uMDg1LjA0Mi0uMTMzLjEyNy0uMTMzaC41MjJjLjEgMCAuMTcuMDE2LjIwNy4wNTMuMDQzLjAzMi4wNzUuMTA3LjEwNy4yMDhsLjg5NCAzLjUzNS44My0zLjUzNWMuMDI2LS4xMDYuMDU4LS4xNzYuMTAxLS4yMDhhLjM2NS4zNjUgMCAwMS4yMTMtLjA1M2guNDI2Yy4xIDAgLjE3LjAxNi4yMTIuMDUzLjA0My4wMzIuMDguMTA3LjEwMi4yMDhsLjg0IDMuNTc4LjkyLTMuNTc4YS40NTkuNDU5IDAgMDEuMTA3LS4yMDguMzQ3LjM0NyAwIDAxLjIwOC0uMDUzaC40OTVjLjA4NSAwIC4xMzMuMDQzLjEzMy4xMzMgMCAuMDI3LS4wMDYuMDU0LS4wMS4wODZhLjc2OC43NjggMCAwMS0uMDM4LjEzM2wtMS4yODMgNC4xMjdjLS4wMzEuMTA3LS4wNjkuMTc3LS4xMTEuMjA5YS4zNC4zNCAwIDAxLS4yMDMuMDUzaC0uNDU3Yy0uMTAxIDAtLjE3LS4wMTYtLjIxMy0uMDUzLS4wNDMtLjAzOC0uMDgtLjEwNy0uMTAxLS4yMTRMOC4yMTMgNS4zN2wtLjgyIDMuNDM5Yy0uMDI2LjEwNy0uMDU4LjE3Ni0uMS4yMTMtLjA0My4wMzgtLjExOC4wNTQtLjIxMy4wNTRoLS40NTh6bTYuODM4LjE0NGEzLjUxIDMuNTEgMCAwMS0uODItLjA5NmMtLjI2Ni0uMDY0LS40NzMtLjEzNC0uNjEyLS4yMTQtLjA4NS0uMDQ4LS4xNDMtLjEwMS0uMTY1LS4xNWEuMzguMzggMCAwMS0uMDMxLS4xNDl2LS4yNzJjMC0uMTEyLjA0Mi0uMTY2LjEyMi0uMTY2YS4zLjMgMCAwMS4wOTYuMDE2Yy4wMzIuMDExLjA4LjAzMi4xMzMuMDU0LjE4LjA4LjM3OC4xNDQuNTg1LjE4Ny4yMTMuMDQyLjQyLjA2NC42MzMuMDY0LjMzNiAwIC41OTYtLjA1OS43NzctLjE3NmEuNTc1LjU3NSAwIDAwLjI3Ny0uNTA4LjUyLjUyIDAgMDAtLjE0NC0uMzczYy0uMDk1LS4xMDItLjI3Ni0uMTkzLS41MzctLjI3OGwtLjc3Mi0uMjRjLS4zODgtLjEyMy0uNjc2LS4zMDUtLjg1MS0uNTQ1YTEuMjc1IDEuMjc1IDAgMDEtLjI2Ni0uNzc0YzAtLjIyNC4wNDgtLjQyMi4xNDMtLjU5My4wOTYtLjE3LjIyNC0uMzIuMzg0LS40MzguMTYtLjEyMi4zNC0uMjEzLjU1My0uMjc3LjIxMy0uMDY0LjQzNi0uMDkxLjY3LS4wOTEuMTE4IDAgLjI0LjAwNS4zNTcuMDIxLjEyMi4wMTYuMjM0LjAzOC4zNDYuMDYuMTA2LjAyNi4yMDguMDUyLjMwMy4wODUuMDk2LjAzMi4xNy4wNjQuMjI0LjA5NmEuNDYxLjQ2MSAwIDAxLjE2LjEzMy4yODkuMjg5IDAgMDEuMDQ3LjE3NnYuMjUxYzAgLjExMi0uMDQyLjE3MS0uMTIyLjE3MWEuNTUyLjU1MiAwIDAxLS4yMDItLjA2NCAyLjQyOCAyLjQyOCAwIDAwLTEuMDIyLS4yMDhjLS4zMDMgMC0uNTQzLjA0OC0uNzA4LjE1LS4xNjUuMS0uMjUuMjU2LS4yNS40NzUgMCAuMTQ5LjA1My4yNzcuMTYuMzc5LjEwNi4xMDEuMzAzLjIwMi41ODUuMjkzbC43NTYuMjRjLjM4My4xMjMuNjYuMjk0LjgyNS41MTMuMTY1LjIxOS4yNDQuNDcuMjQ0Ljc0OCAwIC4yMy0uMDQ3LjQzNy0uMTM4LjYxOWExLjQzNSAxLjQzNSAwIDAxLS4zODguNDdjLS4xNjUuMTMzLS4zNjIuMjMtLjU5MS4yOTktLjI0LjA3NS0uNDkuMTEyLS43NjEuMTEyeiIvPiA8cGF0aCBmaWxsLXJ1bGU9ImV2ZW5vZGQiIGQ9Ik0xNC40NjUgMTEuODEzYy0xLjc1IDEuMjk3LTQuMjk0IDEuOTg2LTYuNDgxIDEuOTg2LTMuMDY1IDAtNS44MjctMS4xMzctNy45MTMtMy4wMjctLjE2NS0uMTUtLjAxNi0uMzUzLjE4LS4yMzUgMi4yNTcgMS4zMTMgNS4wNCAyLjEwOSA3LjkyIDIuMTA5IDEuOTQxIDAgNC4wNzUtLjQwNiA2LjAzOS0xLjIzOS4yOTMtLjEzMy41NDMuMTkyLjI1NS40MDZ6IiBjbGlwLXJ1bGU9ImV2ZW5vZGQiLz4gPHBhdGggZmlsbC1ydWxlPSJldmVub2RkIiBkPSJNMTUuMTk0IDEwLjk4Yy0uMjIzLS4yODctMS40NzktLjEzOC0yLjA0OC0uMDY5LS4xNy4wMjItLjE5Ny0uMTI4LS4wNDMtLjI0IDEtLjcwNSAyLjY0NS0uNTAyIDIuODM2LS4yNjcuMTkyLjI0LS4wNTMgMS44OS0uOTkgMi42OC0uMTQzLjEyMy0uMjgxLjA2LS4yMTctLjEuMjEyLS41My42ODYtMS43Mi40NjItMi4wMDN6IiBjbGlwLXJ1bGU9ImV2ZW5vZGQiLz4gPC9nPiA8L2c+Cgo8L3N2Zz4=
[AWS-url]: https://aws.amazon.com/
[ChatGPT-badge]: https://img.shields.io/badge/chatGPT-74aa9c?style=for-the-badge&logo=openai&logoColor=white
[ChatGPT-url]: https://chatgpt.com/
[Cursor-badge]: https://img.shields.io/badge/Cursor-000000?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciICB2aWV3Qm94PSIwIDAgNDggNDgiIHdpZHRoPSI0OHB4IiBoZWlnaHQ9IjQ4cHgiIGZpbGwtcnVsZT0iZXZlbm9kZCIgY2xpcC1ydWxlPSJldmVub2RkIiBiYXNlUHJvZmlsZT0iYmFzaWMiPjxwb2x5Z29uIGZpbGw9IiNiY2JjYmMiIHBvaW50cz0iMjMuOTc0LDQgNi45NywxNCA2Ljk3LDM0IDIzLjk5OCw0NCA0MC45NywzNCA0MC45NywxNCIvPjxsaW5lIHgxPSI3Ljk3IiB4Mj0iMjMuNTc5IiB5MT0iMzMiIHkyPSIyNC40NTQiIGZpbGw9Im5vbmUiIHN0cm9rZT0iI2JjYmNiYyIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIiBzdHJva2UtbWl0ZXJsaW1pdD0iMTAiIHN0cm9rZS13aWR0aD0iMiIvPjxsaW5lIHgxPSIyMy45NzIiIHgyPSIyMy45NjYiIHkxPSI1LjkwMyIgeTI9IjE1Ljg2NCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjYmNiY2JjIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiIHN0cm9rZS1taXRlcmxpbWl0PSIxMCIgc3Ryb2tlLXdpZHRoPSIyIi8+PGxpbmUgeDE9IjM5Ljk3IiB4Mj0iMzIuOTciIHkxPSIzMyIgeTI9IjI5IiBmaWxsPSJub25lIiBzdHJva2U9IiNiY2JjYmMiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIgc3Ryb2tlLW1pdGVybGltaXQ9IjEwIiBzdHJva2Utd2lkdGg9IjIiLz48cG9seWdvbiBmaWxsPSIjNzU3NTc1IiBwb2ludHM9IjIzLjk3NCw0IDYuOTcsMTQgNi45NywzNCAyMy45NywyNCIvPjxwb2x5Z29uIGZpbGw9IiM0MjQyNDIiIHBvaW50cz0iMjMuOTgxLDE0IDQwLjk3LDE0IDQwLjk3LDM0IDIzLjk3MSwyNCIvPjxwb2x5Z29uIGZpbGw9IiM2MTYxNjEiIGZpbGwtcnVsZT0iZXZlbm9kZCIgcG9pbnRzPSI0MC45NywxNCAyMy45NjYsMTcgMjMuOTc0LDQiIGNsaXAtcnVsZT0iZXZlbm9kZCIvPjxwb2x5Z29uIGZpbGw9IiM2MTYxNjEiIGZpbGwtcnVsZT0iZXZlbm9kZCIgcG9pbnRzPSI2Ljk3LDE0IDIzLjk4MSwxNi44ODEgMjMuOTY2LDI0IDYuOTcsMzQiIGNsaXAtcnVsZT0iZXZlbm9kZCIvPjxwb2x5Z29uIGZpbGw9IiNlZGVkZWQiIHBvaW50cz0iNi45NywxNCAyMy45NywyNCAyMy45OTgsNDQgNDAuOTcsMTQiLz48L3N2Zz4=
[Cursor-url]: https://www.cursor.com/
[Docker-badge]: https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white
[Docker-url]: https://www.docker.com/
[FastAPI-badge]: https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white
[FastAPI-url]: https://fastapi.tiangolo.com/
[Flask-badge]: https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white
[Flask-url]: https://flask.palletsprojects.com/en/2.3.x/
[Git-badge]: https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white
[Git-url]: https://git-scm.com/
[GitHub-badge]: https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white
[GitHub-url]: https://github.com/
[Gradio-badge]: https://img.shields.io/badge/Gradio-fc7404?style=for-the-badge&logo=gradio&logoColor=white
[Gradio-url]: https://gradio.app
[HuggingFace-badge]: https://img.shields.io/badge/Hugging%20Face-ffcc00?style=for-the-badge&logo=huggingface&logoColor=black
[HuggingFace-url]: https://huggingface.co/
[JupyterNotebook-badge]: https://img.shields.io/badge/Jupyter-F37626.svg?style=for-the-badge&logo=Jupyter&logoColor=white
[JupyterNotebook-url]: https://jupyter.org/
[Matplotlib-badge]: https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxODAiIGhlaWdodD0iMTgwIiBzdHJva2U9ImdyYXkiPgo8ZyBzdHJva2Utd2lkdGg9IjIiIGZpbGw9IiNGRkYiPgo8Y2lyY2xlIGN4PSI5MCIgY3k9IjkwIiByPSI4OCIvPgo8Y2lyY2xlIGN4PSI5MCIgY3k9IjkwIiByPSI2NiIvPgo8Y2lyY2xlIGN4PSI5MCIgY3k9IjkwIiByPSI0NCIvPgo8Y2lyY2xlIGN4PSI5MCIgY3k9IjkwIiByPSIyMiIvPgo8cGF0aCBkPSJtOTAsMnYxNzZtNjItMjYtMTI0LTEyNG0xMjQsMC0xMjQsMTI0bTE1MC02MkgyIi8+CjwvZz48ZyBvcGFjaXR5PSIuOCI+CjxwYXRoIGZpbGw9IiM0NEMiIGQ9Im05MCw5MGgxOGExOCwxOCAwIDAsMCAwLTV6Ii8+CjxwYXRoIGZpbGw9IiNCQzMiIGQ9Im05MCw5MCAzNC00M2E1NSw1NSAwIDAsMC0xNS04eiIvPgo8cGF0aCBmaWxsPSIjRDkzIiBkPSJtOTAsOTAtMTYtNzJhNzQsNzQgMCAwLDAtMzEsMTV6Ii8+CjxwYXRoIGZpbGw9IiNEQjMiIGQ9Im05MCw5MC01OC0yOGE2NSw2NSAwIDAsMC01LDM5eiIvPgo8cGF0aCBmaWxsPSIjM0JCIiBkPSJtOTAsOTAtMzMsMTZhMzcsMzcgMCAwLDAgMiw1eiIvPgo8cGF0aCBmaWxsPSIjM0M5IiBkPSJtOTAsOTAtMTAsNDVhNDYsNDYgMCAwLDAgMTgsMHoiLz4KPHBhdGggZmlsbD0iI0Q3MyIgZD0ibTkwLDkwIDQ2LDU4YTc0LDc0IDAgMCwwIDEyLTEyeiIvPgo8L2c+PC9zdmc+
[Matplotlib-url]: https://matplotlib.org/
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
[PyCharm-badge]: https://img.shields.io/badge/pycharm-143?style=for-the-badge&logo=pycharm&logoColor=black&color=black&labelColor=green
[PyCharm-url]: https://www.jetbrains.com/pycharm/
[Pydantic-badge]: https://img.shields.io/badge/Pydantic-3776AB?style=for-the-badge&logo=pydantic&logoColor=white
[Pydantic-url]: https://docs.pydantic.dev/
[Python-badge]: https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[Pytest-badge]: https://img.shields.io/badge/pytest-%23ffffff.svg?style=for-the-badge&logo=pytest&logoColor=2f9fe3
[Pytest-url]: https://docs.pytest.org/
[Python-url]: https://www.python.org/
[scikit-learn-badge]: https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white
[scikit-learn-url]: https://scikit-learn.org/stable/
[Seaborn-badge]: https://img.shields.io/badge/seaborn-%230C4A89.svg?style=for-the-badge&logo=seaborn&logoColor=white
[Seaborn-url]: https://seaborn.pydata.org/
[Selenium-badge]: https://img.shields.io/badge/selenium-%43B02A.svg?style=for-the-badge&logo=selenium&logoColor=white
[Selenium-url]: https://www.selenium.dev/
[Spark-badge]: https://img.shields.io/badge/Apache%20Spark-E25A1C.svg?style=for-the-badge&logo=Apache-Spark&logoColor=white
[Spark-url]: https://spark.apache.org/
[TensorFlow-badge]: https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white
[TensorFlow-url]: https://www.tensorflow.org/
[VSCode-badge]: https://img.shields.io/badge/VS%20Code-0078D4?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMTAwIDEwMCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPG1hc2sgaWQ9Im1hc2swIiBtYXNrLXR5cGU9ImFscGhhIiBtYXNrVW5pdHM9InVzZXJTcGFjZU9uVXNlIiB4PSIwIiB5PSIwIiB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEwMCI+CjxwYXRoIGZpbGwtcnVsZT0iZXZlbm9kZCIgY2xpcC1ydWxlPSJldmVub2RkIiBkPSJNNzAuOTExOSA5OS4zMTcxQzcyLjQ4NjkgOTkuOTMwNyA3NC4yODI4IDk5Ljg5MTQgNzUuODcyNSA5OS4xMjY0TDk2LjQ2MDggODkuMjE5N0M5OC42MjQyIDg4LjE3ODcgMTAwIDg1Ljk4OTIgMTAwIDgzLjU4NzJWMTYuNDEzM0MxMDAgMTQuMDExMyA5OC42MjQzIDExLjgyMTggOTYuNDYwOSAxMC43ODA4TDc1Ljg3MjUgMC44NzM3NTZDNzMuNzg2MiAtMC4xMzAxMjkgNzEuMzQ0NiAwLjExNTc2IDY5LjUxMzUgMS40NDY5NUM2OS4yNTIgMS42MzcxMSA2OS4wMDI4IDEuODQ5NDMgNjguNzY5IDIuMDgzNDFMMjkuMzU1MSAzOC4wNDE1TDEyLjE4NzIgMjUuMDA5NkMxMC41ODkgMjMuNzk2NSA4LjM1MzYzIDIzLjg5NTkgNi44NjkzMyAyNS4yNDYxTDEuMzYzMDMgMzAuMjU0OUMtMC40NTI1NTIgMzEuOTA2NCAtMC40NTQ2MzMgMzQuNzYyNyAxLjM1ODUzIDM2LjQxN0wxNi4yNDcxIDUwLjAwMDFMMS4zNTg1MyA2My41ODMyQy0wLjQ1NDYzMyA2NS4yMzc0IC0wLjQ1MjU1MiA2OC4wOTM4IDEuMzYzMDMgNjkuNzQ1M0w2Ljg2OTMzIDc0Ljc1NDFDOC4zNTM2MyA3Ni4xMDQzIDEwLjU4OSA3Ni4yMDM3IDEyLjE4NzIgNzQuOTkwNUwyOS4zNTUxIDYxLjk1ODdMNjguNzY5IDk3LjkxNjdDNjkuMzkyNSA5OC41NDA2IDcwLjEyNDYgOTkuMDEwNCA3MC45MTE5IDk5LjMxNzFaTTc1LjAxNTIgMjcuMjk4OUw0NS4xMDkxIDUwLjAwMDFMNzUuMDE1MiA3Mi43MDEyVjI3LjI5ODlaIiBmaWxsPSJ3aGl0ZSIvPgo8L21hc2s+CjxnIG1hc2s9InVybCgjbWFzazApIj4KPHBhdGggZD0iTTk2LjQ2MTQgMTAuNzk2Mkw3NS44NTY5IDAuODc1NTQyQzczLjQ3MTkgLTAuMjcyNzczIDcwLjYyMTcgMC4yMTE2MTEgNjguNzUgMi4wODMzM0wxLjI5ODU4IDYzLjU4MzJDLTAuNTE1NjkzIDY1LjIzNzMgLTAuNTEzNjA3IDY4LjA5MzcgMS4zMDMwOCA2OS43NDUyTDYuODEyNzIgNzQuNzU0QzguMjk3OTMgNzYuMTA0MiAxMC41MzQ3IDc2LjIwMzYgMTIuMTMzOCA3NC45OTA1TDkzLjM2MDkgMTMuMzY5OUM5Ni4wODYgMTEuMzAyNiAxMDAgMTMuMjQ2MiAxMDAgMTYuNjY2N1YxNi40Mjc1QzEwMCAxNC4wMjY1IDk4LjYyNDYgMTEuODM3OCA5Ni40NjE0IDEwLjc5NjJaIiBmaWxsPSIjMDA2NUE5Ii8+CjxnIGZpbHRlcj0idXJsKCNmaWx0ZXIwX2QpIj4KPHBhdGggZD0iTTk2LjQ2MTQgODkuMjAzOEw3NS44NTY5IDk5LjEyNDVDNzMuNDcxOSAxMDAuMjczIDcwLjYyMTcgOTkuNzg4NCA2OC43NSA5Ny45MTY3TDEuMjk4NTggMzYuNDE2OUMtMC41MTU2OTMgMzQuNzYyNyAtMC41MTM2MDcgMzEuOTA2MyAxLjMwMzA4IDMwLjI1NDhMNi44MTI3MiAyNS4yNDZDOC4yOTc5MyAyMy44OTU4IDEwLjUzNDcgMjMuNzk2NCAxMi4xMzM4IDI1LjAwOTVMOTMuMzYwOSA4Ni42MzAxQzk2LjA4NiA4OC42OTc0IDEwMCA4Ni43NTM4IDEwMCA4My4zMzM0VjgzLjU3MjZDMTAwIDg1Ljk3MzUgOTguNjI0NiA4OC4xNjIyIDk2LjQ2MTQgODkuMjAzOFoiIGZpbGw9IiMwMDdBQ0MiLz4KPC9nPgo8ZyBmaWx0ZXI9InVybCgjZmlsdGVyMV9kKSI+CjxwYXRoIGQ9Ik03NS44NTc4IDk5LjEyNjNDNzMuNDcyMSAxMDAuMjc0IDcwLjYyMTkgOTkuNzg4NSA2OC43NSA5Ny45MTY2QzcxLjA1NjQgMTAwLjIyMyA3NSA5OC41ODk1IDc1IDk1LjMyNzhWNC42NzIxM0M3NSAxLjQxMDM5IDcxLjA1NjQgLTAuMjIzMTA2IDY4Ljc1IDIuMDgzMjlDNzAuNjIxOSAwLjIxMTQwMiA3My40NzIxIC0wLjI3MzY2NiA3NS44NTc4IDAuODczNjMzTDk2LjQ1ODcgMTAuNzgwN0M5OC42MjM0IDExLjgyMTcgMTAwIDE0LjAxMTIgMTAwIDE2LjQxMzJWODMuNTg3MUMxMDAgODUuOTg5MSA5OC42MjM0IDg4LjE3ODYgOTYuNDU4NiA4OS4yMTk2TDc1Ljg1NzggOTkuMTI2M1oiIGZpbGw9IiMxRjlDRjAiLz4KPC9nPgo8ZyBzdHlsZT0ibWl4LWJsZW5kLW1vZGU6b3ZlcmxheSIgb3BhY2l0eT0iMC4yNSI+CjxwYXRoIGZpbGwtcnVsZT0iZXZlbm9kZCIgY2xpcC1ydWxlPSJldmVub2RkIiBkPSJNNzAuODUxMSA5OS4zMTcxQzcyLjQyNjEgOTkuOTMwNiA3NC4yMjIxIDk5Ljg5MTMgNzUuODExNyA5OS4xMjY0TDk2LjQgODkuMjE5N0M5OC41NjM0IDg4LjE3ODcgOTkuOTM5MiA4NS45ODkyIDk5LjkzOTIgODMuNTg3MVYxNi40MTMzQzk5LjkzOTIgMTQuMDExMiA5OC41NjM1IDExLjgyMTcgOTYuNDAwMSAxMC43ODA3TDc1LjgxMTcgMC44NzM2OTVDNzMuNzI1NSAtMC4xMzAxOSA3MS4yODM4IDAuMTE1Njk5IDY5LjQ1MjcgMS40NDY4OEM2OS4xOTEyIDEuNjM3MDUgNjguOTQyIDEuODQ5MzcgNjguNzA4MiAyLjA4MzM1TDI5LjI5NDMgMzguMDQxNEwxMi4xMjY0IDI1LjAwOTZDMTAuNTI4MyAyMy43OTY0IDguMjkyODUgMjMuODk1OSA2LjgwODU1IDI1LjI0NkwxLjMwMjI1IDMwLjI1NDhDLTAuNTEzMzM0IDMxLjkwNjQgLTAuNTE1NDE1IDM0Ljc2MjcgMS4yOTc3NSAzNi40MTY5TDE2LjE4NjMgNTBMMS4yOTc3NSA2My41ODMyQy0wLjUxNTQxNSA2NS4yMzc0IC0wLjUxMzMzNCA2OC4wOTM3IDEuMzAyMjUgNjkuNzQ1Mkw2LjgwODU1IDc0Ljc1NEM4LjI5Mjg1IDc2LjEwNDIgMTAuNTI4MyA3Ni4yMDM2IDEyLjEyNjQgNzQuOTkwNUwyOS4yOTQzIDYxLjk1ODZMNjguNzA4MiA5Ny45MTY3QzY5LjMzMTcgOTguNTQwNSA3MC4wNjM4IDk5LjAxMDQgNzAuODUxMSA5OS4zMTcxWk03NC45NTQ0IDI3LjI5ODlMNDUuMDQ4MyA1MEw3NC45NTQ0IDcyLjcwMTJWMjcuMjk4OVoiIGZpbGw9InVybCgjcGFpbnQwX2xpbmVhcikiLz4KPC9nPgo8L2c+CjxkZWZzPgo8ZmlsdGVyIGlkPSJmaWx0ZXIwX2QiIHg9Ii04LjM5NDExIiB5PSIxNS44MjkxIiB3aWR0aD0iMTE2LjcyNyIgaGVpZ2h0PSI5Mi4yNDU2IiBmaWx0ZXJVbml0cz0idXNlclNwYWNlT25Vc2UiIGNvbG9yLWludGVycG9sYXRpb24tZmlsdGVycz0ic1JHQiI+CjxmZUZsb29kIGZsb29kLW9wYWNpdHk9IjAiIHJlc3VsdD0iQmFja2dyb3VuZEltYWdlRml4Ii8+CjxmZUNvbG9yTWF0cml4IGluPSJTb3VyY2VBbHBoYSIgdHlwZT0ibWF0cml4IiB2YWx1ZXM9IjAgMCAwIDAgMCAwIDAgMCAwIDAgMCAwIDAgMCAwIDAgMCAwIDEyNyAwIi8+CjxmZU9mZnNldC8+CjxmZUdhdXNzaWFuQmx1ciBzdGREZXZpYXRpb249IjQuMTY2NjciLz4KPGZlQ29sb3JNYXRyaXggdHlwZT0ibWF0cml4IiB2YWx1ZXM9IjAgMCAwIDAgMCAwIDAgMCAwIDAgMCAwIDAgMCAwIDAgMCAwIDAuMjUgMCIvPgo8ZmVCbGVuZCBtb2RlPSJvdmVybGF5IiBpbjI9IkJhY2tncm91bmRJbWFnZUZpeCIgcmVzdWx0PSJlZmZlY3QxX2Ryb3BTaGFkb3ciLz4KPGZlQmxlbmQgbW9kZT0ibm9ybWFsIiBpbj0iU291cmNlR3JhcGhpYyIgaW4yPSJlZmZlY3QxX2Ryb3BTaGFkb3ciIHJlc3VsdD0ic2hhcGUiLz4KPC9maWx0ZXI+CjxmaWx0ZXIgaWQ9ImZpbHRlcjFfZCIgeD0iNjAuNDE2NyIgeT0iLTguMDc1NTgiIHdpZHRoPSI0Ny45MTY3IiBoZWlnaHQ9IjExNi4xNTEiIGZpbHRlclVuaXRzPSJ1c2VyU3BhY2VPblVzZSIgY29sb3ItaW50ZXJwb2xhdGlvbi1maWx0ZXJzPSJzUkdCIj4KPGZlRmxvb2QgZmxvb2Qtb3BhY2l0eT0iMCIgcmVzdWx0PSJCYWNrZ3JvdW5kSW1hZ2VGaXgiLz4KPGZlQ29sb3JNYXRyaXggaW49IlNvdXJjZUFscGhhIiB0eXBlPSJtYXRyaXgiIHZhbHVlcz0iMCAwIDAgMCAwIDAgMCAwIDAgMCAwIDAgMCAwIDAgMCAwIDAgMTI3IDAiLz4KPGZlT2Zmc2V0Lz4KPGZlR2F1c3NpYW5CbHVyIHN0ZERldmlhdGlvbj0iNC4xNjY2NyIvPgo8ZmVDb2xvck1hdHJpeCB0eXBlPSJtYXRyaXgiIHZhbHVlcz0iMCAwIDAgMCAwIDAgMCAwIDAgMCAwIDAgMCAwIDAgMCAwIDAgMC4yNSAwIi8+CjxmZUJsZW5kIG1vZGU9Im92ZXJsYXkiIGluMj0iQmFja2dyb3VuZEltYWdlRml4IiByZXN1bHQ9ImVmZmVjdDFfZHJvcFNoYWRvdyIvPgo8ZmVCbGVuZCBtb2RlPSJub3JtYWwiIGluPSJTb3VyY2VHcmFwaGljIiBpbjI9ImVmZmVjdDFfZHJvcFNoYWRvdyIgcmVzdWx0PSJzaGFwZSIvPgo8L2ZpbHRlcj4KPGxpbmVhckdyYWRpZW50IGlkPSJwYWludDBfbGluZWFyIiB4MT0iNDkuOTM5MiIgeTE9IjAuMjU3ODEyIiB4Mj0iNDkuOTM5MiIgeTI9Ijk5Ljc0MjMiIGdyYWRpZW50VW5pdHM9InVzZXJTcGFjZU9uVXNlIj4KPHN0b3Agc3RvcC1jb2xvcj0id2hpdGUiLz4KPHN0b3Agb2Zmc2V0PSIxIiBzdG9wLWNvbG9yPSJ3aGl0ZSIgc3RvcC1vcGFjaXR5PSIwIi8+CjwvbGluZWFyR3JhZGllbnQ+CjwvZGVmcz4KPC9zdmc+Cg==
[VSCode-url]: https://code.visualstudio.com/
