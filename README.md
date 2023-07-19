![Profile-banner](images/profile-banner.gif)

# Hi there! I'm Jens

<!-- ABOUT ME -->
## 👋 About Me
I'm a passionate and analytical **data scientist** with over eight years of experience in data analysis, data visualization, and data storytelling. I enjoy solving challenging problems, harnessing the power of machine learning to derive valuable insights, and effectively communicating complex information.

<!-- SKILLS -->
## 🛠️ Skills

| Category           | Skill    |
| ------------------ | -------- |
| Programming        | [![Python][Python-badge]][Python-url] [![MySQL][MySQL-badge]][MySQL-url] |
| Data Manipulation  | [![NumPy][NumPy-badge]][NumPy-url] [![Pandas][Pandas-badge]][Pandas-url] |
| Data Visualization | [![Matplotlib][Matplotlib-badge]][Matplotlib-url] [![Plotly][Plotly-badge]][Plotly-url] |
| Machine Learning   | [![scikit-learn][scikit-learn-badge]][scikit-learn-url] [![TensorFlow][TensorFlow-badge]][TensorFlow-url] |
| Web Framework      | [![Flask][Flask-badge]][Flask-url] |
| Version Control    | [![Git][Git-badge]][Git-url] |
| Code Editors       | [![Jupyter Notebook][JupyterNotebook-badge]][JupyterNotebook-url] [![PyCharm][PyCharm-badge]][PyCharm-url] [![Spyder][Spyder-badge]][Spyder-url] [![Google Colab][GoogleColab-badge]][GoogleColab-url] |

<!-- PORTFOLIO -->
## 📄 Portfolio

### [Project 1: Hate Speech Detection](https://github.com/JensBender/hate-speech-detection)
+ Motivation: Develop a hate speech detector for social media comments. 
+ Data: Utilized the [ETHOS Hate Speech Detection Dataset](https://github.com/intelligence-csd-auth-gr/Ethos-Hate-Speech-Dataset).
+ Models: Trained and evaluated the performance of three deep learning models using TensorFlow and scikit-learn. The fine-tuned BERT model demonstrated superior performance (78.0% accuracy) compared to the SimpleRNN (66.3%) and LSTM (70.7%) models.  
+ Deployment: Prepared the fine-tuned BERT model for production by integrating it into a web application and an API endpoint using the Flask web framework.

| Fine-tuned BERT: Confusion Matrix | Model Deployment | 
| ------------------ | ------------------ | 
| ![BERT-confusion-matrix](images/bert_confusion_matrix.png) | <img src="images/hate_speech_model_deployment.PNG" style="width: 275px;"> |



### [Project 2: ChatGPT Cover Letter Generator](https://github.com/JensBender/chatgpt-cover-letter-generator)
+ Developed an AI-assisted cover letter generator that empowers job seekers in crafting personalized and professional cover letters tailored to specific job postings.
+ Employing Python and Beautiful Soup, job postings were scraped and the ChatGPT API was utilized to extract key information, including requirements and tasks, in JSON format.
+ By leveraging the ChatGPT API further, cover letter suggestions were generated, aligning the candidate's education, work experience, skills, and motivation with the specific job's requirements and tasks.

```
{
  "employer": "OpenAI",
  "job title": "Research Scientist",
  "requirements": [
    "Track record of coming up with new ideas or improving 
    upon existing ideas in machine learning",
    "Ability to own and pursue a research agenda",
    "Excitement about OpenAI's approach to research",
    "Nice to have: Interested in and thoughtful about the 
    impacts of AI technology",
    "Nice to have: Past experience in creating high-performance 
    implementations of deep learning algorithms"
  ],
  "tasks": [
    "Develop innovative machine learning techniques",
    "Advance the research agenda of the team",
    "Collaborate with peers across the organization"
  ],
  "contact person": "unknown",
  "address": "San Francisco, California, United States"
}
```

<!-- CREDITS -->
## ©️ Credits
<sup><small>Profile banner GIF based on the video by [RDNE Stock project](https://www.pexels.com/video/business-analytics-presentation-7947451/) from Pexels</small></sup>

<!-- MARKDOWN LINKS -->
[Flask-badge]: https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white
[Flask-url]: https://flask.palletsprojects.com/en/2.3.x/
[Git-badge]: https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white
[Git-url]: https://git-scm.com/
[GoogleColab-badge]: https://img.shields.io/badge/Google%20Colab-F9AB00.svg?style=for-the-badge&logo=Google-Colab&logoColor=white
[GoogleColab-url]: https://colab.research.google.com/
[JupyterNotebook-badge]: https://img.shields.io/badge/Jupyter-F37626.svg?style=for-the-badge&logo=Jupyter&logoColor=white
[JupyterNotebook-url]: https://jupyter.org/
[Matplotlib-badge]: https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black
[Matplotlib-url]: https://matplotlib.org/
[MySQL-badge]: https://img.shields.io/badge/mysql-%2300f.svg?style=for-the-badge&logo=mysql&logoColor=white
[MySQL-url]: https://www.mysql.com/
[NumPy-badge]: https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white
[NumPy-url]: https://numpy.org/
[Pandas-badge]: https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white
[Pandas-url]: https://pandas.pydata.org/
[Plotly-badge]: https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white
[Plotly-url]: https://plotly.com/python/
[PyCharm-badge]: https://img.shields.io/badge/pycharm-143?style=for-the-badge&logo=pycharm&logoColor=black&color=black&labelColor=green
[PyCharm-url]: https://www.jetbrains.com/pycharm/
[Python-badge]: https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[Python-url]: https://www.python.org/
[scikit-learn-badge]: https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white
[scikit-learn-url]: https://scikit-learn.org/stable/
[Spyder-badge]: https://img.shields.io/badge/Spyder-838485?style=for-the-badge&logo=spyder%20ide&logoColor=maroon
[Spyder-url]: https://www.spyder-ide.org/
[TensorFlow-badge]: https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white
[TensorFlow-url]: https://www.tensorflow.org/
