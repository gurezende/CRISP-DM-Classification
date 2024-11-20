# Data Science Project Management With CRISP-DM
---
## Project Description
This data science project uses the data mining lifecycle framework *CRISP-DM*. 

**C**ross **I**ndustry **S**tandard **P**rocess for **D**ata **M**ining is a six step framework to guide Project Managers through the data mining process. It is widely adopted by the Data Science field given it is intuitive and effective. In the framework, the data scientist should go through each step listed next unsing an iterative approach, knowing that this framework is flexible.

* (1) Business Understanding; 
* (2) Data Understanding; 
* (3) Data Preparation; 
* (4) Modeling; 
* (5) Evaluation;
* (6) Deployment 

At each step, due to new learning, the raise of new business questions or insights that come up, the professional can go back and forth on the process to redesign it or enhance the analysis.

## Medium Article

Find this project's complete description at this [Medium article](https://medium.com/towards-data-science/how-i-created-a-data-science-project-following-a-crisp-dm-lifecycle-8c0f5f89bba1?sk=f52e756c664f40ad267fd54b114ab901).


#### CRISP-DM Framework
![Framework](https://upload.wikimedia.org/wikipedia/commons/thumb/b/b9/CRISP-DM_Process_Diagram.png/330px-CRISP-DM_Process_Diagram.png)

## Business Problem
This project aims to create classifier to predict the probability of a customer to convert when offered a financial product (direct term deposit) via a phone call.

**Fixed-term**: investment where money is deposited into an account at a financial institution. Term deposits are also known as certificates of deposit (CDs) or time deposits. Term deposits typically have higher interest rates than traditional savings accounts, but the funds are not accessible until the term ends.

* The Dataset can be found in [this link, from UCI DS Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing)

---

## Python Version

This project was created using version **3.12.1**

---

## Modules
* uv >= 0.5.1
* catboost >= 1.2.5
* category-encoders >= 2.6.4
* feature-engine >= 1.8.2
* ipykernel >= 6.29.5
* matplotlib >= 3.9.2
* mkdocs >= 1.6.1
* numpy == 1.26.4
* pandas >= 2.2.3
* scikit-learn >= 1.5.2
* seaborn >= 0.13.2
* ucimlrepo >= 0.0.7
---
## Project Quick Start

The instructions to run this project are really simple. See them next.

<br>

#### Running on Jupyter Notebook

1. Install the modules and run the notebook.
    * Install modules using the command `pip install *module_name*`
2. Running the project on Jupyter nb does not require to install the module `uv`.

<br>

#### Running with IDE

1. Open a terminal
2. Install the module `uv` for virtual environment management
3. Run `pip install uv`
4. Run the command `uv init crispdm` (or use another name you like for the project)
5. Access the folder created using bash command such as `cd folder_name`
6. Set the Python version for your project with `pyenv local 3.12.1`
7. Create the virtual environment with `uv venv --python 3.12.1`
8. Install the python modules with 
    * `uv add catboost, category_encoders, feature_engine, matplotlib, seaborn, numpy, pandas, scikit-learn, ucimlrepo`
9. Run the code as a python file.

---

## Contact

This project was developed by **Gustavo R Santos**.<br>
>Data Scientist with 13+ years of experience specializing in data analysis, machine learning, and visualization using Python, R, SQL, and PySpark. Author of *Data Wrangling with R* and instructor of a PySpark course, with a passion for sharing knowledge through blogging and teaching. MBA in Data Science & Analytics (USP, GPA 98%) with a proven track record of delivering impactful, data-driven business solutions.

Find me via [Linkedin](https://www.linkedin.com/in/gurezende/)