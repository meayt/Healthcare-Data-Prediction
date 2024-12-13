# BMI healthcare daata Prediction App

## Project Title
BMI Prediction App: Leveraging Healthcare Data for Body Mass Index Predictions

## Group Members
- **Meet Chauhan** (KU ID : KU2407U833)
- **Shyam Thakor** (KU ID : KU2407U832)

## Objective of the Project
To develop a user-friendly application that predicts Body Mass Index (BMI) based on healthcare data, providing users with actionable insights to maintain or improve their health.

## Tools and Libraries Used
- **Programming Language:** Python
- **Libraries:**
  - Pandas
  - NumPy
  - Scikit-learn
  - Flask
  - Matplotlib
  - Seaborn
- **IDE:** Visual Studio Code / Jupyter Notebook
- **Version Control:** Git

## Data Source(s)
- Public healthcare datasets sourced from [Kaggle](https://www.kaggle.com/) and [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php).
- Synthetic datasets generated using Python's random module for supplementary testing.

## Execution Steps
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/BMI-Prediction-App.git
   cd BMI-Prediction-App
   ```
2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Application:**
   ```bash
   python app.py
   ```
4. **Access the App:**
   Open your browser and navigate to `http://127.0.0.1:5000`.

5. **Test the App:**
   Upload a sample healthcare dataset or manually input health metrics (age, height, weight, etc.) to view BMI predictions.

## Summary of Results
- Achieved an **accuracy of 85%** in BMI predictions using the Random Forest Regressor.
- Integrated a user-friendly interface for real-time predictions.
- Visualized trends and distributions in BMI data to enhance interpretability.

## Challenges Faced
- **Data Imbalance:**
  Addressed skewness in the dataset by applying SMOTE (Synthetic Minority Oversampling Technique).
- **Feature Engineering:**
  Extracting meaningful features from raw healthcare data was time-intensive.
- **Model Generalization:**
  Ensured the model performs well on unseen data through rigorous cross-validation.

We hope this project serves as a stepping stone for leveraging healthcare data to improve personal wellness and encourage preventive care.

