# Customer Segmentation with K-Means

An unsupervised machine learning project that groups customers into segments based on age, marital status, education, work experience, family size and spending score.

**Live demo:** _add your Streamlit link here_

## The problem
Businesses can't treat every customer the same. This project finds natural groups in customer data so marketing, pricing and products can be targeted at each group.

## Approach
1. **Cleaning:** numeric gaps (`Work_Experience`, `Family_Size`) filled with the median; categorical gaps (`Ever_Married`, `Graduated`) filled with the mode.
2. **Feature encoding:** Gender, Ever_Married and Graduated mapped to 0/1; Spending_Score mapped to an ordered 0/1/2 (Low < Average < High). `ID` is dropped, and `Profession` / `Var_1` are left out so one-hot encoding doesn't dominate K-Means distances.
3. **Scaling:** `StandardScaler`, because K-Means is distance-based.
4. **Model:** `KMeans` with 4 clusters (`random_state=42`, `n_init=10`).
5. **Profiling:** average feature values and spending/profession mix per segment.
6. **Visualisation:** PCA reduces 7 features to 2 components for plotting.

## Run it locally
```bash
git clone https://github.com/EricVictory01/customer-segmentation.git
cd customer-segmentation
pip install -r requirements.txt
streamlit run app.py
```

## Project structure
```
customer-segmentation/
├── app.py                        # Streamlit app
├── customer_segmentation.ipynb   # Original analysis notebook
├── requirements.txt
├── data/Test.csv                 # Dataset
└── README.md
```

## Tech stack
Python, pandas, scikit-learn, matplotlib, Streamlit

## Author
Eric Victory Onyinyechi, [GitHub](https://github.com/EricVictory01)
