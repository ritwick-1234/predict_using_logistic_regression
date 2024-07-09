import streamlit as st
import pickle

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from  sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression



st.markdown(
    """
    <h1 style='text-align: center;'>DIABETES PREDICTION Using Logistic Regression</h1>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <h4 style='text-align: center;'>Analysis of the dataset</h4>
    """,
    unsafe_allow_html=True
)
df=pd.read_csv("diabetes.csv")
markdown_title = "### **CSV File Upload and Display**"
st.markdown(markdown_title)


uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:

    data_set = df


    st.write("### Dataset")
    html_string = f"<h4><b>The shape of the dataset is:</b> {df.shape}</h4>"
    html_string_1=f" <h4>The size of the csv file is {uploaded_file.size} byte</h4>"
    st.write( html_string_1,unsafe_allow_html=True)

    st.write(html_string, unsafe_allow_html=True)

    st.write(df)
st.sidebar.markdown("<h3 style='text-align:center;'>See Options</h3>",unsafe_allow_html=True)
option = st.sidebar.selectbox(
    "**See bar chart for each column**",
    ('Hide Bar Charts','Pregnancies','Glucose','Blood Pressure','Skin Thickness','Insulin','BMI','DiabetesPedigreeFunction','Age' ,'show all together')
)
if option=='Hide Bar Charts':
    pass
elif option == 'Pregnancies':
    st.write("Bar Chart for Pregnancies")
    st.bar_chart(df['Pregnancies'])
elif option == 'Glucose':
    st.write("Bar Chart for Glucose")
    st.bar_chart(df['Glucose'])
elif option == 'Blood Pressure':
    st.write("Bar Chart for Blood Pressure")
    st.bar_chart(df['BloodPressure'])
elif option == 'Skin Thickness':
    st.write("Bar Chart for Skin Thickness")
    st.bar_chart(df['SkinThickness'])
elif option == 'Insulin':
    st.write("Bar Chart for Insulin")
    st.bar_chart(df['Insulin'])
elif option == 'BMI':
    st.write("Bar Chart for BMI")
    st.bar_chart(df['BMI'])
elif option == 'DiabetesPedigreeFunction':
    st.write("Bar Chart for Diabetes Pedigree Function")
    st.bar_chart(df['DiabetesPedigreeFunction'])
elif option == 'Age':
    st.write("Bar Chart for Age")
    st.bar_chart(df['Age'])
elif option == 'show all together':
    
    st.bar_chart(df)
    

data_frame = pd.DataFrame(df)


# show_stats = st.sidebar.checkbox("Show Statistics")


# if show_stats:
#     st.write("Summary Statistics:")
#     st.write(data_frame.describe())
#     show_stats = st.sidebar.checkbox("Show Statistics")

# show_stats_1 = st.sidebar.checkbox("See contribution of columns")
# if show_stats_1:
#     features = ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 
#             'Insulin', 'BMI', 'Diabetes Pedigree Function', 'Age']
#     contributions = [0.221898, 0.466581, 0.065068, 0.074752, 0.130548, 0.292695, 0.173844, 0.238356]

# # Plotting
#     plt.figure(figsize=(8, 8))
#     plt.pie(contributions, labels=features, autopct='%1.1f%%', startangle=140)
#     plt.title('Contribution of Features to Predict Outcome')
#     plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
#     plt.show()
import streamlit as st
import matplotlib.pyplot as plt


features = ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 
            'Insulin', 'BMI', 'Diabetes Pedigree Function', 'Age']
contributions = [0.221898, 0.466581, 0.065068, 0.074752, 0.130548, 0.292695, 0.173844, 0.238356]


show_stats = st.sidebar.checkbox("Show Statistics")

if show_stats:
    st.write("Summary Statistics:")
    st.write(data_frame.describe())



show_stats_1 = st.sidebar.checkbox("See contribution of columns", key="see_contributions")

if show_stats_1:
    # Plotting
    plt.figure(figsize=(8, 8))
    plt.pie(contributions, labels=features, autopct='%1.1f%%', startangle=140)
    plt.title('Contribution of Features to Predict Outcome')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(plt)  # Display the plot in Streamlit
    st.title("Diabetic or Not")
with open('diabetes_part_1.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


# Function to predict diabetes
def predict_diabetes(data):
    # Preprocess input data if needed (e.g., scaling)
    # Make predictions
    prediction = model.predict(data)
   
    return prediction






st.sidebar.title('Diabetes Prediction')


option = st.sidebar.radio("Select an option", ('Predict Diabetes',))


if option == 'Predict Diabetes':
    st.title('Diabetes Prediction')

    
    with st.form("input_form"):
        st.write("Fill in the details to predict diabetes:")
        
        
        pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
        glucose = st.number_input("Glucose", min_value=0)
        blood_pressure = st.number_input("BloodPressure", min_value=0)
        skin_thickness = st.number_input("SkinThickness", min_value=0)
        insulin = st.number_input("Insulin", min_value=0)
        bmi = st.number_input("BMI", min_value=0.0)
        diabetes_pedigree = st.number_input("DiabetesPedigreeFunction", min_value=0.000,step=0.0001)
        age = st.number_input("Age", min_value=0, step=1)

        
        submitted = st.form_submit_button("Predict")

        if submitted:
            
            input_data = pd.DataFrame({
                'Pregnancies': [pregnancies],
                'Glucose': [glucose],
                'BloodPressure': [blood_pressure],
                'SkinThickness': [skin_thickness],
                'Insulin': [insulin],
                'BMI': [bmi],
                'DiabetesPedigreeFunction': [diabetes_pedigree],
                'Age': [age]
            })

           
            prediction = predict_diabetes(input_data)

          
            if prediction[0] == 1:
                st.error("The person is predicted to be diabetic.")
                st.write("Prediction: Diabetic")
            else:
                st.success("The person is predicted to be non-diabetic.")
                st.write("Prediction: Non-diabetic")
