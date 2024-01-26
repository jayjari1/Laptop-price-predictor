import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the XGBoost model
model_filename = 'xgboost_model.pkl'
xgb_reg = joblib.load(model_filename)

# Columns used during training
all_columns = ['Ram', 'Weight', 'PPI', 'IPS', 'Retina', 'Touchscreen', 'HD', 'SSD', 'HDD', 'Flash_Storage', 'Hybrid', 
               'Company_Apple', 'Company_Asus', 'Company_Dell', 'Company_HP', 'Company_Lenovo', 'Company_MSI', 
               'Company_Toshiba', 'TypeName_Gaming', 'TypeName_Netbook', 'TypeName_Notebook', 'TypeName_Ultrabook', 
               'TypeName_Workstation', 'Cpu_Intel Core i3', 'Cpu_Intel Core i5', 'Cpu_Intel Core i7', 'Gpu_Intel HD Graphics', 
               'Gpu_Intel Other', 'Gpu_Intel UHD Graphics', 'Gpu_Nvidia', 'OpSys_Mac', 'OpSys_No OS', 'OpSys_Windows 10', 
               'OpSys_Windows 7']

# Dropped dummy variables
dropped_dummies = ['Company_Acer', 'TypeName_2 in 1 Convertible', 'Cpu_AMD', 'Gpu_AMD']

# Actual categorical features (excluding dropped dummy variables)
categorical_features = [col for col in all_columns if col not in dropped_dummies]

# Sidebar with user input
st.sidebar.header('Laptop Price Predictor')

# Categorical features at the top
company_options = ['Apple', 'Asus', 'Dell', 'HP', 'Lenovo', 'MSI', 'Toshiba','Acer']
company = st.sidebar.selectbox('Company', company_options)

type_options = ['Gaming', 'Netbook', 'Notebook', 'Ultrabook', 'Workstation','2 in 1 Convertible']
type_name = st.sidebar.selectbox('Type Name', type_options)

cpu_options = ['Intel Core i3', 'Intel Core i5', 'Intel Core i7','AMD']
cpu = st.sidebar.selectbox('CPU', cpu_options)

gpu_options = ['Intel HD Graphics', 'Intel Other', 'Intel UHD Graphics', 'Nvidia','AMD']
gpu = st.sidebar.selectbox('GPU', gpu_options)

opsys_options = ['Mac', 'No OS', 'Windows 10', 'Windows 7', 'Linux']
opsys = st.sidebar.selectbox('Operating System', opsys_options)

# Numeric features
ram = st.sidebar.slider('RAM (GB)', 2, 64, 8)
weight = st.sidebar.slider('Weight (kg)', 1.0, 5.0, 2.5)

# Radio button to choose manual or slider input for Resolution
resolution_input_choice = st.sidebar.radio('Choose Resolution Input', ['Manual', 'Slider'])

# If the user chooses manual input, provide text input for Resolution Width and Height
if resolution_input_choice == 'Manual':
    manual_resolution_width = st.sidebar.text_input('Resolution Width (manual input)')
    manual_resolution_height = st.sidebar.text_input('Resolution Height (manual input)')

    # If both manual input values are provided, use them; otherwise, use sliders
    if manual_resolution_width and manual_resolution_height:
        resolution_width = int(manual_resolution_width)
        resolution_height = int(manual_resolution_height)
    else:
        resolution_width = st.sidebar.slider('Resolution Width', 800, 3840, 1920)
        resolution_height = st.sidebar.slider('Resolution Height', 600, 2160, 1080)
else:
    # Use sliders for Resolution
    resolution_width = st.sidebar.slider('Resolution Width', 800, 3840, 1920)
    resolution_height = st.sidebar.slider('Resolution Height', 600, 2160, 1080)

# Use slider for Inch
inch = st.sidebar.slider('Inch', 10, 20, 15)

# Calculate PPI
ppi = np.sqrt((resolution_width**2 + resolution_height**2) / inch**2)

# Binary features
ips = st.sidebar.checkbox('IPS')
retina = st.sidebar.checkbox('Retina')
touchscreen = st.sidebar.checkbox('Touchscreen')
hd_options = {0: 'No HD', 1: 'Full HD', 2: 'QHD', 3: '4K HD'}  # HD categories
hd = st.sidebar.selectbox('HD', list(hd_options.values()))

# Create a DataFrame with user input
user_input = pd.DataFrame({
    'Ram': [ram],
    'Weight': [weight],
    'PPI': [ppi],
    'IPS': [1 if ips else 0],
    'Retina': [1 if retina else 0],
    'Touchscreen': [1 if touchscreen else 0],
    'HD': [key for key, value in hd_options.items() if value == hd][0],
    'Company_' + company: [1] if 'Company_' + company in categorical_features else [0],
    'TypeName_' + type_name: [1] if 'TypeName_' + type_name in categorical_features else [0],
    'Cpu_' + cpu: [1] if 'Cpu_' + cpu in categorical_features else [0],
    'Gpu_' + gpu: [1] if 'Gpu_' + gpu in categorical_features else [0],
    'OpSys_' + opsys: [1] if 'OpSys_' + opsys in categorical_features else [0],
})
# Ensure all columns used during training are present in the user input
for column in categorical_features:
    if column not in user_input.columns:
        user_input[column] = [0]

# Reorder columns to match the training order
user_input = user_input[all_columns]

# Make prediction with the model
predicted_price = xgb_reg.predict(user_input)

# Display the prediction
st.write('Predicted Price:', predicted_price[0])
