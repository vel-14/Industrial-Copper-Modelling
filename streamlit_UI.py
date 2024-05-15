import pandas as pd
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
import pickle

df = pd.read_csv('/Users/velmurugan/Desktop/velu/python_works/copper_model/preprocessed_data.csv')

status_map = {'Lost':0, 'Won':1, 'Draft':2, 'To be approved':3, 'Not lost for AM':4,
                                 'Wonderful':5, 'Revised':6, 'Offered':7, 'Offerable':8}

item_type_map = {'W':1,'WI':2,'S':3,'Others':4,'PL':5,'IPL':6,'SLAWR':7}

st.set_page_config(page_title="Industrial Copper Modelling",page_icon='🏭',layout='wide')

#setting up the bg color

def setting_bg():
    st.markdown(f""" <style>.stApp {{
                background: linear-gradient(to bottom, #FFD700, #FFA500);
            }}
           </style>""",
        unsafe_allow_html=True)

setting_bg()

st.markdown("<h1 style='text-align: center; color: #333333;'>Industrial Copper Modeling</h1>", unsafe_allow_html=True)

options = option_menu('Predictions',['Selling Price','Product Status'],icons=["cash-coin", "award-fill"])

if options == 'Selling Price':
    column1,column2,column3 = st.columns(3)

    with column1:
        st.image('/Users/velmurugan/Desktop/velu/python_works/copper_model/copper.jpg')

    with column2:
        status = st.selectbox('Status', ['Won','Draft','To be approved','Lost','Not lost for AM','Wonderful',
                                         'Revised','Offered','Offerable'])
        item_type = st.selectbox('Item Type', ['W','WI','S','Others','PL','IPL','SLAWR'])
        application = st.selectbox('Application', df['application'].unique())
        country = st.selectbox('Country Number', df['country'].unique())
        st.write("Minimum value=1,Maximum value=905")
        quantity=st.text_input("Enter Quantity in Tons")
        st.write("Minimum value=1,Maximum value=163416")
        id=st.text_input("Enter ID")

    with column3:
        st.write("Minimum Value=1.17, Maximum Value=14.32")
        thickness=st.text_input("Enter Thickness ")
        st.write("Minimum Value=1, Maximum Value=2990")
        width=st.text_input("Enter Width ")
        st.write("Minimum Value=12458, Maximum Value=30408185")
        customer=st.text_input("Enter  Customer ID ")
        st.write("Minimum Value=611728, Maximum Value=1722207579")
        product_ref=st.text_input("Enter  Product Refernce: ")

#loading the model for prediction

    file_path = '/Users/velmurugan/Desktop/velu/python_works/copper_model/reg_random_forest.pkl'

    with open(file_path,'rb') as f:
        reg_model = pickle.load(f)

    predict_button = st.button('Predict Selling Price')

    if predict_button:
        id = float(id)
        status = status_map.get(status)
        item_type = item_type_map.get(item_type)
        country = float(country) 
        application = float(application)  
        product_ref = float(product_ref)  
        quantity = np.log(float(quantity))
        thickness_log = np.log(float(thickness))
        width = float(width)
        customer = float(customer)

        sample = np.array([[id,quantity,customer,country,status,item_type,application,thickness,width,product_ref]])

        pred = reg_model.predict(sample)[0]

        pred = np.exp(pred)

        round_pred = round(pred,3)

        st.write('# :green[Predicted Selling price:]', f"${round_pred}")

elif options == 'Product Status':
    column1,column2,column3 = st.columns(3)

    with column1:
        st.image('/Users/velmurugan/Desktop/velu/python_works/copper_model/cu.jpg.webp')
    
    with column2:
        st.write('min=1 and max = 926')
        quantity = st.text_input('Enter quantity in Tons')
        
        customer_ids = df['customer'].unique()
        min_customer_id = min(customer_ids)
        max_customer_id = max(customer_ids)

        st.write(f"Min == {min_customer_id} and max = {max_customer_id}")
        customer = st.text_input('Enter customer id')

        country = st.selectbox('Select Country Number', df['country'].unique())
        item = st.selectbox('Select Item Type',['W','WI','S','Others','PL','IPL','SLAWR'])
        application = st.selectbox("Select application number",df['application'].unique())

    with column3:
        st.write("Minimum Value=1.17, Maximum Value=14.32")
        thickness = st.text_input("Enter Thickness")

        width_values = df['width'].unique()
        min_width = min(width_values)
        max_width = max(width_values)
        st.write(f"min = {min_width} and max = {max_width}")
        width = st.text_input("Enter Width")

        st.write("Minimum Value=611728, Maximum Value=1722207579")
        product_ref=st.text_input("Enter  Product Refernce: ")

        st.write(f"min = {396} and max = {1622}")
        selling_price = st.text_input("Enter Selling Price")

    predict_status = st.button("Predict Product Status")

    if predict_status:
        quantity = np.log(float(quantity))
        customer = float(customer)
        country = float(country) 
        item_type = item_type_map.get(item)
        application = float(application) 
        thickness = np.log(float(thickness))
        width = float(width) 
        product_ref = float(product_ref)  
        selling_price = np.log(float(selling_price))

        file_path = '/Users/velmurugan/Desktop/velu/python_works/copper_model/random_forest_model.pkl'

        with open(file_path,'rb') as f:
            loaded_model = pickle.load(f)

        X_new = np.array([[quantity,customer,country,item_type,application,thickness,width,product_ref,selling_price]])

        pred = loaded_model.predict(X_new)[0]

        if pred ==1:
            st.write('# :green[The Status is: Won]')
        else:
            st.write('# :red[The Status is: Lost]')



       





