
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from PIL import Image
import datetime


animal_data = pd.read_csv('C:/Users/user-pc/Documents/Projects/Personal/animalProject/animalProject/animal-data-1.csv')  

st.title('Shelter Animals Database')

st.sidebar.checkbox("Show Analysis by State", True, key=1)
select = st.sidebar.selectbox('Select a State',animal_data['animalage'])

col1, col2,col3, col4 = st.beta_columns(4)
image = Image.open('Cutest-Pictures-Beagles.jpg')
col1.image(image, caption='I need a home',
         width=380)
image = Image.open('download.jpg')
col2.image(image, caption='I need a home',
         width=380)
image = Image.open('image.jpg')
col3.image(image, caption='I need a home',
         width=380)
image = Image.open('cute-cat-photos-1593441022.jpg')
col4.image(image, caption='I need a home',
         width=350)

# left_column, right_column = st.beta_columns(2)
# pressed = left_column.button('See Dashboard')

# if pressed:
if st.checkbox("Show Dashboard"):
    col1, col2 = st.beta_columns(2)
    fig = plt.figure(figsize=(10,3))
    ax = fig.add_subplot(111)

    # Plots
    # get top 10 most frequent names
    n = 5
    l_top = animal_data['speciesname'].value_counts()[:n].index.tolist()
    df_tops = pd.DataFrame(columns=animal_data.columns)
    for top in l_top:
        df_top = animal_data[animal_data['speciesname']==top]
        df_tops = pd.concat([df_tops,df_top])

    ax.hist(df_tops['speciesname'], bins=14,color='green')
    col1.subheader('Top 5 Species in Database')
    col1.pyplot(fig)

    fig = plt.figure(figsize=(10,3))
    ax = fig.add_subplot(111)

    n = 5
    l_top2 = animal_data['animalname'].value_counts()[:n].index.tolist()
    df_tops2 = pd.DataFrame(columns=animal_data.columns)
    for top in l_top2:
        df_top2 = animal_data[animal_data['animalname']==top]
        df_tops2 = pd.concat([df_tops2,df_top2])

    ax.hist(df_tops2['animalname'], bins=14,color='green')
    col2.subheader('Top 5 Names in Database')
    col2.pyplot(fig)

    fig = plt.figure(figsize=(10,3))
    ax = fig.add_subplot(111)
    col1, col2 = st.beta_columns(2)
    n = 5
    l_top = animal_data['breedname'].value_counts()[:n].index.tolist()
    df_tops = pd.DataFrame(columns=animal_data.columns)
    for top in l_top:
        df_top = animal_data[animal_data['breedname']==top]
        df_tops = pd.concat([df_tops,df_top])

    ax.hist(df_tops['breedname'], bins=14,color='green')
    col1.subheader('Top 5 Breeds in Database')
    col1.pyplot(fig)

    fig = plt.figure(figsize=(10,3))
    ax = fig.add_subplot(111)

    n = 5
    l_top2 = animal_data['sexname'].value_counts()[:n].index.tolist()
    df_tops2 = pd.DataFrame(columns=animal_data.columns)
    for top in l_top2:
        df_top2 = animal_data[animal_data['sexname']==top]
        df_tops2 = pd.concat([df_tops2,df_top2])

    ax.hist(df_tops2['sexname'], bins=14,color='green')
    col2.subheader('Genders in Database')
    col2.pyplot(fig)

# pressed = right_column.button('Close Dashboard')


# expander = st.beta_expander("FAQ")
# expander.write("Here you could put in some really, really long explanations...")

status = st.radio("What animal are you checking in?",("Cat","Dog","House Rabbit","Other"))
breed = st.selectbox("The animal breed",["Domestic Short Hair","Domestic Medium Hair","Domestic Long Hair","Bully Breed Mix","Labrador Retriever","Other"])
age = st.slider("What is the animal's age?",0,25)
date = st.date_input("Intake date",datetime.datetime.now())
name = st.text_input("Enter the animal's name","Type here")

if st.button("Submit"):
    result = str("Let's find "+name.title()+" a home!")
    st.success(result)



