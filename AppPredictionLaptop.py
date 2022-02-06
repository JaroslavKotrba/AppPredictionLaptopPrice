
# cd C:\Users\HP\OneDrive\Documents\Python Anaconda\Streamlit_Laptop_App
# streamlit run AppPredictionLaptop.py

# API: https://docs.streamlit.io/library/api-reference
# dashboard: https://www.youtube.com/watch?v=Sb0A9i6d320

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# other libraries
from PIL import Image
import requests
from streamlit_lottie import st_lottie
from plotly import graph_objs as go

file1 = open('pipe.pkl', 'rb')
rf = pickle.load(file1)
file1.close()

data = pd.read_csv("traineddata.csv")

def main():
    # emojis: https://www.webfx.com/tools/emoji-cheat-sheet/
    # animations: https://lottiefiles.com/search?q=pc&category=animations
    st.set_page_config(page_icon=":computer:", layout="wide")

    st.sidebar.title("NAVIGATION")

    menu = ["Home", "Visualisation", "Model described", "About"]
    
    choice = st.sidebar.radio("Please select a page:", menu)
    
    st.sidebar.markdown("""---""")

    st.sidebar.subheader("More info:"); 
    st.sidebar.write(":computer: https://jaroslavkotrba.com")
    
    st.sidebar.write("Copyright © 2022")

    #st.sidebar.write(":star:"*5)

    if choice == "Home":
        # Title
        st.markdown("<h1 style='text-align: center; color: steelblue;'>Laptop Price Predictor</h1>", unsafe_allow_html=True)

        # Image
        from PIL import Image
        image = Image.open('./pc.png')
        st.image(image, caption='Laptop Price Prediction with AI', use_column_width=True)
        
        st.write("<p style='text-align: center; color: pink;'>Predict the price of a laptop that would suit your needs the best.</p>", unsafe_allow_html=True)

        # Brand
        default_company = list(data['Company'].unique()).index('Apple')
        company = st.selectbox('Brand', data['Company'].unique(), index=default_company)

        # Type of laptop
        type = st.selectbox('Type', data['TypeName'].unique(), index=1)

        # Ram present in laptop
        ram = st.selectbox('Ram (in GB)', data['Ram'].unique(), index=1)

        # Os of laptop
        os = st.selectbox('OS', data['OpSys'].unique(), index=0)

        # Weight of laptop
        weight = st.number_input('Weight of the laptop', 1.25)

        # Touchscreen available in laptop or not
        touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'], index=0)

        # IPS
        ips = st.selectbox('IPS', ['No', 'Yes'], index=0)

        # Screen size
        screen_size = st.number_input('Screen Size', 13)

        # Resolution of laptop
        resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'], index=6)

        # Cpu
        cpu = st.selectbox('CPU', data['CPU_name'].unique(), index=1)

        # Hdd
        hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048], index=0)

        # Ssd
        ssd = st.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024], index=4)

        gpu = st.selectbox('GPU brand', data['Gpu brand'].unique(), index=0)

        left_column, right_column = st.columns(2)
        with left_column:
            pretty_result = {"company": company, "type": type, "ram": ram, "weight": weight, "touchs_creen": touchscreen,
            "ips": ips, "screen_size": screen_size, "screen_resolution": resolution, "cpu": cpu, "hdd": hdd, "ssd": ssd, "gpu": gpu}
            st.json(pretty_result)
        
        with right_column:
            import requests
            from streamlit_lottie import st_lottie
        
            def load_lottieurl(url):
                r = requests.get(url)
                if r.status_code != 200:
                    return None
                return r.json()
        
            lottie_coding = load_lottieurl("https://assets1.lottiefiles.com/private_files/lf30_zdeFcW.json")

            st_lottie(lottie_coding, height=370, key="coding")

        st.markdown("""---""")

        if st.button('Predict'):
            
            ppi = None
            if touchscreen == 'Yes':
                touchscreen = 1
            else:
                touchscreen = 0
                
            if ips == 'Yes':
                ips = 1
            else:
                ips = 0

            X_resolution = int(resolution.split('x')[0])
            Y_resolution = int(resolution.split('x')[1])

            ppi = ((X_resolution**2)+(Y_resolution**2))**0.5/(screen_size)

            query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])

            query = query.reshape(1, 12)

            prediction = int(np.exp(rf.predict(query)[0]))

            st.title("Predicted price: " + str(round(prediction*0.4))+" CZK")

        st.subheader("More info:")
        st.write("To see other author’s projects: https://jaroslavkotrba.com")
                # ---- HIDE STREAMLIT STYLE ----
        hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
        st.markdown(hide_st_style, unsafe_allow_html=True)
    
    elif choice == "Visualisation":
        # Title
        st.markdown("<h1 style='text-align: center; color: #DC143C;'>Laptop Data Visualisation</h1>", unsafe_allow_html=True)

        st.write("<p style='text-align: center; color: pink;'>Predict the price of a laptop that would suit your needs the best.</p>", unsafe_allow_html=True)

        from plotly import graph_objs as go
        st.subheader("Laptop Brand Count:")
        company = data.groupby(["Company"]).size().reset_index(name='Freq')
        def plotly_data():
            fig = go.Figure([go.Bar(x=company['Company'], y=company['Freq'])])
            fig.update_traces(marker_color='#DC143C', marker_line_color='white', marker_line_width=1.5, opacity=1)
            fig.update_layout(
                    plot_bgcolor = "rgba(0,0,0,0)",
                    autosize=True,
                    yaxis=dict(showgrid=False),
                    xaxis_title="Brand",
                    yaxis_title="Amount of models",
                    title={'text': "", 'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'})
            st.plotly_chart(fig)
        plotly_data()

        st.markdown("""---""")

        st.subheader("Laptop Ram Count:")
        ram = data.groupby(["Ram"]).size().reset_index(name='Freq')
        ram['Ram'] = ram['Ram'].astype(str)
        def plotly_data():
            fig = go.Figure([go.Bar(x=ram['Ram'], y=ram['Freq'])])
            fig.update_traces(marker_color='#DC143C', marker_line_color='white', marker_line_width=1.5, opacity=1)
            fig.update_layout(
                    plot_bgcolor = "rgba(0,0,0,0)",
                    autosize=True,
                    yaxis=dict(showgrid=False),
                    xaxis_title="Ram",
                    yaxis_title="Amount of models",
                    title={'text': "", 'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'})
            st.plotly_chart(fig)
        plotly_data()

        st.markdown("""---""")

        st.subheader("Laptop Weight Mean:")
        weight = data.groupby(["Company"]).agg({'Weight':'mean'}).reset_index().rename(columns={"Weight":"Weight mean"})
        weight['Weight mean'] = round(weight['Weight mean'], 2)
        def plotly_data():
            fig = go.Figure([go.Bar(x=weight['Company'], y=weight['Weight mean'])])
            fig.update_traces(marker_color='#DC143C', marker_line_color='white', marker_line_width=1.5, opacity=1)
            fig.update_layout(
                    plot_bgcolor = "rgba(0,0,0,0)",
                    autosize=True,
                    yaxis=dict(showgrid=False),
                    xaxis_title="Company",
                    yaxis_title="Weight average in (kg)",
                    title={'text': "", 'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'})
            st.plotly_chart(fig)
        plotly_data()

        st.markdown("""---""")

        st.subheader("More info:")
        st.write("To see other author’s projects: https://jaroslavkotrba.com")
                # ---- HIDE STREAMLIT STYLE ----
        hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
        st.markdown(hide_st_style, unsafe_allow_html=True)
    
    elif choice == "Model described":
        # Title
        st.markdown("<h1 style='text-align: center; color: dodgerblue;'>Laptop Model Described</h1>", unsafe_allow_html=True)

        st.write("<p style='text-align: center; color: pink;'>Predict the price of a laptop that would suit your needs the best.</p>", unsafe_allow_html=True)

        st.write("A random forest is a machine learning technique that's used to solve regression and classification problems.")
        st.write("Random Forest Regression is a supervised learning algorithm that uses ensemble learning method for regression. Ensemble learning method is a technique that combines predictions from multiple machine learning algorithms to make a more accurate prediction than a single model.")
        st.write("Random Forest Regression model is powerful and accurate. It usually performs great on many problems, including features with non-linear relationships. Disadvantages, however, include the following: there is no interpretability, overfitting may easily occur, we must choose the number of trees to include in the model.")

        st.markdown("""---""")

        import requests
        from streamlit_lottie import st_lottie
        
        def load_lottieurl(url):
            r = requests.get(url)
            if r.status_code != 200:
                return None
            return r.json()
        
        lottie_coding = load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_ba013t74.json")

        st_lottie(lottie_coding, height=200, key="coding")

        st.markdown("""---""")

        st.subheader("More info:")
        st.write("To see other author’s projects: https://jaroslavkotrba.com")
                # ---- HIDE STREAMLIT STYLE ----
        hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
        st.markdown(hide_st_style, unsafe_allow_html=True)

    elif choice == "About":
        # Title
        st.markdown("<h1 style='text-align: center; color: #DC143C;'>Laptop Price About</h1>", unsafe_allow_html=True)
        st.write("<p style='text-align: center; color: pink;'>Predict the price of a laptop that would suit your needs the best.</p>", unsafe_allow_html=True)

        st.write("I created this app to be able to predict price of a laptop that I can configure on my own. To know a new laptop price according all models on the market is essential, just wanted to make sure that the price I am going to pay will be correct :)")

        st.markdown("""---""")

        # CONTACT
        # Use local CSS
        def local_css(file_name):
            with open(file_name) as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

        local_css("style.css")

        left_column, right_column = st.columns(2)
        with left_column:
            st.write("##")
            import requests
            from streamlit_lottie import st_lottie

            def load_lottieurl(url):
                r = requests.get(url)
                if r.status_code != 200:
                    return None
                return r.json()
                
            lottie_coding = load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_naj9ijgt.json")

            st_lottie(lottie_coding, height=300, key="coding")
            
        with right_column:
            with st.container():
                st.header("Contact me: ")
                st.write("##")
                # Documention: https://formsubmit.co/
                contact_form = """
                <form action="https://formsubmit.co/jaroslav.kotrba@gmail.com" method="POST">
                    <input type="hidden" name="_captcha" value="false">
                    <input type="text" name="name" placeholder="Your name" required>
                    <input type="email" name="email" placeholder="Your email" required>
                    <textarea name="message" placeholder="Your message here" required></textarea>
                    <button type="submit">Send</button>
                </form>
                """
            st.markdown(contact_form, unsafe_allow_html=True)

        st.markdown("""---""")

        st.subheader("More info:")
        st.write("To see other author’s projects: https://jaroslavkotrba.com")
                # ---- HIDE STREAMLIT STYLE ----
        hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
        st.markdown(hide_st_style, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
