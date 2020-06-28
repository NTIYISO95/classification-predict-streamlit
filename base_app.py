"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os


# Data dependencies
import pandas as pd
from PIL import Image


#image
image = Image.open('resources/testimage.jpg')
infor = Image.open('resources/infor.jpg')
image2 = Image.open('resources/skl.png')

# Vectorizer
news_vectorizer = open("resources/vectorizer_draft.pkl","rb")
labeller = open("resources/labeller_draft.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Hi! I am a Tweet Classifier.")
	#st.subheader("Choose an option on the left and lets see what we can do")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Home", "Prediction", "Classifier Information", "Creator Information"]
	
	#application production here
	st.sidebar.markdown("Choose an option from the dropdown over here ")
	selection = st.sidebar.selectbox("Choose Option", options)

	#home page - landing page
	if selection == "Home":
		st.subheader("All of my learning is thanks to SKLearn. I am trained in four classification models, using sklearn libraries!")
		st.markdown('**Head over to the predictions tab to see how much I can do.**')
		# You can read a markdown file from supporting resources folder
		st.image(image2,use_column_width=True)


	# Building out the "Information" page
	if selection == "Classifier Information":
		st.subheader("You can browse through this information if you would like to understand how I classify your tweets.")
		st.markdown('*this information can be found on Wikipedia*')
		# You can read a markdown file from supporting resources folder
		st.image(infor,use_column_width=True)

		st.subheader("The data I train with - Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the "Creators" page
	if selection == "Creator Information":
		st.markdown("## **These are my trainers**")
		# You can read a markdown file from supporting resources folder
		st.image(image, caption='They are all blue belts! I have great mentors',use_column_width=True)


	# Building out the predication page
	if selection == "Prediction":
		st.markdown("### Enter a message below, choose a classification and allow me to analyse it for you.")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		if st.button("Linear SVC Classification"):
			# Transforming user input with vectorizer
			#vect_text = tweet_cv.transform([tweet_text]).toarray()
			vect_text = [tweet_text]
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/LinearSVC_upsample.pkl"),"rb"))
			answer = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			if answer[0] == 1:
  				st.success('Your message has been classified as showing positive belief in climate change')
			elif answer[0] == 0:
  				st.success('Your message has been classified as showing being neutral towards climate change')
			elif answer[0] == 2:
  				st.success('Your message has been classified as news')
			else:
  				st.success('Your message has been classified as showing negative belief in climate change')
			#st.success("Text Categorized as: {}".format(prediction))

		if st.button("Naive Bayes Classification"):
			# Transforming user input with vectorizer
			#vect_text = tweet_cv.transform([tweet_text]).toarray()
			vect_text = [tweet_text]
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/naivebayes_mixsample.pkl"),"rb"))
			answer = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			if answer[0] == 1:
  				st.success('Your message has been classified as showing positive belief in climate change')
			elif answer[0] == 0:
  				st.success('Your message has been classified as showing being neutral towards climate change')
			elif answer[0] == 2:
  				st.success('Your message has been classified as news')
			else:
  				st.success('Your message has been classified as showing negative belief in climate change')
			#st.success("Text Categorized as: {}".format(prediction))
		
		if st.button("Random Forest Classification"):
			# Transforming user input with vectorizer
			#vect_text = tweet_cv.transform([tweet_text]).toarray()
			vect_text = [tweet_text]
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/randomforest_mixsample.pkl"),"rb"))
			answer = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			if answer[0] == 1:
  				st.success('Your message has been classified as showing positive belief in climate change')
			elif answer[0] == 0:
  				st.success('Your message has been classified as showing being neutral towards climate change')
			elif answer[0] == 2:
  				st.success('Your message has been classified as news')
			else:
  				st.success('Your message has been classified as showing negative belief in climate change')
			#st.success("Text Categorized as: {}".format(prediction))

		if st.button("SGD Classifier"):
			# Transforming user input with vectorizer
			#vect_text = tweet_cv.transform([tweet_text]).toarray()
			vect_text = [tweet_text]
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/SGD_upsample.pkl"),"rb"))
			answer = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			if answer[0] == 1:
  				st.success('Your message has been classified as showing positive belief in climate change')
			elif answer[0] == 0:
  				st.success('Your message has been classified as showing being neutral towards climate change')
			elif answer[0] == 2:
  				st.success('Your message has been classified as news')
			else:
  				st.success('Your message has been classified as showing negative belief in climate change')
			#st.success("Text Categorized as: {}".format(prediction))

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
