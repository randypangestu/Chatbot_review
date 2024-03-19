# Spotify Review Chatbot

Welcome to the Spotify Review Chatbot project! This interactive chatbot offers a unique way to delve into Spotify reviews from the Google Play Store. By leveraging advanced language models (LLMs), it provides insightful analyses and responses to user queries.

The chatbot uses llama index library and  utilizes a <b>local</b> LLM for embedding generation. This enables the chatbot to provide accurate and contextually relevant responses to user queries.


### Screenshot of the App
for the screenshot of the app, please check the screenshot folder

### Video of the app running
you can find the video of app demo [here](https://drive.google.com/file/d/1Z3yXrxAgzT8PKbrKG6wOFTHSSqGObfFW/view?usp=drive_link)

## Getting Started
 
To run this project for the first time, please follow these steps:

1. Download the Spotify review dataset from [here](https://drive.google.com/file/d/1_xaRB6d2K_9-1dUmdU0GjtaqPO7uQnTM/view).
2. Run the Python script `data_preprocessing.py` with the path to the Spotify review dataset as an argument.
    ```
    python data_preprocessing.py <spotify_review_path>
    ```
3. Run the script `create_embedding.py`.
    ```
    python create_embedding.py
    ```
4. Run the chatbot app using Streamlit.
    ```
    streamlit run app.py
    ```

## Installations

To install the required dependencies, run the following command:
```
pip install -r requirements.txt
```
notes: the requirements might not be completed, some packages might be missing from the list
## Dependencies

Make sure you have the following dependencies installed:

- llamaindex
- Streamlit
- pandas