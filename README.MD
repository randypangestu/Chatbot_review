# Spotify Review Chatbot

This project is a Spotify review chatbot that allows users to interactively analyze the spotify review from google play store.


## Getting Started

To run this project for the first time, please follow the steps below:

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
## Dependencies

Make sure you have the following dependencies installed:

- Python 3.x
- Streamlit
- [Add any other dependencies here]
