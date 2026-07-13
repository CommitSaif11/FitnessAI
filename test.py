import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import streamlit as st
print("Mediapipe version:", mp.__version__)
print("OpenCV version:", cv2.__version__)
print("NumPy version:", np.__version__)
print("Pandas version:", pd.__version__)
st.write("Streamlit version:", st.__version__)