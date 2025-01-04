import streamlit as st

# Title and subtitle
st.title("Pupil Detection System")
st.subheader("Welcome to the interactive application for pupil detection and response analysis!")

# Introduction text
st.markdown("""
This project is designed to leverage advanced machine learning techniques for accurate pupil detection 
in response to various stimuli. The system aims to provide insights into human visual response patterns 
by processing real-time video feed from a webcam or other imaging devices.

### Key Features:
- **Real-time Pupil Tracking:** Uses advanced algorithms for accurate and efficient pupil localization.
- **Customizable Settings:** Adjust brightness, contrast, and zoom for optimized performance under different lighting conditions.
- **Stimulus Response Analysis:** Tracks and records pupil responses to stimuli for research and analysis.

### Applications:
- **Medical Research:** Useful for diagnosing and studying neurological conditions.
- **Behavioral Studies:** Insights into cognitive and emotional states based on pupil dilation.
- **User Interface Design:** Adapts systems based on user attention and focus.

### Getting Started:
1. **Adjust Settings:** Navigate to the settings tab to configure brightness, contrast, and zoom levels.
2. **Start the Feed:** Use the main application to initialize the camera and begin detection.
3. **Analyze Results:** Save and interpret the detected patterns for your research or application.

We hope this tool enhances your understanding and application of pupil response analysis. Let's dive in!
""")
