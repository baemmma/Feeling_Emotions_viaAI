## Feeling Emotions via AI

![Image](https://static.scientificamerican.com/sciam/cache/file/5514ED3E-E231-41CA-BDBC452744D57FC0_source.jpg?w=590&h=800&3CB5EA0B-23DC-4BF5-A917D649121E5204)

*Photo by [Mark Daynes](https://unsplash.com/photos/J6p8nfCEuS4)*

### Overview

This project aims to decipher human emotions through real-time facial expressions. It encompasses steps to acquire the dataset, train and validate the model, develop a Streamlit application for real-time emotion analysis, and deploy the application on Google Cloud.

### Getting Started

#### Dataset Acquisition

1. Download the AffectNet dataset from the provided link: [AffectNet Dataset](https://paperswithcode.com/dataset/affectnet).
2. Extract the dataset files and divide it into training and validation sets. 
2. Conduct label-to-tag mapping for better comprehension. 
=> The model underwent multiple training sessions to acquire the weights used in `utils.py`, subsequently called in `app.py`.



#### Model Training and Validation

1. Utilize Google Colab's computational resources for training and validation.
2. Implement necessary pre-processing steps and utilize deep learning frameworks for fine-tuning.
3. Preserve the architecture and weights of your trained model for future use.

#### Application Development

1. Use Streamlit to create an interactive web application for real-time emotion analysis.
2. Integrate your trained emotion detection model for real-time analysis via webcam input.
3. Enhance the user experience by incorporating visual elements such as bounding boxes and emotion labels.

#### Deployment on Google Cloud

1. Prepare your Streamlit application for deployment by specifying dependencies in a `requirements.txt` file.
2. Utilize Google Cloud's infrastructure to deploy your Streamlit app, making it accessible via a web URL.
3. Share your deployed app with the world to explore the fascinating realm of emotion decoding.

### About the Author

This project was created by Emna BAHRI from Artefact.

### Dedication

This project is dedicated to individuals with autism, demonstrating the potential of AI to enhance emotional understanding and interaction.

