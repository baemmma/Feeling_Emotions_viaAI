import torch
from PIL import Image
from torchvision import transforms
from model import FaceEmotionModel  

MODEL_WEIGHTS_PATH = './best_trained_model_weights.pt'

def load_trained_model(path_to_weights: str = MODEL_WEIGHTS_PATH) -> FaceEmotionModel:
    model = FaceEmotionModel(num_classes=8) 
    model_dict = model.state_dict()
    pretrained_dict = torch.load(path_to_weights, map_location=torch.device('cpu'))
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model

def predict_emotion(image: Image) -> (str, float):
    model = load_trained_model()

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    preprocessed_image = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(preprocessed_image)
        predicted_class_index = torch.argmax(outputs, dim=1)
        confidence_score = torch.softmax(outputs, dim=1)[0, predicted_class_index[0]].item()

    emotions = {0: "anger", 1: "contempt", 2: "disgust", 3: "fear", 4: "happy", 5: "neutral", 6: "sad", 7: "surprise"}
    predicted_emotion = emotions.get(predicted_class_index[0], "Unknown")

    return predicted_emotion, confidence_score
