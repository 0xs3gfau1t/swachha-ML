# Swochha - Smart garbage management system
## Machine Learning Implementation

**Included Features**
- Roadside litter throwing detection using yolo v8 model
- Household garbage classification using CNN Architecture
- Interactive chatbot using Rasaframework

**Setup Guide**  
Python version >=3.10 required. 
```
git clone https://github.com/0xs3gfau1t/waste-ML.git
python -m venv venv
source venv/bin/activate
pip install rasa
pip install -r garbage_classification/requirements.txt
```
**Start Rasa Chatbot**
```
cd chatbot
rasa run
rasa run actions
```

**Start garbage classification and detection servers**
```
cd garbage_classification
python app.py
```
Main application repo: https://github.com/0xs3gfau1t/Swachha
