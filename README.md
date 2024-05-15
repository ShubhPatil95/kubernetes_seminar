# kubernetes_seminar

<h4>Before starting with this project, make sure you have docker for desktop is installed and kubernetes is enabled there as shown in below image1</h4>

<h4>Also verify if minikube is installed and it should nbe started using below command</h4>

<br>[Click Here To Go To Image1](https://github.com/ShubhPatil95/kubernetes_seminar/assets/74223025/29e58edf-13a0-4648-a013-89d9fa5773d3)</br>


<h4> Create virtual environment</h4>
```ruby
python -m venv venv_kubernetes
source venv_kubernetes/Scripts/activate
```

<p>
    
```ruby
minikube start
```

Step1: Create directory structure.
```ruby
mkdir kubernetes_demo
cd kubernetes_demo
mkdir src
mkdir data
```

Step2:Create and install requirnment.txt with below content
```ruby
nano requirements.txt 
```
```ruby
blinker==1.8.2
click==8.1.7
colorama==0.4.6
Flask==3.0.3
itsdangerous==2.2.0
Jinja2==3.1.4
joblib==1.4.2
MarkupSafe==2.1.5
numpy==1.26.4
pandas==2.2.2
python-dateutil==2.9.0.post0
pytz==2024.1
scikit-learn==1.4.2
scipy==1.13.0
six==1.16.0
threadpoolctl==3.5.0
tzdata==2024.1
Werkzeug==3.0.3
```

Step3: Create below files in under src folder
```ruby
nano data_preprocessing.py
```
```ruby
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

def load_and_preprocess_data(filepath):
    # Load dataset
    df = pd.read_csv(filepath)

    # Separate features and target
    X = df[['area', 'bedrooms', 'stories', 'mainroad', 'basement']]
    y = df['price']

    # Define column transformer
    numeric_features = ['area', 'bedrooms', 'stories']
    categorical_features = ['mainroad', 'basement']

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Preprocess data
    X_preprocessed = preprocessor.fit_transform(X)

    # Save the preprocessor for future use
    joblib.dump(preprocessor, 'preprocessor.joblib')

    return X_preprocessed, y

if __name__ == "__main__":
    print("Execution started for data preprocessing")
    X, y = load_and_preprocess_data('../data/Housing.csv')
    print("Shape of data",X.shape, y.shape)
    print("Execution is finished for data preprocessing")
```

```ruby
nano model_training.py 
```

```ruby
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from data_preprocessing import load_and_preprocess_data

print("Model training is started")
# Load and preprocess the data
X, y = load_and_preprocess_data('./data/Housing.csv')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

print("Model training is finished")
# Save the trained model
joblib.dump(model, '../house_price_model.joblib')

# Evaluate the model
score = model.score(X_test, y_test)
print(f'Model R^2 score: {score}')

```

```ruby
nano model_inference.py  
```
```ruby
import joblib
import numpy as np
import pandas as pd

# Load the pre-trained model and preprocessor
model = joblib.load('../house_price_model.joblib')
preprocessor = joblib.load('preprocessor.joblib')

def predict_price(area, bedrooms, stories, mainroad, basement):
    # Create a dataframe with the input data
    input_data = pd.DataFrame([[area, bedrooms, stories, mainroad, basement]],
                              columns=['area', 'bedrooms', 'stories', 'mainroad', 'basement'])

    # Preprocess the input data
    input_data_preprocessed = preprocessor.transform(input_data)

    # Make prediction
    predicted_price = model.predict(input_data_preprocessed)
    return predicted_price[0]

if __name__ == "__main__":
    # Example usage
    area = 3000
    bedrooms = 3
    stories = 2
    mainroad = 'yes'
    basement = 'no'
    print("Given inputs are area=3000, bedrooms=3, stories=2, mainroad='yes', basement='no',")
    price = predict_price(area, bedrooms, stories, mainroad, basement)
    print(f'Predicted house price: {price}')

```


Step 4: Create app.py 
```ruby
nano app.py
```

```ruby
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("house_price_model.joblib")

@app.route('/predict', methods=['POST'])
def predict():
    # Get input features from request
    data = request.get_json()

    # Extract features
    area = data['area']
    bedrooms = data['bedrooms']
    stories = data['stories']
    mainroad = 1 if data['mainroad'] == 'yes' else 0
    basement = 1 if data['basement'] == 'yes' else 0

    # Make prediction
    prediction = model.predict([[area, bedrooms, stories, mainroad, basement]])

    # Return prediction as JSON
    return jsonify({'predicted_price': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

```

Step 5: test flask application
```ruby
python app.py
```
```ruby
curl -X POST -H "Content-Type: application/json" -d '{"area":1000,"bedrooms":2,"stories":1,"mainroad":"yes","basement":"no"}' http://localhost:5000/predict 
```

Step 6: Create Dockerfile
```ruby
nano Dockerfile
```

```ruby
# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install Flask and scikit-learn
RUN pip install -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python", "app.py"]

```

Step 7: Build and push image to dockerhub
```ruby
docker build -t model_image .
docker tag model_image shubhpatil95/model_image
docker push shubhpatil95/model_image
docker run -p 5000:5000 model_image
```
Step 8: create deployment.yaml

```ruby
nano deployment.yaml
```
```ruby
apiVersion: apps/v1
kind: Deployment
metadata:
  name: house-price-api
spec:
  replicas: 2
  selector:
    matchLabels:
      app: house-price-api
  template:
    metadata:
      labels:
        app: house-price-api
    spec:
      containers:
      - name: house-price-api
        image: shubhpatil95/model_image
        ports:
        - containerPort: 5000
```

Step 9: create service.yaml

```ruby
nano service.yaml
```
```ruby
apiVersion: v1
kind: Service
metadata:
  name: house-price-api
spec:
  selector:
    app: house-price-api
  ports:
    - protocol: TCP
      port: 80
      targetPort: 5000
  type: ClusterIP
```


Step 10: Port forwarding
```ruby
kubectl port-forward service/house-price-api 8080:80
```

Step 11: Access curl
```ruby
curl -X POST -H "Content-Type: application/json" -d '{"area":1000,"bedrooms":2,"stories":1,"mainroad":"yes","basement":"no"}' http://localhost:8080/predict
```
</p>
