# kubernetes_seminar

<h4>Before starting with this project, make sure you have docker for desktop is installed and kubernetes is enabled there as shown in below image1</h4>

<h4>Also verify if minikube is installed and it should nbe started using below command</h4>
<p>
```ruby
minikube start
```
<br>[Click Here To Go To Image1](https://github.com/ShubhPatil95/kubernetes_seminar/assets/74223025/29e58edf-13a0-4648-a013-89d9fa5773d3)</br>


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


</p>
