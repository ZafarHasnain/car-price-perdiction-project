import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

# 1. Load the data
car = pd.read_csv('Cleaned_Car_data.csv')

# 2. Separate Features and Target
X = car[['name', 'company', 'year', 'kms_driven', 'fuel_type']]
y = car['Price']

# 3. Create OneHotEncoder and fit it to get the categories
# This ensures the model knows all the possible car names, companies, etc.
ohe = OneHotEncoder()
ohe.fit(X[['name', 'company', 'fuel_type']])

# 4. Create the Column Transformer
column_trans = make_column_transformer(
    (OneHotEncoder(categories=ohe.categories_), ['name', 'company', 'fuel_type']),
    remainder='passthrough'
)

# 5. Train the model
# We loop 1000 times to find the random state that gives the best split (highest accuracy), just like in your notebook.
scores = []
for i in range(1000):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=i)
    lr = LinearRegression()
    pipe = make_pipeline(column_trans, lr)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    scores.append(r2_score(y_test, y_pred))

best_random_state = np.argmax(scores)
print(f"Best random state found: {best_random_state} with R2 score: {scores[best_random_state]}")

# 6. Train the final model using the best random state
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=best_random_state)
lr = LinearRegression()
pipe = make_pipeline(column_trans, lr)
pipe.fit(X_train, y_train)

# 7. Save the new model
pickle.dump(pipe, open('LinearRegressionModel.pkl', 'wb'))
print("Success! Model retrained and saved as 'LinearRegressionModel.pkl'")