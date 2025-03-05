import pandas as pd
from sqlalchemy import create_engine, exc
from sqlalchemy.types import Integer, String, Float
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def fetch_data_from_db():
    try:
        # SQLAlchemy connection string
        engine = create_engine('mysql+mysqlconnector://root:sagar123@localhost:3306/sagarsam')

        # SQL query to fetch data from the table
        query = 'SELECT * FROM Kaggle_Employee_DB'

        # Execute the query and fetch data
        df = pd.read_sql(query, engine)

        # Print success message
        print("Data fetched successfully!")

        return df

    except exc.SQLAlchemyError as err:
        print(f"Error: {err}")
        return None

def model_attrition_probability(data):
    # Data preprocessing
    data['Attrition'] = data['Attrition'].map({'Yes': 1, 'No': 0})

    # Select features and target
    features = ['Age', 'DistanceFromHome', 'EnvironmentSatisfaction', 'JobSatisfaction', 'MonthlyIncome']
    target = 'Attrition'

    X = data[features]
    y = data[target]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define the hyperparameter grid to search
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization parameter
        'solver': ['newton-cg', 'lbfgs', 'liblinear'],  # Optimization algorithm
        'max_iter': [100, 200, 400, 800]  # Maximum iterations
    }

    # Create a logistic regression model instance
    LR = LogisticRegression(random_state=86)  # Keep random state for reproducibility

    # Create a GridSearchCV object
    grid_search = GridSearchCV(LR, param_grid, cv=5)  # Use F1 macro for imbalanced classes

    # Fit the grid search to the training data
    grid_search.fit(X_train_scaled, y_train)

    # Get the best model and its parameters
    best_model = grid_search.best_estimator_

    # Predict probabilities on the entire dataset
    X_scaled = scaler.transform(X)
    data['Attrition_Probability'] = best_model.predict_proba(X_scaled)[:, 1] * 100

    # Return the updated DataFrame
    return data

def truncate_table(engine):
    try:
        # SQL query to truncate the table
        truncate_query = 'TRUNCATE TABLE Test_Attrition_Database'

        # Execute the query
        with engine.connect() as connection:
            connection.execute(truncate_query)

        # Print success message
        print("Test_Attrition_Database table truncated successfully!")

    except exc.SQLAlchemyError as err:
        print(f"Error: {err}")

def insert_data_to_new_table(data, engine):
    try:
        # Calculate the Attrition_Probability using the model
        data = model_attrition_probability(data)

        # Define the table columns and their data types
        data_columns = [
            'Age', 'Attrition', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome',
            'Education', 'EducationField', 'EmployeeCount', 'EmployeeNumber', 'EnvironmentSatisfaction',
            'Gender', 'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction',
            'MaritalStatus', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'Over18', 'OverTime',
            'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction', 'StandardHours',
            'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
            'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager',
            'Attrition_Probability'
        ]

        # Convert dataframe columns to match MySQL table
        data = data[data_columns]

        # Define column types for insertion
        column_types = {
            'Age': Integer(),
            'Attrition': String(3),
            'BusinessTravel': String(50),
            'DailyRate': Integer(),
            'Department': String(50),
            'DistanceFromHome': Integer(),
            'Education': Integer(),
            'EducationField': String(50),
            'EmployeeCount': Integer(),
            'EmployeeNumber': Integer(),
            'EnvironmentSatisfaction': Integer(),
            'Gender': String(10),
            'HourlyRate': Integer(),
            'JobInvolvement': Integer(),
            'JobLevel': Integer(),
            'JobRole': String(50),
            'JobSatisfaction': Integer(),
            'MaritalStatus': String(20),
            'MonthlyIncome': Integer(),
            'MonthlyRate': Integer(),
            'NumCompaniesWorked': Integer(),
            'Over18': String(3),
            'OverTime': String(3),
            'PercentSalaryHike': Integer(),
            'PerformanceRating': Integer(),
            'RelationshipSatisfaction': Integer(),
            'StandardHours': Integer(),
            'StockOptionLevel': Integer(),
            'TotalWorkingYears': Integer(),
            'TrainingTimesLastYear': Integer(),
            'WorkLifeBalance': Integer(),
            'YearsAtCompany': Integer(),
            'YearsInCurrentRole': Integer(),
            'YearsSinceLastPromotion': Integer(),
            'YearsWithCurrManager': Integer(),
            'Attrition_Probability': Float()
        }

        # Insert data into the Test_Attrition_Database table
        # data.to_sql('Test_Attrition_Database', con=engine, if_exists='append', index=False, dtype=column_types)

        # Print success message
        print("Data inserted successfully into Test_Attrition_Database!")

    except exc.SQLAlchemyError as err:
        print(f"Error: {err}")

if __name__ == '__main__':
    engine = create_engine('mysql+mysqlconnector://root:sagar123@localhost:3306/sagarsam')
    data = fetch_data_from_db()
    if data is not None:
        truncate_table(engine)
        insert_data_to_new_table(data, engine)
        print(data.head())  # Display the first few rows of the updated DataFrame