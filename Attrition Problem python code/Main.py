from flask import Flask, render_template, jsonify
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import mysql.connector

app = Flask(__name__)

def fetch_data_from_db():
    # MySQL connection parameters
    connection = mysql.connector.connect(
        host='localhost',
        user='root',
        password='sagar123',
        port='3306',
        database='sagarsam'
    )

    # SQL query to fetch data from the table
    query = 'SELECT * FROM Attrition_Database'

    # Execute the query and fetch data
    df = pd.read_sql(query, connection)

    # Clean up: close cursor and connection
    connection.close()

    return df

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/data')
def get_data():
    df = fetch_data_from_db()
    df['Is Eligible For Promotion'] = 'Yes'  # Ensuring this is still here
    # Add dummy data for buttons just for consistency, not necessary though
    df['Assign Training'] = ''
    records = df[['EmployeeNumber', 'Attrition_Probability', 'Department', 'JobInvolvement', 'Is Eligible For Promotion', 'Assign Training']].to_dict(orient='records')
    return jsonify({"data": records})

@app.route('/graphs/<category>')
def get_graphs(category):
    df = fetch_data_from_db()

    # Data preprocessing
    df.drop(columns=['EmployeeCount', 'Over18', 'StandardHours'], inplace=True)

    graphs = []

    if category == 'JobSatisfaction':
        # Job Satisfaction Distribution
        attrition_counts = df['JobSatisfaction'].value_counts()
        labels = [f'{label} ({count})\n{count/sum(attrition_counts)*100:.1f}%' for label, count in attrition_counts.items()]
        plt.figure(figsize=(10, 6))
        plt.pie(attrition_counts, labels=labels, autopct='%1.1f%%', startangle=90, explode=[0.04,0.02,0.02,0.02], textprops={"fontsize":14})
        plt.title('Job Satisfaction Distribution', fontsize=16, weight='bold')
        plt.axis('equal')
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf8')
        graphs.append(img_base64)
        plt.close()

        # Attrition with Job Satisfaction
        plt.figure(figsize=(10, 6))
        ax = sns.countplot(x='JobSatisfaction', hue='Attrition', data=df, palette='Set2')
        plt.xlabel("JobSatisfaction", fontsize=14)
        plt.ylabel("Number of Employees", fontsize=14)
        plt.title(label='Attrition with Job Satisfaction', fontsize=16, weight='bold')
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'{height}', (p.get_x() + p.get_width() / 2., height), ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=10, color='black', weight='bold')
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf8')
        graphs.append(img_base64)
        plt.close()

    elif category == 'Promotion':
        # Attrition Rate by Years Since Last Promotion
        attrition_rate = (df[df['Attrition'] == 'Yes'].groupby('YearsSinceLastPromotion').size() /
                         df.groupby('YearsSinceLastPromotion').size()) * 100  # Convert to percentage
        plt.figure(figsize=(10, 6))  # Adjust figure size for better visualization
        plt.bar(attrition_rate.index, attrition_rate.values, color='skyblue')
        plt.xlabel("Years Since Last Promotion", fontsize=14)
        plt.ylabel("Attrition Rate (%)", fontsize=14)
        plt.title("Attrition Rate by Years Since Last Promotion", fontsize=16, weight='bold')
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf8')
        graphs.append(img_base64)
        plt.close()

        # Attrition with Years Since Last Promotion
        plt.figure(figsize=(15, 6))  # Adjust figure size for better visualization
        ax = sns.countplot(x='YearsSinceLastPromotion', hue='Attrition', data=df, palette='Set2')
        plt.xlabel("Years Since Last Promotion", fontsize=14)
        plt.ylabel("Number of Employees", fontsize=14)
        plt.title("Attrition with Years Since Last Promotion", fontsize=16, weight='bold')
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'{height}', (p.get_x() + p.get_width() / 2., height), ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=10, color='black', weight='bold')
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf8')
        graphs.append(img_base64)
        plt.close()

    elif category == 'OverTime':
        # OverTime Distribution
        percentages = (df[df['Attrition'] == 'Yes']['OverTime'].value_counts() / df['OverTime'].value_counts()) * 100
        plt.figure(figsize=(10, 6))
        percentages.plot(kind='pie', autopct='%1.1f%%', startangle=90, explode=[0.04, 0.02], textprops={'fontsize': 14})
        plt.title('Percentage of Employee Attrition wrt OverTime', fontsize=16, weight='bold')
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf8')
        graphs.append(img_base64)
        plt.close()

        # Attrition with OverTime
        plt.figure(figsize=(15, 6))  # Adjust figure size for better visualization
        ax = sns.countplot(x='OverTime', hue='Attrition', data=df, palette='Set2')
        plt.xlabel("OverTime", fontsize=14)
        plt.ylabel("Number of Employees", fontsize=14)
        plt.title("Attrition with OverTime", fontsize=16, weight='bold')
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'{height}', (p.get_x() + p.get_width() / 2., height), ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=10, color='black', weight='bold')
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf8')
        graphs.append(img_base64)
        plt.close()

    elif category == 'MaritalStatus':
        # Marital Status Distribution
        #percentages = df['MaritalStatus'].value_counts(normalize=True) * 100
        percentages = df[(df["Attrition"] == 'Yes')].groupby(["MaritalStatus"]).size() / df.groupby(["MaritalStatus"]).size()
        explode = [0.02] * len(percentages)
        plt.figure(figsize=(10, 6))
        plt.pie(percentages, explode=explode, labels=percentages.index, autopct="%1.1f%%", textprops={"fontsize":14})
        plt.title("Percentage of Employee Attrition wrt Marital Status", fontsize=16, weight='bold')
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf8')
        graphs.append(img_base64)
        plt.close()

    elif category == 'YearsInCurrentRole':
        # Attrition Rate by Years in Current Role
        x = df.groupby(df["YearsInCurrentRole"])["Attrition"].value_counts().reset_index(name="count")
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=x, x="YearsInCurrentRole", y="count", hue="Attrition")
        plt.title("Attrition Rate by Years in Current Role", fontsize=16, weight='bold')
        plt.xlabel("Years in Current Role", fontsize=14)
        plt.ylabel("Number of Employees", fontsize=14)
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf8')
        graphs.append(img_base64)
        plt.close()

        # Attrition with Years in Current Role
        plt.figure(figsize=(15, 6))
        plt.title("Attrition with Years in Current Role", fontsize=16, weight='bold')
        ax = sns.countplot(x='YearsInCurrentRole', hue='Attrition', data=df, palette='Set2')
        plt.xlabel("Years in Current Role", fontsize=14)
        plt.ylabel("Number of Employees", fontsize=14)
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'{height}', (p.get_x() + p.get_width() / 2., height), ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=10, color='black', weight='bold')
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf8')
        graphs.append(img_base64)
        plt.close()

    return jsonify(graphs)

if __name__ == '__main__':
    app.run(debug=True)