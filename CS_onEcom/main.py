import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import hdbscan

df = pd.read_csv('Pakistan Largest Ecommerce Dataset.csv', low_memory=False)

df.shape

# Set max columns to display all columns
pd.set_option('display.max_columns', None)
print(df.head())

# Display summary statistics of dataset
df.describe()

# Display information about the DataFrame, including the number of non-null entries, data types and size.
df.info()

# Show all column names
df.columns

# Renaming the columns for better clarity and consistency in the dataset
df.rename(columns = {
    'status': 'order_status',
    'created_at': 'order_date',
    'sku': 'product_id',
    'qty_ordered' : 'order_quantity',
    'category_name_1': 'category',
    'Working Date': 'working_date',
    'BI Status': 'BI_status',
    ' MV ': 'market_value', 
    'Year': 'year',
    'Month': 'month',
    'Customer Since': 'customer_since',
    'M-Y': 'month_year',
    'FY': 'financial_year',
    'Customer ID': 'customer_id'
}, inplace=True)

# Lets See Renamed columns
print("Data After Renaming Columns:")
print(df.columns)

# Clean the dataset
df_cleaned = df.drop(columns=['Unnamed: 21', 'Unnamed: 22', 'Unnamed: 23', 'Unnamed: 24', 'Unnamed: 25'])
num_cols = ['price', 'order_quantity', 'grand_total', 'discount_amount', 'market_value']

# ........................Data Preprocessing..................
# Handling Missing values
numeric_cols = df.select_dtypes(include=['number']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
print("Data After handling Missing values:")
print(df[numeric_cols])

# Let's see the last completely empty columns
df.iloc[:, -5:]

# Now lets check the missing values
df.isnull().sum().sort_values(ascending=False)

# I am removing "sales_commission_code" column because it has 601229 missing values and is not so important.
df.drop(columns='sales_commission_code', inplace=True)

# Removing empty rows based on the "sku" and "Customer ID" columns. The "sku" column cannot be filled because each item has a unique SKU number.
df.dropna(subset=['product_id', 'customer_id'], inplace=True)

# Filling missing values in these columns with their most frequent values (mode).
df[['category', 'customer_since', 'order_status']] = df[['category', 'customer_since', 'order_status']].fillna(df[['category', 'customer_since', 'order_status']].mode().iloc[0])

df.isnull().sum()

# Lets See all unique categories
df['category'].unique()

# Finding Top 10 ordered categories in Pakistan
top_cat = df.groupby('category')['order_quantity'].sum().sort_values(ascending=False).reset_index().head(10)
top_cat

# Top 10 ordered categories in Pakistan
plt.figure(figsize=(16,6))
plt.xticks(rotation=45, fontsize=12)
fig = sns.barplot(data = top_cat, x= 'category', y = 'order_quantity', palette = 'viridis')
plt.xlabel('Name of Categories', size=16)
plt.ylabel('No of Orders', size=16)
plt.title('Names of Categories vs. Number of Times Each Product was Ordered', size=16)
for bars in fig.containers:
    fig.bar_label(bars, fontsize=12)    
plt.show()

# Unique values in payment_method Columns
df['payment_method'].unique()

# 'year' column to integers using .loc
df.loc[:, 'year'] = df['year'].astype(int)
# Check the year with the highest number of orders
year_sales = df.groupby(['year'])['order_quantity'].count().sort_values(ascending=False).reset_index()
year_sales

# Pie Chart plot to Display Yearly Sales Trends
plt.pie(year_sales['order_quantity'], labels=year_sales['year'], autopct='%1.1f%%', startangle=140)
plt.title('Sales Distribution by Year')
plt.axis('equal')
plt.show()


# Extract the day from 'order_date' column
df.loc[:,'date'] = df['order_date'].dt.day
day_wise_sales = df.groupby('date')['order_quantity'].sum().sort_values(ascending=False).reset_index().head(10)
day_wise_sales


# Plot to visualize the day wise oder quantity in months
plt.figure(figsize=(10,6))
sns.lineplot(x='date', y='order_quantity', data=day_wise_sales, marker='o', color='blue')
plt.title('Day-Wise Order Quantity', fontsize=16)
plt.xlabel('Day', fontsize=12)
plt.ylabel('Order Quantity', fontsize=12)
plt.show()


# Group by customer_id and count the number of items ordered per customer.
customer_orders = df.groupby('customer_id')['item_id'].count().value_counts(ascending=False).reset_index(name = 'customers')
customer_orders.head(10)


#......................Data Normalization and Feature Scaling.................

# Correlation between features
plt.figure(figsize = (16,10))
sns.heatmap(df.corr(numeric_only = True),cmap = "coolwarm", annot = True)
plt.show()

# Standardize numerical columns
numerical_cols = ['order_quantity', 'market_value']
scaler_standard = StandardScaler()
df[numerical_cols] = scaler_standard.fit_transform(df[numerical_cols])


# Normalize data for clustering using Min-Max scaling
scaler_minmax = MinMaxScaler()
df[numerical_cols] = scaler_minmax.fit_transform(df[numerical_cols])


# Encoding Categorical Variables
from sklearn.preprocessing import LabelEncoder
categorical_cols = ['category', 'order_status', 'BI_status']
encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col])

# Feature Engineering
# Extracting useful date features
df['order_date'] = pd.to_datetime(df['order_date'])
df['order_weekday'] = df['order_date'].dt.day_name()
df['order_month'] = df['order_date'].dt.month
df['order_year'] = df['order_date'].dt.year

# Customer lifetime value approximation
df['customer_lifetime'] = df['year'] - pd.to_datetime(df['customer_since']).dt.year

# .....................Machine Learning Models....................

#  Clustering Model (K-Means Clustering)
X = df[['order_quantity', 'market_value', 'customer_lifetime']]
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X)
plt.scatter(df['order_quantity'], df['market_value'], c=df['cluster'], cmap='viridis')
plt.xlabel('Order Quantity')
plt.ylabel('Market Value')
plt.title('K-Means Clustering')
plt.show()
print("Cluster Centers:", kmeans.cluster_centers_)


#  Anomaly Detection using Isolation Forest
X = df[['order_quantity', 'market_value']]
# Fit the Isolation Forest model
iso_forest = IsolationForest(random_state=42)
df['anomaly'] = iso_forest.fit_predict(X)
# Anomalies are labeled as -1, normal data points are labeled as 1
anomalies = df[df['anomaly'] == -1]
print("Anomalies Detected:\n", anomalies)


# Customer Segmentation (K-Means)
X = df[['order_quantity', 'market_value', 'customer_lifetime']]
kmeans = KMeans(n_clusters=4, random_state=42)
df['customer_segment'] = kmeans.fit_predict(X)
# Display the clusters
print("Customer Segmentation Results:\n", df[['customer_id', 'customer_segment']].head())
plt.scatter(df['order_quantity'], df['market_value'], c=df['customer_segment'], cmap='viridis')
plt.xlabel('Order Quantity')
plt.ylabel('Market Value')
plt.title('Customer Segmentation using K-Means')
plt.show()


# Logistic Regression
selected_features = ['order_quantity', 'market_value', 'customer_lifetime'] 
target_col = 'order_status'  

if all(col in df.columns for col in selected_features) and target_col in df.columns:
    X = df[selected_features]
    y = df[target_col]
else:
    raise ValueError("Selected features or target column are not in the DataFrame.")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

logistic_model = LogisticRegression(random_state=42)
logistic_model.fit(X_train, y_train)
y_pred = logistic_model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))


# PCA for Dimensionality Reduction
pca = PCA(n_components=2)
df_pca = pca.fit_transform(X)
plt.scatter(df_pca[:, 0], df_pca[:, 1], c=y, cmap='viridis')
plt.title("PCA Visualization")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()


# Random Forest Classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
# Predictions
y_pred_rf = rf_model.predict(X_test)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {accuracy * 100:.2f}%")
# Generate confusion matrix and plot
all_classes = sorted(y.unique()) 
cm = confusion_matrix(y_test, y_pred_rf, labels=all_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=all_classes)
disp.plot(cmap='viridis')
plt.title("Confusion Matrix - Random Forest")
plt.show()


# DB Scan Clustering
clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
df['hdbscan_cluster'] = clusterer.fit_predict(X)
print("HDBSCAN Clustering Results:")
print(df['hdbscan_cluster'].value_counts())

# Monthly sales Trend
df['order_month'] = pd.to_datetime(df['order_date']).dt.to_period('M')
monthly_sales = df.groupby('order_month')['order_quantity'].sum()
plt.figure(figsize=(12, 6))
monthly_sales.plot()
plt.title("Monthly Sales Trends")
plt.xlabel("Month")
plt.ylabel("Sales Quantity")
plt.show()


# Plot Kernel Density Distribution
plt.figure(figsize=(20, 16))
for i, col in enumerate(num_cols):
    plt.subplot(2, 3, i + 1)
    sns.kdeplot(data=df_cleaned, x=col, color="deepskyblue")
    plt.title(f"Kernel Density Distribution of {col}", fontsize=20)
    plt.xlabel(f"{col}", fontsize=20)
    plt.ylabel("Kernel Density", fontsize=20)
    plt.tick_params(axis="both", labelsize=15)

plt.tight_layout(w_pad=2, h_pad=4)
plt.show()


# Payment Method Contribution to Total Revenue
plt.figure(figsize = (12,6))
best_payment_method = df.groupby("payment_method")["grand_total"].sum().sort_values()
best_payment_method.plot(kind = "barh",edgecolor = "black" , color = "honeydew")
plt.title("Payment Method Contribution to Total Revenue",fontsize = 15)
plt.ylabel("Payment Method",fontsize = 15)
plt.xlabel("Grand Total( Revenue (PKR) )",fontsize = 15)
plt.show()


















