import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pgmpy.estimators import HillClimbSearch
from pgmpy.parameter_estimator.discrete_mle import DiscreteMLE
from pgmpy.models import DiscreteBayesianNetwork

# 1. Data Extraction ------------------------------------------------------------------------------------------------------------
print("Loading Yelp business dataset...")

# Reads json data into DataFrame
df_raw = pd.read_json('yelp_academic_dataset_business.json', lines=True)

# Filter to only restaurants
df_filtered = df_raw.dropna(subset=['categories']).copy()
df_filtered = df_filtered[df_filtered['categories'].str.contains('Restaurants')]


# Extracting attributes
print("Extracting attributes...")

def get_attribute(attr_dict, key):
    if isinstance(attr_dict, dict) and key in attr_dict:
        val = attr_dict[key]
        if val == 'None' or val is None:
            return np.nan
        return str(val) 
    return np.nan

# Extract specific features that could affect star rating
features_to_extract = [
    'RestaurantsPriceRange2', 
    'OutdoorSeating', 
    'RestaurantsDelivery', 
    'HasTV',
    'BikeParking'
]

for feature in features_to_extract:
    df_filtered[feature] = df_filtered['attributes'].apply(lambda x: get_attribute(x, feature))


print("Discretizing continuous variables...")

# Divide review count to 3 buckets
df_filtered['review_count_bin'] = pd.qcut(
    df_filtered['review_count'], 
    q=3, 
    labels=['Low', 'Medium', 'High']
).astype(str)

# Stars treated as strings
df_filtered['stars'] = df_filtered['stars'].astype(str)


# Remove only relevant data
columns_for_bn = ['stars', 'review_count_bin'] + features_to_extract
df_clean = df_filtered[columns_for_bn].copy()

# Drop rows with missing values
df = df_clean.dropna().reset_index(drop=True)

print(f"Data preparation complete! Final dataset shape: {df.shape}")
print("Sample of the cleaned data ready for the Network:")
print(df.head())

# 2. Data Split ------------------------------------------------------------------------------------------------

# Hold out 20% of the data for the final test set
train_data, test_data = train_test_split(df, test_size=0.20, random_state=42)

# We will predict 'stars', so separate it from the test features
test_features = test_data.drop(columns=['stars'])
test_labels = test_data['stars']

# 3. EXPERIMENT SETUP ------------------------------------------------------------------------------------------------

# The increments of training data we want to test
data_fractions = [0.10, 0.25, 0.50, 0.75, 1.0]

# Lists to store our metrics for plotting later
accuracies = []
shd_scores = []
previous_edges = set()

print("Starting Bayesian Network Data Scaling Experiment...\n")

# 4. Training and eval loop -------------------------------------------------------------------------------------
for frac in data_fractions:
    print(f"--- Training on {frac * 100}% of data ---")
    
    # Sample the specific fraction of the training data
    if frac == 1.0:
        sample_train = train_data
    else:
        sample_train = train_data.sample(frac=frac, random_state=42)
        
    # Use Hill Climb Search with BIC scoring to find the best graph structure
    hc = HillClimbSearch(sample_train)
    best_model_structure = hc.estimate(scoring_method='bic-d')
    
    current_edges = set(best_model_structure.edges())
    
    # Calculate Structural Hamming Distance (SHD) 
    if frac == 0.10:
        shd = 0 # Baseline, nothing to compare to yet
    else:
        shd = len(current_edges.symmetric_difference(previous_edges))
    
    shd_scores.append(shd)
    previous_edges = current_edges
    print(f"Learned {len(current_edges)} edges. SHD from previous: {shd}")
    
    # Create Bayesian Network from the learned structure
    bn_model = DiscreteBayesianNetwork(list(current_edges))

    
    # Fit the Conditional Probability Tables (CPTs) using the training data
    bn_model.fit(sample_train, estimator=DiscreteMLE())

    
    # Predict the 'stars' node on the 20% holdout test set
    # Note: pgmpy's predict function can take a moment on large datasets
    predictions = bn_model.predict(test_features)
    
    # Calculate accuracy
    acc = accuracy_score(test_labels, predictions['stars'])
    accuracies.append(acc)
    print(f"Test Set Accuracy: {acc:.4f}\n")

# 5. Results ------------------------------------------------------------------------------------

print("=== FINAL EXPERIMENT RESULTS ===")
print(f"Data Fractions Used: {data_fractions}")
print(f"Graph Stability (SHD): {shd_scores}")
print(f"Predictive Accuracies: {accuracies}")
