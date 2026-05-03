import pandas as pd
import random
from sklearn.tree import DecisionTreeClassifier

def train_and_predict_best_deal(current_data_df):
    """
    Trains a simple Decision Tree model on a generated dataset
    to predict which platform gives the best deal, then applies
    it to the current search data.
    """
    # 1. Generate training dataset (simulating historical data)
    train_data = []
    for _ in range(200):
        price = random.uniform(1000, 100000)
        rating = random.uniform(3.0, 5.0)
        discount = random.randint(0, 30)
        
        # Simple logic to create "labels" for the training data
        # Best deals generally have higher ratings, lower relative prices, and higher discounts
        score = (rating * 10) + discount - (price * 0.0001)
        
        # Label: 1 if it's considered a "Good Deal", 0 if not
        label = 1 if score > 40 else 0
        train_data.append([price, rating, discount, label])

    train_df = pd.DataFrame(train_data, columns=['Price', 'Rating', 'Discount', 'Label'])

    # 2. Train the Model
    X = train_df[['Price', 'Rating', 'Discount']]
    y = train_df['Label']

    model = DecisionTreeClassifier(max_depth=3)
    model.fit(X, y)

    # 3. Predict on Current Data
    X_current = current_data_df[['Price (₹)', 'Rating', 'Discount (%)']]
    X_current.columns = ['Price', 'Rating', 'Discount'] # Match training feature names
    
    # Predict probabilities of being a "Good Deal" (class 1)
    probabilities = model.predict_proba(X_current)[:, 1]
    
    # Add score to dataframe
    current_data_df['Deal_Score'] = probabilities
    
    # Find the row with the highest deal score
    # If there's a tie, fallback to lowest price
    best_deal_idx = current_data_df.sort_values(by=['Deal_Score', 'Price (₹)'], ascending=[False, True]).index[0]
    best_row = current_data_df.loc[best_deal_idx]
    
    return best_row, model
