#!/usr/bin/env python3
"""
Training script to create ML model and pickle files for emotion detection
"""

import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """Load and preprocess the trading data"""
    print("📊 Loading CSV data...")
    df = pd.read_csv('./data/Book1.csv')
    
    # Clean up the data
    print(f"📈 Loaded {len(df)} trading records")
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort by wallet and timestamp to ensure proper order
    df = df.sort_values(['wallet', 'timestamp']).reset_index(drop=True)
    
    # Handle invalid values in amount_out
    df['amount_out'] = pd.to_numeric(df['amount_out'], errors='coerce')
    df['amount_out'].fillna(df['amount_out'].median(), inplace=True)
    
    print("✅ Data preprocessing completed")
    return df

def engineer_features(df):
    """Engineer features for emotion detection"""
    print("🔧 Engineering features...")
    
    # Group by wallet to calculate features per user
    wallet_dfs = []
    
    for wallet in df['wallet'].unique():
        wallet_df = df[df['wallet'] == wallet].copy()
        
        if len(wallet_df) < 3:  # Need at least 3 trades for features
            continue
            
        # Time-based features
        wallet_df['time_diff'] = wallet_df['timestamp'].diff().dt.total_seconds() / 60
        wallet_df['time_diff'].fillna(wallet_df['time_diff'].median(), inplace=True)
        
        # Price and volume features
        wallet_df['slippage_factor'] = 0.99
        wallet_df['price_in'] = (wallet_df['volume_usd'] / wallet_df['amount_in']) * wallet_df['slippage_factor']
        wallet_df['price_out'] = (wallet_df['volume_usd'] / wallet_df['amount_out']) * wallet_df['slippage_factor']
        
        # Market price (rolling average)
        wallet_df['market_price'] = wallet_df['price_out'].rolling(window=min(10, len(wallet_df))).mean()
        wallet_df['market_price'].fillna(wallet_df['price_out'], inplace=True)
        
        # Price change percentage
        wallet_df['price_change_pct'] = ((wallet_df['price_out'] - wallet_df['market_price']) / wallet_df['market_price']) * 100
        wallet_df['price_change_pct'].fillna(0, inplace=True)
        
        # Position and leverage features
        wallet_df['account_equity'] = 1000  # Assumed base equity
        wallet_df['leverage'] = (wallet_df['amount_in'] * wallet_df['price_in']) / wallet_df['account_equity']
        wallet_df['position_change'] = wallet_df['leverage'].pct_change()
        wallet_df['position_change'].fillna(0, inplace=True)
        
        # PnL calculation
        wallet_df['entry_price'] = wallet_df.groupby('in_token')['price_in'].shift(1)
        wallet_df['exit_price'] = wallet_df['price_out']
        wallet_df['pnl'] = (wallet_df['exit_price'] - wallet_df['entry_price']) * wallet_df['amount_out'] * 0.997
        wallet_df['pnl'].fillna(0, inplace=True)
        
        # Win/Loss streaks
        wallet_df['is_win'] = wallet_df['pnl'] > 0
        wallet_df['is_loss'] = wallet_df['pnl'] < 0
        
        # Calculate consecutive wins/losses
        wallet_df['win_streak'] = (wallet_df['pnl'] > 0).groupby((wallet_df['pnl'] <= 0).cumsum()).cumcount() + 1
        wallet_df['loss_streak'] = (wallet_df['pnl'] < 0).groupby((wallet_df['pnl'] >= 0).cumsum()).cumcount() + 1
        
        # Consecutive patterns
        wallet_df['consecutive_losses'] = (wallet_df['pnl'] < 0).rolling(window=3).sum()
        wallet_df['consecutive_wins'] = (wallet_df['pnl'] > 0).rolling(window=3).sum()
        
        wallet_dfs.append(wallet_df)
    
    # Combine all wallet dataframes
    final_df = pd.concat(wallet_dfs, ignore_index=True)
    print(f"✅ Feature engineering completed for {len(final_df)} records")
    return final_df

def detect_emotions(df):
    """Detect emotions based on trading patterns"""
    print("🎭 Detecting emotions...")
    
    df['emotion'] = 'neutral'
    df['trigger_details'] = None
    
    for i in range(1, len(df)):
        prev_row = df.iloc[i-1]
        curr_row = df.iloc[i]
        
        # FOMO: Rapid trading after price spikes with winning streak
        if (abs(curr_row['price_change_pct']) > 15 and 
            curr_row['time_diff'] < 30 and 
            curr_row['win_streak'] > 1):
            df.loc[i, 'emotion'] = 'fomo'
            df.loc[i, 'trigger_details'] = f"Price spike: {curr_row['price_change_pct']:.2f}%, win streak: {curr_row['win_streak']}"
        
        # REVENGE: Quick trades after consecutive losses
        elif (curr_row['time_diff'] < 5 and 
              prev_row['consecutive_losses'] >= 2):
            df.loc[i, 'emotion'] = 'revenge'
            df.loc[i, 'trigger_details'] = f"After {prev_row['consecutive_losses']} losses, latency: {curr_row['time_diff']}"
        
        # GREED: Position size increase after wins
        elif (curr_row['position_change'] > 0.3 and 
              prev_row['consecutive_wins'] >= 2):
            df.loc[i, 'emotion'] = 'greed'
            df.loc[i, 'trigger_details'] = f"After {prev_row['consecutive_wins']} wins, pos change: {curr_row['position_change']:.2f}"
        
        # FEAR: Quick exits with small profits
        elif (curr_row['time_diff'] < 2 and 
              curr_row['pnl'] > 0 and 
              abs(curr_row['pnl']) < 50):
            df.loc[i, 'emotion'] = 'fear'
            df.loc[i, 'trigger_details'] = f"Early close: {curr_row['pnl']:.2f}, latency: {curr_row['time_diff']}"
    
    emotion_counts = df['emotion'].value_counts()
    print("📊 Emotion distribution:")
    for emotion, count in emotion_counts.items():
        print(f"   {emotion}: {count}")
    
    return df

def prepare_ml_dataset(df):
    """Prepare dataset for ML training"""
    print("🤖 Preparing ML dataset...")
    
    # Select features for ML
    feature_cols = [
        'time_diff', 'price_change_pct', 'position_change', 
        'consecutive_wins', 'consecutive_losses', 'win_streak', 'loss_streak'
    ]
    
    # Create samples using sliding window of last 3 trades
    ml_data = []
    
    grouped = df.groupby('wallet')
    for wallet, group in grouped:
        group = group.sort_values('timestamp').reset_index(drop=True)
        
        for i in range(3, len(group)):
            # Use previous 3 trades as features
            prev3 = group.iloc[i-3:i]
            current = group.iloc[i]
            
            if len(prev3) == 3 and current['emotion'] != 'neutral':
                sample = {
                    'time_diff_mean': prev3['time_diff'].mean(),
                    'price_change_pct_mean': prev3['price_change_pct'].mean(),
                    'position_change_mean': prev3['position_change'].mean(),
                    'consecutive_wins': prev3['consecutive_wins'].iloc[-1],
                    'consecutive_losses': prev3['consecutive_losses'].iloc[-1],
                    'win_streak': prev3['win_streak'].iloc[-1],
                    'loss_streak': prev3['loss_streak'].iloc[-1],
                    'emotion': current['emotion']
                }
                ml_data.append(sample)
    
    ml_df = pd.DataFrame(ml_data)
    print(f"✅ Created {len(ml_df)} training samples")
    return ml_df

def train_model(ml_df):
    """Train the emotion prediction model"""
    print("🚀 Training emotion prediction model...")
    
    # Prepare features and labels
    feature_cols = [
        'time_diff_mean', 'price_change_pct_mean', 'position_change_mean',
        'consecutive_wins', 'consecutive_losses', 'win_streak', 'loss_streak'
    ]
    
    X = ml_df[feature_cols]
    y = ml_df['emotion']
    
    # Handle any remaining NaN values
    X = X.fillna(0)
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Train Random Forest model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✅ Model trained with accuracy: {accuracy:.3f}")
    print("\n📈 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    return model, le

def save_model_artifacts(model, le, df):
    """Save model and related artifacts as pickle files"""
    print("💾 Saving model artifacts...")
    
    # Save trained model
    joblib.dump(model, './data/emotion_predictor.pkl')
    print("✅ Saved: ./data/emotion_predictor.pkl")
    
    # Save label encoder
    joblib.dump(le, './data/label_encoder.pkl')
    print("✅ Saved: ./data/label_encoder.pkl")
    
    # Save sample features for testing
    sample_features = df[['time_diff', 'price_change_pct', 'position_change', 
                         'consecutive_wins', 'consecutive_losses', 'win_streak', 'loss_streak']].tail(100)
    joblib.dump(sample_features.to_numpy(), 'sample_features.pkl')
    print("✅ Saved: sample_features.pkl")
    
    print("🎉 All model artifacts saved successfully!")

def main():
    """Main training pipeline"""
    print("🚀 Starting ML model training pipeline...")
    print("=" * 50)
    
    try:
        # Step 1: Load and preprocess data
        df = load_and_preprocess_data()
        
        # Step 2: Engineer features
        df = engineer_features(df)
        
        # Step 3: Detect emotions
        df = detect_emotions(df)
        
        # Step 4: Prepare ML dataset
        ml_df = prepare_ml_dataset(df)
        
        if len(ml_df) < 10:
            print("❌ Not enough emotional trading samples found for training")
            print("💡 Consider adjusting emotion detection thresholds")
            return
        
        # Step 5: Train model
        model, le = train_model(ml_df)
        
        # Step 6: Save artifacts
        save_model_artifacts(model, le, df)
        
        print("=" * 50)
        print("✅ Training pipeline completed successfully!")
        print("📦 Ready for deployment with the following files:")
        print("   - emotion_predictor.pkl")
        print("   - label_encoder.pkl")
        print("   - sample_features.pkl")
        
    except Exception as e:
        print(f"❌ Error in training pipeline: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
