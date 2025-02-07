import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pypickle
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

class AdvancedNetworkAnomalyDetector:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.models = {}

    def preprocess_data(self, data):
        """Enhanced data preprocessing with advanced feature engineering"""
        # Numerical features
        numerical_features = [
            'Duration', 'PacketCount', 'ByteCount',
            'SourcePort', 'DestinationPort'
        ]

        # Feature engineering
        data['BytesPerPacket'] = data['ByteCount'] / (data['PacketCount'] + 1)
        data['PacketRate'] = data['PacketCount'] / data['Duration']
        data['ByteRate'] = data['ByteCount'] / data['Duration']

        # Interaction features
        data['Duration_PacketCount_Ratio'] = data['Duration'] / (data['PacketCount'] + 1)

        # Categorical feature encoding
        categorical_features = ['Protocol']
        for feature in categorical_features:
            le = LabelEncoder()
            data[f'{feature}_encoded'] = le.fit_transform(data[feature])
            self.label_encoders[feature] = le

        # Combining features
        feature_columns = (
            numerical_features +
            [f'{f}_encoded' for f in categorical_features] +
            ['BytesPerPacket', 'PacketRate', 'ByteRate',
             'Duration_PacketCount_Ratio']
        )

        # Preparing the data
        X = data[feature_columns]
        y = (data['Label'] == 'Attack').astype(int)

        # Scaling the features
        X_scaled = self.scaler.fit_transform(X)

        return X_scaled, y

    def build_lstm_model(self, input_shape):
        """LSTM model with improved architecture"""
        model = Sequential([
            LSTM(128, input_shape=(input_shape[1], 1), return_sequences=True),
            Dropout(0.3),
            LSTM(64, return_sequences=False),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def build_autoencoder(self, input_shape):
        """Autoencoder for anomaly detection"""
        input_layer = Input(shape=(input_shape[1],))

        # Encoder
        encoded = Dense(128, activation='relu')(input_layer)
        encoded = Dropout(0.2)(encoded)
        encoded = Dense(64, activation='relu')(encoded)
        encoded = Dropout(0.2)(encoded)
        encoded = Dense(32, activation='relu')(encoded)

        # Decoder
        decoded = Dense(64, activation='relu')(encoded)
        decoded = Dropout(0.2)(decoded)
        decoded = Dense(128, activation='relu')(decoded)
        decoded = Dense(input_shape[1], activation='sigmoid')(decoded)

        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        return autoencoder

    def train_models(self, X_train, y_train, X_val, y_val):
        """Training multiple models"""
        # Calculating class weights
        class_weights = class_weight.compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weights_dict = dict(enumerate(class_weights))

        # Early stopping and learning rate reduction
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        lr_reducer = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5
        )

        # Isolation Forest
        self.models['isolation_forest'] = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        self.models['isolation_forest'].fit(X_train)

        # Random Forest
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42
        )
        self.models['random_forest'].fit(X_train, y_train)

        # LSTM Model
        X_train_lstm = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_val_lstm = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)

        self.models['lstm'] = self.build_lstm_model(X_train.shape)
        self.models['lstm'].fit(
            X_train_lstm, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_val_lstm, y_val),
            class_weight=class_weights_dict,
            callbacks=[early_stopping, lr_reducer],
            verbose=1
        )

        # Autoencoder
        self.models['autoencoder'] = self.build_autoencoder(X_train.shape)
        self.models['autoencoder'].fit(
            X_train, X_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_val, X_val),
            callbacks=[early_stopping],
            verbose=1
        )

        # Calculating reconstruction threshold for autoencoder
        reconstructed = self.models['autoencoder'].predict(X_train)
        mse = np.mean(np.power(X_train - reconstructed, 2), axis=1)
        self.reconstruction_threshold = np.percentile(mse, 95)

    def ensemble_predict(self, X):
        """Ensembling prediction with weighted voting"""
        # Isolation Forest prediction
        if_pred = self.models['isolation_forest'].predict(X)
        if_pred = (if_pred == -1).astype(int)

        # Random Forest prediction
        rf_pred = self.models['random_forest'].predict(X)

        # LSTM prediction
        X_lstm = X.reshape(X.shape[0], X.shape[1], 1)
        lstm_pred = (self.models['lstm'].predict(X_lstm) > 0.5).astype(int)

        # Autoencoder prediction
        reconstructed = self.models['autoencoder'].predict(X)
        mse = np.mean(np.power(X - reconstructed, 2), axis=1)
        ae_pred = (mse > self.reconstruction_threshold).astype(int)

        # Weighted voting
        ensemble_pred = np.column_stack([if_pred, rf_pred, lstm_pred.ravel(), ae_pred])
        weights = [0.25, 0.25, 0.25, 0.25]  # Balanced weights

        # Weighted soft voting
        weighted_votes = ensemble_pred * weights
        return (weighted_votes.sum(axis=1) > np.mean(weights)).astype(int)

def plot_confusion_matrix(y_true, y_pred):
    """Visualizing confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

def main():
    # Loading the data
    print("Loading data...")
    data = pd.read_csv('data/network_traffic_data.csv')

    # Initializing the detector
    detector = AdvancedNetworkAnomalyDetector()

    # Preprocessing the data
    print("Preprocessing data...")
    X, y = detector.preprocess_data(data)

    # Splitting the dataset with stratification
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )

    # Training the models
    print("Training models...")
    detector.train_models(X_train, y_train, X_val, y_val)

    # Making predictions
    print("Making predictions...")
    predictions = detector.ensemble_predict(X_test)

    # Evaluating the results
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    # Visualizing the confusion matrix
    plot_confusion_matrix(y_test, predictions)

if __name__ == "__main__":
    main()
