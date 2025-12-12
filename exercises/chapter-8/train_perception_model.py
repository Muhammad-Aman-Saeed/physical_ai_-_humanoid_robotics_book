"""
Exercise: Training a Perception Model
Chapter 8: Machine Learning for Physical Systems

This exercise demonstrates how to train a perception model for robotics applications.
The task is to train a model that can classify objects based on sensor data.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt


def generate_robot_sensor_data(n_samples=1000):
    """
    Generate synthetic robot sensor data for object classification.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        X: Sensor readings (features)
        y: Object labels (targets)
    """
    np.random.seed(42)
    
    # Define object types
    object_types = ['cube', 'sphere', 'cylinder', 'pyramid']
    n_objects = len(object_types)
    
    # Generate features based on object characteristics
    X = []
    y = []
    
    for i in range(n_samples):
        obj_type = i % n_objects  # Cycle through object types
        
        # Generate sensor readings based on object type
        if obj_type == 0:  # Cube
            # Cube: more uniform sensor readings
            sensor_readings = np.random.normal([0.8, 0.8, 0.8], [0.1, 0.1, 0.1], size=(10,))
        elif obj_type == 1:  # Sphere
            # Sphere: more curved surface readings
            sensor_readings = np.random.normal([0.5, 0.5, 0.5], [0.2, 0.2, 0.2], size=(10,))
        elif obj_type == 2:  # Cylinder
            # Cylinder: intermediate characteristics
            sensor_readings = np.random.normal([0.7, 0.7, 0.6], [0.15, 0.15, 0.15], size=(10,))
        else:  # Pyramid
            # Pyramid: more variable readings
            sensor_readings = np.random.normal([0.6, 0.4, 0.7], [0.2, 0.25, 0.15], size=(10,))
        
        # Add noise to make it more realistic
        sensor_readings += np.random.normal(0, 0.05, size=(10,))
        
        X.append(sensor_readings)
        y.append(object_types[obj_type])
    
    return np.array(X), np.array(y)


def extract_features(sensor_data):
    """
    Extract meaningful features from raw sensor data.
    
    Args:
        sensor_data: Raw sensor readings
        
    Returns:
        features: Extracted features
    """
    # For each sensor reading, compute various statistics
    features = []
    
    for reading in sensor_data:
        # Statistical features
        mean_val = np.mean(reading)
        std_val = np.std(reading)
        min_val = np.min(reading)
        max_val = np.max(reading)
        range_val = max_val - min_val
        median_val = np.median(reading)
        
        # Shape-related features
        # These could be based on how sensor readings vary across the object
        gradient = np.gradient(reading)
        gradient_magnitude = np.mean(np.abs(gradient))
        
        features.append([
            mean_val, std_val, min_val, max_val, range_val, 
            median_val, gradient_magnitude
        ])
    
    return np.array(features)


def train_perception_model():
    """
    Train a perception model to classify objects based on sensor data.
    
    This function performs the following steps:
    1. Generate synthetic sensor data
    2. Extract relevant features
    3. Train a classifier
    4. Evaluate the model
    """
    print("Training Perception Model for Robotic Object Classification")
    print("=" * 60)
    
    # Step 1: Generate sensor data
    print("Step 1: Generating synthetic sensor data...")
    X_raw, y = generate_robot_sensor_data(n_samples=1000)
    print(f"Generated {len(X_raw)} samples of sensor data")
    
    # Step 2: Extract features
    print("\nStep 2: Extracting features from sensor data...")
    X = extract_features(X_raw)
    print(f"Extracted {X.shape[1]} features from raw sensor data")
    
    # Step 3: Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Step 4: Train classifier
    print("\nStep 3: Training Random Forest classifier...")
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train, y_train)
    
    # Step 5: Evaluate model
    print("\nStep 4: Evaluating model performance...")
    y_pred_train = classifier.predict(X_train)
    y_pred_test = classifier.predict(X_test)
    
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    print(f"Training Accuracy: {train_accuracy:.3f}")
    print(f"Test Accuracy: {test_accuracy:.3f}")
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred_test))
    
    # Step 6: Feature importance analysis
    print("\nStep 5: Analyzing feature importance...")
    feature_names = [
        'Mean', 'Std Dev', 'Min', 'Max', 'Range', 
        'Median', 'Gradient Magnitude'
    ]
    
    importances = classifier.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("Feature Importance Ranking:")
    for i in range(len(feature_names)):
        print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.3f}")
    
    # Step 7: Visualization
    print("\nGenerating visualizations...")
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Feature importance
    plt.subplot(1, 3, 1)
    plt.bar(range(len(feature_names)), importances[indices])
    plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=45)
    plt.title('Feature Importance')
    plt.ylabel('Importance')
    
    # Plot 2: Confusion matrix (simplified)
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    plt.subplot(1, 3, 2)
    cm = confusion_matrix(y_test, y_pred_test)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Plot 3: Accuracy comparison
    plt.subplot(1, 3, 3)
    plt.bar(['Training', 'Test'], [train_accuracy, test_accuracy])
    plt.title('Accuracy Comparison')
    plt.ylabel('Accuracy')
    for i, v in enumerate([train_accuracy, test_accuracy]):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    plt.show()
    
    return classifier, feature_names


def test_new_objects(classifier, feature_names):
    """
    Test the trained model on new object measurements.
    
    Args:
        classifier: Trained classifier
        feature_names: Names of the features
    """
    print("\nTesting on new object measurements...")
    
    # Create some new test measurements
    new_measurements = [
        # Cube-like object
        [0.81, 0.09, 0.72, 0.89, 0.17, 0.81, 0.02],  
        # Sphere-like object  
        [0.52, 0.18, 0.31, 0.72, 0.41, 0.51, 0.08],
        # Cylinder-like object
        [0.71, 0.12, 0.58, 0.85, 0.27, 0.70, 0.05],
    ]
    
    for i, measurement in enumerate(new_measurements):
        prediction = classifier.predict([measurement])[0]
        probabilities = classifier.predict_proba([measurement])[0]
        
        print(f"\nTest Object {i+1}:")
        print(f"  Sensor Features: {dict(zip(feature_names, measurement))}")
        print(f"  Predicted Object: {prediction}")
        print(f"  Prediction Probabilities: {dict(zip(classifier.classes_, probabilities))}")


def main():
    """Main function to run the perception model exercise."""
    print("Exercise: Training a Perception Model for Robotics")
    print("Chapter 8: Machine Learning for Physical Systems")
    print()
    
    # Train the perception model
    classifier, feature_names = train_perception_model()
    
    # Test on new objects
    test_new_objects(classifier, feature_names)
    
    print("\n" + "="*60)
    print("Exercise Summary:")
    print("- Learned to extract meaningful features from raw sensor data")
    print("- Trained a classifier to recognize different object types")
    print("- Evaluated model performance and analyzed feature importance")
    print("- Applied the model to classify new objects")
    print("="*60)


if __name__ == "__main__":
    main()