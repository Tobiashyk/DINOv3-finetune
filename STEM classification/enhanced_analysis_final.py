#!/usr/bin/env python3
"""
Enhanced Analysis with More Epochs - Final Version
Multiple epochs with comprehensive text-based analysis
"""

import os
import torch
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import json
from collections import defaultdict

# Set up paths
REPO_DIR = "../dinov3"
MODEL_PATH = "../dinov3/weight/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"
IMAGE_DIR = "./image"

def load_dinov3_vits16plus_model():
    """Load DINOv3 vits16plus model"""
    print("Loading DINOv3 vits16plus model...")
    
    try:
        # Try loading vits16plus first
        model = torch.hub.load(
            REPO_DIR, 
            'dinov3_vits16plus', 
            source='local', 
            weights=MODEL_PATH
        )
    except Exception as e:
        print(f"Failed to load vits16plus: {e}")
        # Fall back to vits16
        model = torch.hub.load(
            REPO_DIR, 
            'dinov3_vits16', 
            source='local', 
            weights=MODEL_PATH
        )
    
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    
    print("DINOv3 model loaded successfully!")
    return model

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess image for DINOv3 input"""
    image = Image.open(image_path).convert('RGB')
    image = image.resize(target_size)
    image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_tensor = (image_tensor - mean) / std
    return image_tensor.unsqueeze(0)

def extract_single_feature(model, image_path):
    """Extract feature for a single image"""
    tensor = preprocess_image(image_path)
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    
    with torch.no_grad():
        feature = model(tensor)
    
    return feature.cpu().numpy().flatten()

def load_training_data():
    """Load training data from train_pos_1T and train_pos_2H"""
    print("Loading training data from train_pos_1T and train_pos_2H...")
    
    train_1t_dir = os.path.join(IMAGE_DIR, "train_pos_1T")
    train_2h_dir = os.path.join(IMAGE_DIR, "train_pos_2H")
    
    # Get all image paths
    train_1t_images = [os.path.join(train_1t_dir, f) for f in os.listdir(train_1t_dir) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    train_2h_images = [os.path.join(train_2h_dir, f) for f in os.listdir(train_2h_dir) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Create labels (0 for 1T, 1 for 2H)
    all_images = train_1t_images + train_2h_images
    all_labels = [0] * len(train_1t_images) + [1] * len(train_2h_images)
    
    print(f"Found {len(train_1t_images)} 1T training images and {len(train_2h_images)} 2H training images")
    print(f"Total training images: {len(all_images)}")
    
    return all_images, all_labels

def load_validation_data():
    """Load validation data from validate_pos"""
    print("Loading validation data from validate_pos...")
    
    val_1t_dir = os.path.join(IMAGE_DIR, "validate_pos", "1T")
    val_2h_dir = os.path.join(IMAGE_DIR, "validate_pos", "2H")
    
    val_1t_images = [os.path.join(val_1t_dir, f) for f in os.listdir(val_1t_dir) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    val_2h_images = [os.path.join(val_2h_dir, f) for f in os.listdir(val_2h_dir) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    val_images = val_1t_images + val_2h_images
    val_labels = [0] * len(val_1t_images) + [1] * len(val_2h_images)
    
    print(f"Found {len(val_1t_images)} validation 1T images and {len(val_2h_images)} validation 2H images")
    print(f"Total validation images: {len(val_images)}")
    
    return val_images, val_labels

def train_many_epochs(X_train, y_train, n_epochs=20):
    """Train many epochs with different random states"""
    print(f"\n=== Training {n_epochs} Epochs ===")
    
    epoch_results = []
    trained_models = []
    
    for epoch in range(n_epochs):
        print(f"\n--- Epoch {epoch + 1}/{n_epochs} ---")
        
        # Use different random state for each epoch
        model = RandomForestClassifier(
            n_estimators=150,
            max_depth=18,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight='balanced',
            random_state=42 + epoch,  # Different random state each epoch
            n_jobs=2
        )
        
        model.fit(X_train, y_train)
        trained_models.append(model)
        
        # Evaluate on training data
        train_pred = model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_pred)
        
        epoch_results.append({
            'epoch': epoch + 1,
            'train_accuracy': train_accuracy,
            'model': model
        })
        
        print(f"Epoch {epoch + 1} - Training Accuracy: {train_accuracy:.4f}")
    
    return epoch_results, trained_models

def create_large_ensemble(trained_models, X_train, y_train):
    """Create large ensemble from many epoch models"""
    print("\n=== Creating Large Ensemble ===")
    
    # Use voting classifier for ensemble
    from sklearn.ensemble import VotingClassifier
    
    estimators = [(f'Epoch_{i+1}', model) for i, model in enumerate(trained_models)]
    ensemble = VotingClassifier(
        estimators=estimators,
        voting='soft',
        n_jobs=2
    )
    
    ensemble.fit(X_train, y_train)
    print(f"Large ensemble created with {len(trained_models)} models!")
    return ensemble

def detailed_classification_analysis(models, X_test, y_test, test_images, dataset_name="Test"):
    """Detailed classification analysis with per-image results"""
    print(f"\n=== Detailed {dataset_name} Classification Analysis ===")
    
    # Get predictions from ensemble
    if isinstance(models, dict) and 'Ensemble' in models:
        ensemble_pred = models['Ensemble'].predict(X_test)
        ensemble_proba = models['Ensemble'].predict_proba(X_test)
    else:
        # Use the first model if no ensemble dict
        ensemble_pred = models[0].predict(X_test)
        ensemble_proba = models[0].predict_proba(X_test)
    
    # Individual model predictions
    individual_predictions = {}
    for i, model in enumerate(models):
        if isinstance(model, RandomForestClassifier):
            pred = model.predict(X_test)
            individual_predictions[f'Epoch_{i+1}'] = pred
    
    # Detailed results
    detailed_results = []
    correct_count = 0
    
    for i, (img_path, true_label, pred_label, proba) in enumerate(zip(test_images, y_test, ensemble_pred, ensemble_proba)):
        true_name = "1T" if true_label == 0 else "2H"
        pred_name = "1T" if pred_label == 0 else "2H"
        correct = true_label == pred_label
        confidence = max(proba)
        
        if correct:
            correct_count += 1
        
        # Get individual model predictions
        individual_preds = {}
        for epoch_name, preds in individual_predictions.items():
            individual_preds[epoch_name] = "1T" if preds[i] == 0 else "2H"
        
        result = {
            'image': os.path.basename(img_path),
            'true_label': true_name,
            'predicted_label': pred_name,
            'correct': str(correct),  # Convert to string for JSON serialization
            'confidence': float(confidence),
            'individual_predictions': individual_preds,
            'probability_1T': float(proba[0]),
            'probability_2H': float(proba[1])
        }
        
        detailed_results.append(result)
        
        # Print detailed result
        status = "✓" if correct else "✗"
        print(f"  {status} Image {i+1}: {os.path.basename(img_path)}")
        print(f"    True: {true_name}, Predicted: {pred_name}, Confidence: {confidence:.3f}")
        if not correct:
            print(f"    Individual predictions: {individual_preds}")
    
    accuracy = correct_count / len(y_test)
    print(f"\nDetailed Analysis - {dataset_name} Accuracy: {accuracy:.1%}")
    print(f"Correct predictions: {correct_count}/{len(y_test)}")
    
    return detailed_results, accuracy

def create_text_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix"):
    """Create text-based confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    
    print(f"\n{title}")
    print("=" * 50)
    print(f"{'':>10} {'Predicted':>20}")
    print(f"{'Actual':>10} {class_names[0]:>10} {class_names[1]:>10}")
    print("-" * 35)
    print(f"{class_names[0]:>10} {cm[0,0]:>10} {cm[0,1]:>10}")
    print(f"{class_names[1]:>10} {cm[1,0]:>10} {cm[1,1]:>10}")
    print("-" * 35)
    print(f"Accuracy: {accuracy:.1%}")
    print("=" * 50)
    
    return cm, accuracy

def create_text_epoch_performance(epoch_results):
    """Create text-based epoch performance visualization"""
    print("\n" + "=" * 60)
    print("TRAINING ACCURACY ACROSS EPOCHS")
    print("=" * 60)
    
    for result in epoch_results:
        epoch = result['epoch']
        accuracy = result['train_accuracy']
        bar_length = int(accuracy * 40)  # Scale to 40 characters
        bar = "█" * bar_length + "░" * (40 - bar_length)
        print(f"Epoch {epoch:2d}: [{bar}] {accuracy:.1%}")
    
    print("=" * 60)
    print(f"Average Training Accuracy: {sum(r['train_accuracy'] for r in epoch_results)/len(epoch_results):.1%}")
    print("=" * 60)

def main():
    """Main execution function"""
    print("=== Enhanced Analysis with More Epochs - Final Version ===")
    print("Using train_pos_1T, train_pos_2H for training")
    print("Using validate_pos for testing")
    print("Running more epochs with comprehensive text-based analysis")
    
    # Load DINOv3 vits16plus model
    dinov3_model = load_dinov3_vits16plus_model()
    
    # Load training data
    train_images, train_labels = load_training_data()
    
    # Load validation data
    val_images, val_labels = load_validation_data()
    
    # Extract features efficiently
    print("\n--- Extracting features from training data ---")
    train_features = []
    for i, img_path in enumerate(train_images):
        if i % 5 == 0 or i + 1 == len(train_images):
            print(f"  Processing {i+1}/{len(train_images)} training images")
        
        feature = extract_single_feature(dinov3_model, img_path)
        train_features.append(feature)
    
    train_features = np.array(train_features)
    
    print("\n--- Extracting features from validation data ---")
    val_features = []
    for i, img_path in enumerate(val_images):
        if i % 3 == 0 or i + 1 == len(val_images):
            print(f"  Processing {i+1}/{len(val_images)} validation images")
        
        feature = extract_single_feature(dinov3_model, img_path)
        val_features.append(feature)
    
    val_features = np.array(val_features)
    
    # Split training data for internal validation (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        train_features, train_labels, test_size=0.2, random_state=42, stratify=train_labels
    )
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Internal test set: {len(X_test)} samples")
    print(f"External validation set: {len(val_features)} samples")
    
    # Train many epochs
    print("\n=== Training Many Epochs ===")
    n_epochs = 30  # Increased to 30 epochs for maximum analysis
    epoch_results, trained_models = train_many_epochs(X_train, y_train, n_epochs=n_epochs)
    
    # Create large ensemble
    print("\n=== Creating Large Ensemble ===")
    ensemble = create_large_ensemble(trained_models, X_train, y_train)
    
    # Evaluate each epoch individually
    print("\n=== Individual Epoch Evaluations ===")
    for epoch_result in epoch_results:
        print(f"Epoch {epoch_result['epoch']}: Training Accuracy = {epoch_result['train_accuracy']:.4f}")
    
    # Create text-based epoch performance visualization
    create_text_epoch_performance(epoch_results)
    
    # Detailed analysis on test set
    print("\n=== Detailed Test Set Analysis ===")
    test_detailed_results, test_accuracy = detailed_classification_analysis(
        trained_models, X_test, y_test, 
        [f"test_img_{i}" for i in range(len(y_test))], "Test"
    )
    
    # Create text-based confusion matrix for test set
    test_pred = trained_models['Ensemble'].predict(X_test) if isinstance(trained_models, dict) and 'Ensemble' in trained_models else trained_models[0].predict(X_test)
    test_cm, test_cm_accuracy = create_text_confusion_matrix(y_test, test_pred, ['1T', '2H'], "Test Set Confusion Matrix")
    
    # Detailed analysis on validation set
    print("\n=== Detailed Validation Set Analysis ===")
    val_detailed_results, val_accuracy = detailed_classification_analysis(
        trained_models, val_features, val_labels, val_images, "Validation"
    )
    
    # Create text-based confusion matrix for validation set
    val_pred = trained_models['Ensemble'].predict(val_features) if isinstance(trained_models, dict) and 'Ensemble' in trained_models else trained_models[0].predict(val_features)
    val_cm, val_cm_accuracy = create_text_confusion_matrix(val_labels, val_pred, ['1T', '2H'], "Validation Set Confusion Matrix")
    
    # Save detailed results
    detailed_results = {
        'epoch_results': [{'epoch': r['epoch'], 'train_accuracy': r['train_accuracy']} for r in epoch_results],
        'test_detailed_results': test_detailed_results,
        'validation_detailed_results': val_detailed_results,
        'test_accuracy': test_accuracy,
        'validation_accuracy': val_accuracy,
        'test_confusion_matrix': test_cm.tolist(),
        'validation_confusion_matrix': val_cm.tolist(),
        'feature_extractor': 'DINOv3_vits16plus',
        'n_epochs': n_epochs
    }
    
    # Save the ensemble model
    model_save_path = "enhanced_analysis_final_vits16plus.pkl"
    joblib.dump({
        'ensemble_model': ensemble,
        'individual_models': trained_models,
        'test_accuracy': test_accuracy,
        'validation_accuracy': val_accuracy,
        'feature_extractor': 'DINOv3_vits16plus',
        'detailed_results': detailed_results
    }, model_save_path)
    
    print(f"\nEnhanced analysis model saved to {model_save_path}")
    
    # Save detailed results summary
    results_summary = {
        'model_type': 'Enhanced_Analysis_Final_vits16plus',
        'feature_extractor': 'DINOv3_vits16plus',
        'n_epochs': n_epochs,
        'test_accuracy': test_accuracy,
        'validation_accuracy': val_accuracy,
        'test_confusion_matrix_accuracy': test_cm_accuracy,
        'validation_confusion_matrix_accuracy': val_cm_accuracy,
        'individual_epoch_accuracies': [r['train_accuracy'] for r in epoch_results],
        'test_detailed_analysis': test_detailed_results,
        'validation_detailed_analysis': val_detailed_results
    }
    
    with open('enhanced_analysis_final_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print("\n=== Enhanced Analysis Complete! ===")
    print(f"Test Accuracy: {test_accuracy:.1%}")
    print(f"Validation Accuracy: {val_accuracy:.1%}")
    print(f"Total samples processed: {len(train_features) + len(val_features)}")
    print(f"Comprehensive text-based analysis completed for all {len(val_images)} validation images")
    print(f"Confusion matrices and detailed per-image results saved")
    
    return test_accuracy, val_accuracy, detailed_results

if __name__ == "__main__":
    main()
