# %%
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import psutil
import time
import keras_tuner as kt
import shap

# %%

# Load and preprocess the dataset
data = pd.read_csv('UNSW_NB15_training-set.csv')

# Print the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# %%
# Print the shape of the dataset
print("\nShape of the dataset:")
print(data.shape)

# %%

# Print descriptive statistics
print("\nDescriptive statistics:")
print(data.describe())

# %%
# Plot the distribution of each feature
data.hist(figsize=(20, 20))
plt.tight_layout()
plt.show()

# %%
# Handle non-numeric columns by encoding them
data_encoded = pd.get_dummies(data)

# %%
# Print the first few rows of the encoded dataset
print("\nFirst few rows of the encoded dataset:")
print(data_encoded.head())

# %%

# Plot the correlation heatmap
plt.figure(figsize=(15, 15))
sns.heatmap(data_encoded.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# %%
# Plot the distribution of each feature
data.hist(figsize=(20, 20))
plt.tight_layout()
plt.show()

# %%
# Filter numerical features
data_numeric = data.select_dtypes(include=['float64', 'int64'])

# Define label generation criteria
def generate_labels(row):
    return 1 if row['label'] == 1 else 0

# Apply label generation function to create labels
data_numeric['label'] = data_numeric.apply(generate_labels, axis=1)

# Split features and labels
labels = data_numeric['label'].values
features = data_numeric.drop('label', axis=1)

# Normalize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Reshape features for CNN
n_samples, n_features = features_scaled.shape
features_reshaped = features_scaled.reshape(n_samples, n_features, 1)

# Split data into train and test sets
train_features, test_features, train_labels, test_labels = train_test_split(features_reshaped, labels, test_size=0.2, random_state=42)


# %%

# Define the HyperModel for Keras Tuner
class CNNHyperModel(kt.HyperModel):
    def build(self, hp):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv1D(filters=hp.Int('filters', min_value=16, max_value=128, step=16),
                                         kernel_size=hp.Int('kernel_size', min_value=2, max_value=5, step=1),
                                         activation='relu',
                                         input_shape=train_features.shape[1:]))
        model.add(tf.keras.layers.MaxPooling1D(pool_size=hp.Int('pool_size', min_value=2, max_value=4, step=1)))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(units=hp.Int('units', min_value=64, max_value=256, step=32), activation='relu'))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

# Initialize the RandomSearch Tuner
hypermodel = CNNHyperModel()
tuner = kt.RandomSearch(
    hypermodel,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=2,
    directory='hyperparam_tuning_advanced',
    project_name='unsw_nb15_cnn_advanced'
)

# Split training data into training and validation sets
train_features, val_features, train_labels, val_labels = train_test_split(train_features, train_labels, test_size=0.2, random_state=42)

# Run the hyperparameter search
tuner.search(train_features, train_labels, epochs=10, validation_data=(val_features, val_labels))

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The optimal number of filters in the first convolution layer is {best_hps.get('filters')},
the optimal kernel size is {best_hps.get('kernel_size')},
the optimal pool size is {best_hps.get('pool_size')},
and the optimal number of units in the dense layer is {best_hps.get('units')}.
""")

# %%

# Train the baseline model with the best parameters
def create_cnn_model(input_shape, filters, kernel_size, pool_size, units):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=pool_size),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer with sigmoid activation for binary classification
    ])
    return model

input_shape = train_features.shape[1:]
baseline_model = create_cnn_model(input_shape, 
                                  filters=best_hps.get('filters'), 
                                  kernel_size=best_hps.get('kernel_size'), 
                                  pool_size=best_hps.get('pool_size'), 
                                  units=best_hps.get('units'))

baseline_model.compile(optimizer='adam',
                       loss='binary_crossentropy',  # Binary crossentropy loss for binary classification
                       metrics=['accuracy'])

# Manually set the number of epochs
epochs = 10

# Train and Evaluate the Baseline Model
def train_and_evaluate(model, train_features, train_labels, test_features, test_labels, epochs=10):
    start_time = time.time()
    history = model.fit(train_features, train_labels, epochs=epochs, validation_data=(test_features, test_labels))
    end_time = time.time()
    
    # Measure memory usage
    process = psutil.Process()
    memory_usage = process.memory_info().rss / (1024 ** 2)  # in MB
    
    # Measure inference time and CPU utilization
    inference_start_time = time.time()
    cpu_percent_start = psutil.cpu_percent(interval=None)
    model.predict(test_features[:100])  # Predict on a small batch
    cpu_percent_end = psutil.cpu_percent(interval=None)
    inference_end_time = time.time()
    inference_time = inference_end_time - inference_start_time
    cpu_utilization = (cpu_percent_end + cpu_percent_start) / 2  # Average CPU utilization
    
    return history, end_time - start_time, memory_usage, inference_time, cpu_utilization

# Train and evaluate the baseline model
baseline_history, train_time, memory_usage, inference_time, cpu_utilization = train_and_evaluate(baseline_model, train_features, train_labels, test_features, test_labels, epochs=epochs)

# Plot training and validation accuracy and loss
plt.figure(figsize=(12, 6))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(baseline_history.history['loss'], label='Training Loss')
plt.plot(baseline_history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(baseline_history.history['accuracy'], label='Training Accuracy')
plt.plot(baseline_history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Save the baseline model
baseline_model.save('model_baseline.h5')

# Evaluate Models
def evaluate_model(model, test_features, test_labels):
    predictions = (model.predict(test_features) > 0.5).astype("int32")
    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions)
    recall = recall_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions)
    roc_auc = roc_auc_score(test_labels, predictions)
    cm = confusion_matrix(test_labels, predictions)
    fpr, tpr, _ = roc_curve(test_labels, predictions)
    return accuracy, precision, recall, f1, roc_auc, cm, fpr, tpr, predictions

baseline_metrics = evaluate_model(baseline_model, test_features, test_labels)
baseline_accuracy, baseline_precision, baseline_recall, baseline_f1, baseline_roc_auc, baseline_cm, baseline_fpr, baseline_tpr, baseline_predictions = baseline_metrics

# Log metrics for the baseline model
baseline_metrics_df = pd.DataFrame({
    'model': ['baseline'],
    'train_time': [train_time],
    'memory_usage': [memory_usage],
    'inference_time': [inference_time],
    'cpu_utilization': [cpu_utilization],
    'accuracy': [baseline_accuracy],
    'precision': [baseline_precision],
    'recall': [baseline_recall],
    'f1_score': [baseline_f1],
    'roc_auc': [baseline_roc_auc]
})


# %%

# Static Pruning Based on Percentiles
def prune_model(model, pruning_percentage):
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv1D) or isinstance(layer, tf.keras.layers.Dense):
            weights, biases = layer.get_weights()
            flattened_weights = np.abs(weights).flatten()
            percentile = np.percentile(flattened_weights, pruning_percentage)
            pruned_weights = np.where(np.abs(weights) < percentile, 0, weights)
            layer.set_weights([pruned_weights, biases])

pruning_percentages = [10, 20, 30, 40, 50, 60, 70, 80, 90]
all_metrics = []
pruned_metrics = []
model_sizes = []

for pruning_percentage in pruning_percentages:
    pruned_model = create_cnn_model(input_shape, 
                                    filters=best_hps.get('filters'), 
                                    kernel_size=best_hps.get('kernel_size'), 
                                    pool_size=best_hps.get('pool_size'), 
                                    units=best_hps.get('units'))
    pruned_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    prune_model(pruned_model, pruning_percentage)
    
    # Retrain pruned model
    history_pruned, train_time_pruned, memory_usage_pruned, inference_time_pruned, cpu_utilization_pruned = train_and_evaluate(pruned_model, train_features, train_labels, test_features, test_labels, epochs=epochs)
    pruned_metrics.append(evaluate_model(pruned_model, test_features, test_labels))
    
    accuracy_pruned, precision_pruned, recall_pruned, f1_pruned, roc_auc_pruned, pruned_cm, pruned_fpr, pruned_tpr, pruned_predictions = pruned_metrics[-1]
    model_metrics = {
        'model': [f'pruned_{pruning_percentage}%'],
        'train_time': [train_time_pruned],
        'memory_usage': [memory_usage_pruned],
        'inference_time': [inference_time_pruned],
        'cpu_utilization': [cpu_utilization_pruned],
        'accuracy': [accuracy_pruned],
        'precision': [precision_pruned],
        'recall': [recall_pruned],
        'f1_score': [f1_pruned],
        'roc_auc': [roc_auc_pruned]
    }
    
    all_metrics.append(model_metrics)

    # Save the pruned model
    pruned_model.save(f'model_{pruning_percentage}_pruned.h5')
    
    # Log model size
    model_size = pruned_model.count_params()
    model_sizes.append((pruning_percentage, model_size))

# Combine metrics for static pruning
static_pruning_metrics_df = pd.concat([baseline_metrics_df] + [pd.DataFrame(m) for m in all_metrics], ignore_index=True)
static_pruning_metrics_df.to_csv('static_pruning_metrics.csv', index=False)

# Save model sizes to CSV
model_sizes_df = pd.DataFrame(model_sizes, columns=['pruning_percentage', 'model_size'])
model_sizes_df.to_csv('model_sizes.csv', index=False)

# %%
# Adaptive Pruning Strategy
def dynamic_magnitude_pruning(model, initial_threshold, final_threshold, num_iterations, train_features, train_labels, val_features, val_labels):
    threshold = initial_threshold
    step = (final_threshold - initial_threshold) / num_iterations
    
    # Initialize lists to store training and validation loss and accuracy
    training_loss = []
    validation_loss = []
    training_accuracy = []
    validation_accuracy = []
    
    for iteration in range(num_iterations):
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Conv1D) or isinstance(layer, tf.keras.layers.Dense):
                weights, biases = layer.get_weights()
                flattened_weights = np.abs(weights).flatten()
                percentile = np.percentile(flattened_weights, threshold)
                pruned_weights = np.where(np.abs(weights) < percentile, 0, weights)
                layer.set_weights([pruned_weights, biases])
        
        history = model.fit(train_features, train_labels, epochs=1, validation_data=(val_features, val_labels))
        training_loss.append(history.history['loss'][0])
        validation_loss.append(history.history['val_loss'][0])
        training_accuracy.append(history.history['accuracy'][0])
        validation_accuracy.append(history.history['val_accuracy'][0])
        threshold += step
    
    return training_loss, validation_loss, training_accuracy, validation_accuracy

# Create and compile a new model for adaptive pruning
adaptive_model = create_cnn_model(input_shape, 
                                  filters=best_hps.get('filters'), 
                                  kernel_size=best_hps.get('kernel_size'), 
                                  pool_size=best_hps.get('pool_size'), 
                                  units=best_hps.get('units'))

adaptive_model.compile(optimizer='adam',
                       loss='binary_crossentropy',  # Binary crossentropy loss for binary classification
                       metrics=['accuracy'])

# Apply dynamic magnitude pruning and measure metrics
start_time = time.time()
training_loss, validation_loss, training_accuracy, validation_accuracy = dynamic_magnitude_pruning(adaptive_model, initial_threshold=10, final_threshold=90, num_iterations=10, train_features=train_features, train_labels=train_labels, val_features=val_features, val_labels=val_labels)
end_time = time.time()

# Plot training and validation loss and accuracy
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(training_loss) + 1), training_loss, label='Training Loss')
plt.plot(range(1, len(validation_loss) + 1), validation_loss, label='Validation Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training and Validation Loss During Adaptive Pruning')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, len(training_accuracy) + 1), training_accuracy, label='Training Accuracy')
plt.plot(range(1, len(validation_accuracy) + 1), validation_accuracy, label='Validation Accuracy')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy During Adaptive Pruning')
plt.legend()

plt.tight_layout()
plt.show()

# Measure memory usage
process = psutil.Process()
memory_usage_adaptive = process.memory_info().rss / (1024 ** 2)  # in MB

# Measure inference time and CPU utilization
inference_start_time = time.time()
cpu_percent_start = psutil.cpu_percent(interval=None)
adaptive_model.predict(test_features[:100])  # Predict on a small batch
cpu_percent_end = psutil.cpu_percent(interval=None)
inference_end_time = time.time()
inference_time_adaptive = inference_end_time - inference_start_time
cpu_utilization_adaptive = (cpu_percent_end + cpu_percent_start) / 2  # Average CPU utilization

# Save the adaptively pruned model
adaptive_model.save('model_adaptive_pruned.h5')

# Evaluate adaptively pruned model
adaptive_pruned_metrics = evaluate_model(adaptive_model, test_features, test_labels)
adaptive_pruned_accuracy, adaptive_pruned_precision, adaptive_pruned_recall, adaptive_pruned_f1, adaptive_pruned_roc_auc, adaptive_pruned_cm, adaptive_pruned_fpr, adaptive_pruned_tpr, adaptive_pruned_predictions = adaptive_pruned_metrics

# Log metrics for the adaptively pruned model
adaptive_pruned_metrics_df = pd.DataFrame({
    'model': ['adaptive_pruned'],
    'train_time': [end_time - start_time],
    'memory_usage': [memory_usage_adaptive],
    'inference_time': [inference_time_adaptive],
    'cpu_utilization': [cpu_utilization_adaptive],
    'accuracy': [adaptive_pruned_accuracy],
    'precision': [adaptive_pruned_precision],
    'recall': [adaptive_pruned_recall],
    'f1_score': [adaptive_pruned_f1],
    'roc_auc': [adaptive_pruned_roc_auc]
})

# Combine metrics
all_metrics_df = pd.concat([static_pruning_metrics_df, adaptive_pruned_metrics_df], ignore_index=True)
all_metrics_df.to_csv('all_metrics.csv', index=False)


# %%

# Define Visualization Functions
def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, cbar=False)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

def plot_roc_curve(fpr, tpr, title='ROC Curve'):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, marker='.')
    plt.title(title)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()

def plot_inference_time_pie_chart(inference_times, labels, title='Inference Time Distribution'):
    plt.figure(figsize=(8, 8))
    plt.pie(inference_times, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title(title)
    plt.show()

def plot_metrics_bar_chart(metrics_df, metric, title):
    plt.figure(figsize=(12, 6))
    sns.barplot(x='model', y=metric, data=metrics_df, palette='viridis')
    plt.title(title)
    plt.xlabel('Model')
    plt.ylabel(metric.capitalize())
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_metrics_line_chart(metrics_df, metric, title):
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='model', y=metric, data=metrics_df, marker='o')
    plt.title(title)
    plt.xlabel('Model')
    plt.ylabel(metric.capitalize())
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_heatmap(matrix, title='Heatmap', cmap=plt.cm.Blues):
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt='d', cmap=cmap, cbar=False)
    plt.title(title)
    plt.show()

def plot_radar_chart(metrics_df, metrics, title):
    labels = metrics
    num_vars = len(labels)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for i, row in metrics_df.iterrows():
        values = row[metrics].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=row['model'])
        ax.fill(angles, values, alpha=0.25)

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    plt.title(title)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.show()

def plot_scatter(metrics_df, x_metric, y_metric, title):
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=metrics_df, x=x_metric, y=y_metric, hue='model', style='model', palette='viridis')
    plt.title(title)
    plt.xlabel(x_metric.capitalize())
    plt.ylabel(y_metric.capitalize())
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# %%

# Visualization of Metrics
plot_confusion_matrix(baseline_cm, title='Confusion Matrix - Baseline Model')

# Plot confusion matrices for pruned models
for i, pruning_percentage in enumerate(pruning_percentages):
    pruned_accuracy, pruned_precision, pruned_recall, pruned_f1, pruned_roc_auc, pruned_cm, pruned_fpr, pruned_tpr, _ = pruned_metrics[i]
    plot_confusion_matrix(pruned_cm, title=f'Confusion Matrix - Pruned Model ({pruning_percentage}%)')

# Plot confusion matrix for adaptively pruned model
plot_confusion_matrix(adaptive_pruned_cm, title='Confusion Matrix - Adaptively Pruned Model')

# %%

# Plot ROC Curves
plot_roc_curve(baseline_fpr, baseline_tpr, title='ROC Curve - Baseline Model')
for i, pruning_percentage in enumerate(pruning_percentages):
    pruned_accuracy, pruned_precision, pruned_recall, pruned_f1, pruned_roc_auc, pruned_cm, pruned_fpr, pruned_tpr, _ = pruned_metrics[i]
    plot_roc_curve(pruned_fpr, pruned_tpr, title=f'ROC Curve - Pruned Model ({pruning_percentage}%)')
plot_roc_curve(adaptive_pruned_fpr, adaptive_pruned_tpr, title='ROC Curve - Adaptively Pruned Model')

# %%

# Plot Inference Time Distribution
inference_times = [baseline_metrics_df['inference_time'].iloc[0]] + [metrics['inference_time'][0] for metrics in all_metrics]
labels = ['Baseline'] + [f'Pruned {percentage}%' for percentage in pruning_percentages]

# Add the inference time for the adaptively pruned model
inference_times.append(adaptive_pruned_metrics_df['inference_time'].iloc[0])
labels.append('Adaptive Pruned')

# Ensure the labels and inference_times have the same length
if len(inference_times) != len(labels):
    print(f"Error: The number of inference times ({len(inference_times)}) does not match the number of labels ({len(labels)}).")
else:
    plot_inference_time_pie_chart(inference_times, labels, title='Inference Time Distribution')

# %%

# Plot Metrics Bar Charts
metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
for metric in metrics_to_plot:
    plot_metrics_bar_chart(all_metrics_df, metric, title=f'{metric.capitalize()} Across Models')

# %%

# Plot Metrics Line Charts
for metric in metrics_to_plot:
    plot_metrics_line_chart(all_metrics_df, metric, title=f'{metric.capitalize()} Over Models')

# %%

# Plot CPU Utilization
plot_metrics_bar_chart(all_metrics_df, 'cpu_utilization', title='CPU Utilization Across Models')

# %%

# Plot Model Size Reduction
plt.figure(figsize=(12, 6))
sns.barplot(x='pruning_percentage', y='model_size', data=model_sizes_df, palette='viridis')
plt.title('Model Size Reduction After Pruning')
plt.xlabel('Pruning Percentage')
plt.ylabel('Number of Parameters')
plt.tight_layout()
plt.show()

# %%

# Plot Heatmaps of Confusion Matrices
plot_heatmap(baseline_cm, title='Heatmap - Baseline Model')
for i, pruning_percentage in enumerate(pruning_percentages):
    _, _, _, _, _, pruned_cm, _, _, _ = pruned_metrics[i]
    plot_heatmap(pruned_cm, title=f'Heatmap - Pruned Model ({pruning_percentage}%)')
plot_heatmap(adaptive_pruned_cm, title='Heatmap - Adaptively Pruned Model')

# %%

# Plot Radar Charts for Metrics Comparison
plot_radar_chart(all_metrics_df, metrics_to_plot, title='Radar Chart - Metrics Comparison Across Models')

# %%

# Plot Scatter Plots for Metrics Comparison
plot_scatter(all_metrics_df, 'accuracy', 'inference_time', title='Scatter Plot - Accuracy vs Inference Time')
plot_scatter(all_metrics_df, 'accuracy', 'cpu_utilization', title='Scatter Plot - Accuracy vs CPU Utilization')

# %%

# Plot Line Graphs for Inference Time
plot_metrics_line_chart(all_metrics_df, 'inference_time', title='Inference Time Across Models')


# %%
#Function to preprocess data for SHAP
def preprocess_for_shap(data):
    data_reshaped = data.reshape((data.shape[0], -1))
    return data_reshaped

# SHAP values for explainability

# Preprocess test features for SHAP
test_features_2d = preprocess_for_shap(test_features[:100])
train_features_2d = preprocess_for_shap(train_features[:100])

# Ensure the SHAP explainer receives the function that returns predictions
def model_predict(data):
    data_reshaped = data.reshape((data.shape[0], n_features, 1))
    return baseline_model.predict(data_reshaped)

def adaptive_model_predict(data):
    data_reshaped = data.reshape((data.shape[0], n_features, 1))
    return adaptive_model.predict(data_reshaped)

# Explainability using SHAP for static pruning
explainer = shap.KernelExplainer(model_predict, train_features_2d)
shap_values = explainer.shap_values(test_features_2d)

# Check shapes of SHAP values and test features
print("Shape of SHAP values:", np.array(shap_values).shape)
print("Shape of test features:", test_features_2d.shape)

# Ensure the shapes are compatible
if np.array(shap_values).shape[1] == test_features_2d.shape[1]:
    shap.summary_plot(shap_values, test_features_2d)
else:
    print("Error: The shape of SHAP values does not match the shape of the provided test features.")

# Explainability using SHAP for adaptive pruning
explainer_adaptive = shap.KernelExplainer(adaptive_model_predict, train_features_2d)
shap_values_adaptive = explainer_adaptive.shap_values(test_features_2d)

# Check shapes of SHAP values and test features for adaptive pruning
print("Shape of SHAP values for adaptive pruning:", np.array(shap_values_adaptive).shape)

# Ensure the shapes are compatible
if np.array(shap_values_adaptive).shape[1] == test_features_2d.shape[1]:
    shap.summary_plot(shap_values_adaptive, test_features_2d)
else:
    print("Error: The shape of SHAP values for adaptive pruning does not match the shape of the provided test features.")

# %%
# SHAP analysis for unpruned model
explainer_unpruned = shap.DeepExplainer(baseline_model, train_features[:100])
shap_values_unpruned = explainer_unpruned.shap_values(test_features[:100])

# SHAP analysis for pruned model
explainer_pruned = shap.DeepExplainer(pruned_model, train_features[:100])
shap_values_pruned = explainer_pruned.shap_values(test_features[:100])

# Summary plot for unpruned model
plt.title("SHAP Summary Plot - Unpruned Model")
shap.summary_plot(shap_values_unpruned, test_features[:100])

# Summary plot for pruned model
plt.title("SHAP Summary Plot - Pruned Model")
shap.summary_plot(shap_values_pruned, test_features[:100])

# Dependence plot for a specific feature (e.g., feature index 0)
feature_idx = 0
plt.title(f"SHAP Dependence Plot - Feature {feature_idx} (Unpruned)")
shap.dependence_plot(feature_idx, shap_values_unpruned, test_features[:100])

plt.title(f"SHAP Dependence Plot - Feature {feature_idx} (Pruned)")
shap.dependence_plot(feature_idx, shap_values_pruned, test_features[:100])

# Force plot for the first prediction in the test set
shap.force_plot(explainer_unpruned.expected_value[0], shap_values_unpruned[0], test_features[0], matplotlib=True)
shap.force_plot(explainer_pruned.expected_value[0], shap_values_pruned[0], test_features[0], matplotlib=True)


# %%
# Load the all_metrics.csv file
all_metrics_df = pd.read_csv('all_metrics.csv')

# Plot bar graph for CPU utilization
plt.figure(figsize=(12, 6))
sns.barplot(x='model', y='cpu_utilization', data=all_metrics_df, palette='deep')
plt.title('CPU Utilization Across Models')
plt.xlabel('Model')
plt.ylabel('CPU Utilization (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
plt.figure(figsize=(12, 6))
sns.barplot(x='model', y='cpu_utilization', data=all_metrics_df, palette='Set2')
plt.title('CPU Utilization Across Models')
plt.xlabel('Model')
plt.ylabel('CPU Utilization (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Display a table of accuracy scores
accuracy_table = all_metrics_df[['model', 'accuracy']]
print("Accuracy Scores for Different Models:")
print(accuracy_table)

# %%

# Plot heatmap of metrics
metrics = all_metrics_df.set_index('model')
plt.figure(figsize=(12, 8))
sns.heatmap(metrics, annot=True, cmap='viridis', cbar=True)
plt.title('Heatmap of Model Metrics')
plt.xlabel('Metrics')
plt.ylabel('Models')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# %%
