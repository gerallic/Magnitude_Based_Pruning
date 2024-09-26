import sys
import time
import psutil
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QComboBox, QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QMutex, QMutexLocker
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, recall_score
from sklearn.preprocessing import LabelEncoder

class RealTimePredictionThread(QThread):
    update_signal = pyqtSignal(str)

    def __init__(self, model, data):
        QThread.__init__(self)
        self.model = model
        self.data = data
        self.running = True
        self.mutex = QMutex()

    def run(self):
        for index, row in self.data.iterrows():
            if not self.is_running():
                break
            input_data = row.values.astype(np.float32)
            input_data = input_data[:self.model.input_shape[1]]
            input_data_reshaped = input_data.reshape((1,) + self.model.input_shape[1:])
            prediction = self.model.predict(input_data_reshaped)
            prediction_score = prediction[0][0] if len(prediction[0]) > 1 else prediction[0]
            attack_type = "Anomaly" if prediction_score >= 0.5 else "Normal"
            self.update_signal.emit(f'Index {index}: {attack_type}, Score: {prediction_score}')
            QThread.sleep(1)  # Simulate real-time prediction interval

    def stop(self):
        with QMutexLocker(self.mutex):
            self.running = False

    def is_running(self):
        with QMutexLocker(self.mutex):
            return self.running

class NetworkIntrusionDetectionGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Network Intrusion Detection')
        self.setGeometry(100, 100, 600, 800)  # Increased height to accommodate the second image

        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)

        # Add image to the header
        self.header_image_label = QLabel()
        header_image = QPixmap('int.png').scaledToHeight(200)  # Resize header image
        self.header_image_label.setPixmap(header_image)
        self.header_image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.header_image_label)

        # QTimer for dynamic movement
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.move_header_image)
        self.timer.start(10)  # Adjust the timeout interval (milliseconds) as needed
        
        # Styles
        style_sheet = """
            QPushButton {
                border: 2px solid #4CAF50;
                border-radius: 6px;
                padding: 5px;
                background-color: white;
                color: #4CAF50;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #4CAF50;
                color: white;
            }
            QLabel {
                font-weight: bold;
                font-size: 14px;
            }
            QComboBox {
                padding: 5px;
                border: 2px solid #4CAF50;
                border-radius: 6px;
            }
        """
        self.setStyleSheet(style_sheet)

        self.upload_button = QPushButton('Upload CSV')
        self.upload_button.clicked.connect(self.upload_csv)

        self.index_label = QLabel('Select Index:')
        self.index_combo = QComboBox()

        self.model_upload_button = QPushButton('Upload Model')
        self.model_upload_button.clicked.connect(self.upload_model)

        self.model_label = QLabel('Select Model:')
        self.model_combo = QComboBox()

        self.predict_button = QPushButton('Predict')
        self.predict_button.clicked.connect(self.predict)

        self.plot_button = QPushButton('Plot Metrics')
        self.plot_button.clicked.connect(self.plot_metrics)

        self.real_time_button = QPushButton('Start Real-time Predictions')
        self.real_time_button.clicked.connect(self.start_real_time_predictions)
        
        self.stop_real_time_button = QPushButton('Stop Real-time Predictions')
        self.stop_real_time_button.clicked.connect(self.stop_real_time_predictions)

        layout.addWidget(self.upload_button)
        layout.addWidget(self.index_label)
        layout.addWidget(self.index_combo)
        layout.addWidget(self.model_upload_button)
        layout.addWidget(self.model_label)
        layout.addWidget(self.model_combo)
        layout.addWidget(self.predict_button)
        layout.addWidget(self.plot_button)
        layout.addWidget(self.real_time_button)
        layout.addWidget(self.stop_real_time_button)

        # Add image to the footer
        footer_image_label = QLabel()
        footer_image = QPixmap('prun.png').scaledToHeight(300)  # Resize footer image
        footer_image_label.setPixmap(footer_image)
        footer_image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(footer_image_label)

        self.setLayout(layout)

        self.model = None  # Models will be loaded after CSV upload
        self.metrics = {}  # Collect pruning metrics
        self.direction = 1  # Initial direction for movement
        self.real_time_thread = None  # Initialize the real-time thread to None

    def move_header_image(self):
        current_pos = self.header_image_label.pos()
        new_y = current_pos.y() + self.direction
        if new_y <= 0 or new_y >= self.height() - self.header_image_label.height():
            self.direction *= -1
        self.header_image_label.move(current_pos.x(), new_y)

    def upload_csv(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self,"Select CSV file", "","CSV Files (*.csv)", options=options)
        if file_name:
            self.load_csv(file_name)

    def load_csv(self, file_name):
        self.data_df = pd.read_csv(file_name)
        self.preprocess_data()
        self.index_combo.clear()
        self.index_combo.addItems([str(i) for i in range(1, len(self.data_df) + 1)])

    def preprocess_data(self):
        # Encode non-numeric columns
        self.label_encoders = {}
        for column in self.data_df.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            self.data_df[column] = le.fit_transform(self.data_df[column])
            self.label_encoders[column] = le

    def upload_model(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self,"Select Model file", "","Pickle Files (*.pkl);;HDF5 Files (*.h5)", options=options)
        if file_name:
            self.load_model(file_name)
            self.model_combo.addItem(file_name.split('/')[-1])

    def load_model(self, file_name):
        try:
            if file_name.endswith('.pkl'):
                self.model = joblib.load(file_name)
            elif file_name.endswith('.h5'):
                import tensorflow as tf
                self.model = tf.keras.models.load_model(file_name)
            self.expected_input_shape = self.model.input_shape[1:]  # Save the expected input shape
            QMessageBox.information(self, 'Success', 'Model loaded successfully.')
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to load model: {e}')
            self.model = None

    def predict(self):
        if not hasattr(self, 'data_df'):
            QMessageBox.warning(self, 'Warning', 'Please upload a CSV file.')
            return

        if self.model is None:
            QMessageBox.warning(self, 'Warning', 'Please upload a model.')
            return

        index = int(self.index_combo.currentText())
        input_data = self.data_df.iloc[index - 1].values.astype(np.float32)  # Convert to float32
        
        # Ensure the input data matches the expected input shape
        input_data = input_data[:self.expected_input_shape[0]]  # Trim or pad the input data to match the expected shape
        input_data_reshaped = input_data.reshape((1,) + self.expected_input_shape)  # Adjust shape based on the model's expected input shape

        try:
            # Measure CPU usage before prediction
            cpu_before = psutil.cpu_percent(interval=None)

            start_time = time.time()
            if hasattr(self.model, 'predict_proba'):
                prediction_score = self.model.predict_proba(input_data_reshaped)[0][1]  # Assuming binary classification
            else:
                prediction = self.model.predict(input_data_reshaped)
                prediction_score = prediction[0][0] if len(prediction[0]) > 1 else prediction[0]
            end_time = time.time()

            # Measure CPU usage after prediction
            cpu_after = psutil.cpu_percent(interval=None)

            attack_type = "Anomaly" if prediction_score >= 0.5 else "Normal"
            prediction_time = end_time - start_time
            cpu_utilization = (cpu_after - cpu_before) / 100

            QMessageBox.information(self, 'Prediction', f'Predicted Attack Type: {attack_type}, Prediction Score: {prediction_score}, Prediction Time: {prediction_time:.4f} seconds, CPU Utilization: {cpu_utilization:.4f}')
            
            # Store efficiency metrics
            self.metrics['prediction_time'] = prediction_time
            self.metrics['cpu_utilization'] = cpu_utilization

        except Exception as e:
            QMessageBox.warning(self, 'Error', f'An error occurred: {str(e)}')

    def plot_metrics(self):
        if not hasattr(self, 'data_df'):
            QMessageBox.warning(self, 'Warning', 'Please upload a CSV file.')
            return

        metrics_to_plot = ['accuracy', 'recall']
        values = []

        # Calculate metrics for the entire dataset
        X = self.data_df.drop(columns=['label'])  # Assuming 'label' is the target column
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)  # Ensure all columns are numeric and handle missing values
        X = X.values.astype(np.float32)
        X = X[:, :self.expected_input_shape[0]]  # Ensure X matches the expected input shape
        X = X.reshape((X.shape[0],) + self.expected_input_shape)  # Adjust shape based on the model's expected input shape
        y = self.data_df['label']

        y_pred_continuous = self.model.predict(X)
        y_pred = (y_pred_continuous >= 0.5).astype(int)  # Convert continuous predictions to binary

        values.append(accuracy_score(y, y_pred))
        values.append(recall_score(y, y_pred))

        plt.figure()
        plt.bar(metrics_to_plot, values)
        plt.xlabel('Metrics')
        plt.ylabel('Values')
        plt.title('Model Performance Metrics')
        plt.show()

    def start_real_time_predictions(self):
        if not hasattr(self, 'data_df'):
            QMessageBox.warning(self, 'Warning', 'Please upload a CSV file.')
            return

        if self.model is None:
            QMessageBox.warning(self, 'Warning', 'Please upload a model.')
            return

        self.real_time_thread = RealTimePredictionThread(self.model, self.data_df)
        self.real_time_thread.update_signal.connect(self.display_real_time_prediction)
        self.real_time_thread.start()
        self.stop_real_time_button.setEnabled(True)  # Enable the stop button

    def stop_real_time_predictions(self):
        if self.real_time_thread is not None:
            self.real_time_thread.stop()
            self.real_time_thread.wait()
            QMessageBox.information(self, 'Information', 'Real-time predictions stopped.')
            self.stop_real_time_button.setEnabled(False)  # Disable the stop button

    def display_real_time_prediction(self, message):
        QMessageBox.information(self, 'Real-time Prediction', message)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = NetworkIntrusionDetectionGUI()
    gui.show()
    sys.exit(app.exec_())
