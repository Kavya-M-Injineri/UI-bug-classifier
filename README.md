# UI Bug Screenshot Classifier

A web application that uses a convolutional neural network to classify UI screenshots into common bug categories. Built with Flask (Python) on the backend and vanilla HTML/CSS/JavaScript on the frontend.

The system identifies five types of UI issues from screenshots: layout breaks, text overflow, dark mode rendering problems, alignment inconsistencies, and clean (no bug) screens. It provides a confidence score, probable root causes, and actionable fixes for each classification.

---

## How It Works

1. A user uploads a UI screenshot through the web interface.
2. The image is sent to the Flask backend, which runs it through a MobileNetV2-based CNN.
3. The model returns a predicted bug class with a confidence percentage.
4. The app displays root causes and recommended CSS/layout fixes based on the prediction.

If no trained model is found on disk, the backend falls back to a simulated classifier so the frontend can still be demonstrated without any ML setup.

---

## Project Structure

```
UI-bug classifier/
|-- app.py                    # Flask backend (API + static file server + JWT auth)
|-- train_model.py            # Model training script (MobileNetV2 transfer learning)
|-- generate_samples.py       # Generates synthetic training images for testing
|-- requirements.txt          # Python dependencies
|
|-- static/
|   |-- index.html            # Main SPA (dashboard, upload, reports, metrics, settings)
|   |-- login.html            # Login and registration page
|   |-- css/
|   |   +-- styles.css        # Full design system (pastel glassmorphism theme)
|   +-- js/
|       +-- app.js            # Client-side routing, charts, upload logic, JWT handling
|
|-- model/                    # Created after training
|   |-- ui_bug_classifier.keras
|   +-- training_history.json
|
|-- dataset/                  # Training images (not tracked in git)
|   |-- Alignment Issue/
|   |-- Dark Mode Issue/
|   |-- Layout Broken/
|   |-- No Bug/
|   +-- Text Overflow/
|
+-- uploads/                  # Uploaded screenshots (not tracked in git)
```

---

## Getting Started

### Prerequisites

- Python 3.9 or higher
- pip

### Installation

```bash
git clone https://github.com/<your-username>/UI-bug-classifier.git
cd UI-bug-classifier
pip install -r requirements.txt
```

### Running the Application

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser. You will be redirected to the login page on first visit.

---

## Training the Model

The training script uses MobileNetV2 (pretrained on ImageNet) as a frozen backbone, with a custom classification head fine-tuned on your dataset.

### Step 1 -- Prepare the Dataset

Organize UI bug screenshots into subfolders under `dataset/`, one folder per class:

```
dataset/
|-- Alignment Issue/    (50+ images recommended)
|-- Dark Mode Issue/
|-- Layout Broken/
|-- No Bug/
+-- Text Overflow/
```

If you want to test the pipeline without real data, you can generate synthetic samples:

```bash
python generate_samples.py
```

### Step 2 -- Train

```bash
python train_model.py
```

Training runs for up to 20 epochs with early stopping (patience of 5). The best model checkpoint is saved to `model/ui_bug_classifier.keras`. Training history (accuracy, loss per epoch) is saved to `model/training_history.json`.

### Step 3 -- Use the Trained Model

Restart the Flask server. It automatically detects the saved model file and switches from simulation to real inference. No configuration changes are needed.

---

## Authentication

The application uses JSON Web Tokens (JWT) for authentication.

- **Register** -- Create an account at `/login` (click the "Create Account" tab).
- **Login** -- Sign in with your credentials. A JWT is stored in the browser's local storage.
- **Protected Routes** -- All API endpoints require a valid token in the `Authorization: Bearer <token>` header. The frontend handles this automatically.
- **Logout** -- Click the user avatar in the top navigation bar.

User credentials are stored locally in `users.json` (passwords are SHA-256 hashed). This is suitable for development and personal use. For production, replace with a proper database and bcrypt hashing.

---

## API Endpoints

| Method | Endpoint             | Auth Required | Description                          |
|--------|----------------------|---------------|--------------------------------------|
| POST   | /api/auth/register   | No            | Create a new account                 |
| POST   | /api/auth/login      | No            | Authenticate and receive a JWT       |
| GET    | /api/auth/verify     | Yes           | Check if current token is valid      |
| POST   | /api/classify        | Yes           | Upload and classify a screenshot     |
| GET    | /api/dashboard       | Yes           | Dashboard statistics                 |
| GET    | /api/metrics         | Yes           | Model architecture and training data |
| GET    | /api/reports         | Yes           | Classification history               |
| POST   | /api/retrain         | Yes           | Trigger model retraining             |

---

## Model Architecture

```
MobileNetV2 (ImageNet, frozen)
    --> GlobalAveragePooling2D
    --> Dropout(0.4)
    --> Dense(128, ReLU)
    --> Dense(5, Softmax)
```

- Input size: 128 x 128 x 3
- Optimizer: Adam (learning rate 0.0001)
- Loss: Categorical Crossentropy
- Callbacks: EarlyStopping (patience 5), ReduceLROnPlateau, ModelCheckpoint

---

## Bug Classes

| Class             | Priority | What It Detects                                      |
|-------------------|----------|------------------------------------------------------|
| Layout Broken     | High     | Collapsed containers, broken grids, overflow clipping |
| Text Overflow     | Medium   | Text spilling outside its container                   |
| Dark Mode Issue   | Medium   | Poor contrast, invisible text on dark backgrounds     |
| Alignment Issue   | Low      | Misaligned elements, inconsistent spacing             |
| No Bug            | Low      | Clean, correctly rendered UI                          |

---

## Tech Stack

- **Backend**: Flask, TensorFlow/Keras, NumPy, Pillow
- **Frontend**: Vanilla HTML, CSS, JavaScript, Chart.js
- **Model**: MobileNetV2 (transfer learning)
- **Auth**: JWT (HMAC-SHA256)

---

## License

This project is for educational and personal use.
