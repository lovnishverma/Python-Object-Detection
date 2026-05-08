# Python Object Detection

A simple Python-based Object Detection project using OpenCV.

---

## Requirements

- Python 3.x
- pip

---

## Setup Instructions (Windows)

### 1. Create Virtual Environment

```powershell
py -3 -m venv .venv
```

---

### 2. Activate Virtual Environment

```powershell
.venv\Scripts\activate
```

After activation, you should see:

```powershell
(.venv)
```

---

### 3. Install Required Packages

Install NumPy:

```powershell
pip install numpy
```

Install imutils:

```powershell
pip install imutils
```

Install OpenCV:

```powershell
pip install opencv-python
```

---

## Verify Installation

Run the following command:

```powershell
python -c "import cv2, numpy, imutils; print('All packages installed successfully')"
```

Expected output:

```text
All packages installed successfully
```

---

## Common Fixes

### PowerShell Execution Policy Error

If activation fails, run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate the environment again:

```powershell
.venv\Scripts\activate
```

---

## Project Structure

```text
Python-Object-Detection/
│
├── .venv/
├── main.py
├── requirements.txt
└── README.md
```

---

## Optional: Create requirements.txt

```powershell
pip freeze > requirements.txt
```

Install dependencies later using:

```powershell
pip install -r requirements.txt
```
