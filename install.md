#  Installation Guide

## Prerequisites
Before you begin, ensure you have the following:

- **Python 3.9 or higher**
- **At least 8GB RAM** ( GPU with 8GB VRAM recommended for model inference)
- **Stable internet connection** (for initial model downloads)

---

##  Setup Instructions

### 1. Clone the Repository
If you're working from a GitHub repo:

```bash
git clone <repository-url>
cd <repository-folder>
```

---

### 2. Create and Activate a Virtual Environment

#### macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

#### Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

---

### 3. Install Project Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Run the Application

#### Start the Report Generation Server:
```bash
python message_generator_server.py
```

#### Start the Translation Server:
```bash
python translation_server.py
```

#### Launch the Streamlit Interface:
```bash
streamlit run interface.py
```

---

###  You're all set!
Once the servers are running, access the app through the Streamlit interface in your browser.
