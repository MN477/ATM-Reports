
# ATM Technical Report Generator & Translator

This project automates the generation and translation of **technical service reports** for **ATM interventions**, enabling companies to send **structured English and French reports** to clients during open or closed ticket workflows.

## Overview

When an ATM issue occurs, the technician selects predefined technical inputs describing the issue and any actions taken. These inputs are then used to automatically generate:

- A **structured technical report** in English
- An **optional French translation** of the report

The system ensures accurate technical vocabulary and consistent phrasing using:
- A **local LLM model (Mistral)** for text generation
- **Meta’s NLLB** model for translation
- An **Excel-driven classification pipeline** that organizes abbreviations into:
  - Components
  - Ticket types
  - Actions

## Project Structure

```
atm_report_project/
├── data_processor.py            # Loads Excel file, classifies terms into JSON files
├── message_generator_server.py  # Generates English reports using local Mistral model
├── translation_server.py        # Translates English reports to French using NLLB model
├── streamlit_interface.py       # Streamlit UI to test and interact with both backends
├── actions.json                 # Auto-classified terms: actions (EN/FR)
├── components.json              # Auto-classified terms: components (EN/FR)
├── ticket_types.json            # Auto-classified terms: ticket types (EN/FR)
├── Abreviation_et_Description_V1.xlsx  # Input Excel file of abbreviations
├── requirements.txt
└── .gitignore
```

##  How It Works

### 1. Classification Phase
- `data_processor.py`:
  - Reads the Excel file containing ATM-related abbreviations in English and French.
  - Uses a fine-tuned **DeepSeek LLM** to classify each term into:
    - `action`
    - `ticket_type`
    - `component`
  - Saves the results in:
    - `actions.json`
    - `ticket_types.json`
    - `components.json`

### 2. Technical Report Generation
- `message_generator_server.py`:
  - Accepts technical abbreviations via a POST request.
  - Uses Mistral LLM to:
    - Generate an enhanced technical problem description.
    - Generate a formal intervention phrase if an action is included.
  - Returns the full client-facing **English report**.

### 3. Translation Phase
- `translation_server.py`:
  - Receives English reports and:
    - Pre-processes them by replacing technical terms with their French equivalents.
    - Translates the rest using **NLLB-200**.
    - Replaces placeholders with proper French technical terms.
  - Returns the **French version** of the report.

### 4. Streamlit Interface
- `interface.py`:
  - Frontend testing interface to:
    - Input technical issues
    - Generate reports (English/French)
    - Review and debug formatting
  - Calls both backends (Mistral + NLLB) and displays formatted output

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> Ensure you have:
> - CUDA-compatible GPU (optional for performance)
> - `llama-cpp-python` installed with local `.gguf` model (Mistral)

### 2. Run the classification step

```bash
python data_processor.py
```

This will generate or update:
- `actions.json`
- `components.json`
- `ticket_types.json`

### 3. Start the backends

**English generator (Mistral):**
```bash
python message_generator_server.py
```

**French translator (NLLB):**
```bash
python translation_server.py
```

### 4. Launch Streamlit UI
```bash
streamlit run streamlit_interface.py
```

##  Technologies Used

- **Python**
- **Transformers (HuggingFace)**
- **Mistral (GGUF, via `llama-cpp-python`)**
- **NLLB-200**: `facebook/nllb-200-distilled-600M`
- **Streamlit**
- **Pandas / Excel / JSON**

##  Example Technical Input

```
 HW ISSUE_CDM_Feeder 2_Repair /  SW ISSUE_ATM APP_Reinstallation
```

Will generate a (EN) message like:

Dear Customer,
Following the completion of the required intervention, we confirm that the ATM has been returned to service and is now operational.
Our technical team identified the following problems:
1. A hardware issue affecting the Pick note Mechanism Slot 2, which is within the Cash Dispenser Unit
2. A software issue affecting the atm application
Please find below our intervention report:
1. The affected component was repaired
2. The affected component was reinstallation

##  Status

 Fully functional and tested  
 Can be extended with:
- More categories (e.g., software issues)
- PDF/email export
- User authentication for technician input
