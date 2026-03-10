# Gen-AI Learning

## Requirements

- Python >= 3.13

## Installation

### Using pip

```bash
pip install -r requirements.txt
```

### Using UV

```bash
uv venv
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows
uv pip install -r requirements.txt
```

Or directly with UV (without requiring a virtual environment first):

```bash
uv pip install -r requirements.txt
```

## Dependencies

- **python-dotenv** (>=1.0.0) - Load environment variables from .env files
- **pandas** (>=2.0.0) - Data manipulation and analysis
- **numpy** (>=1.24.0) - Numerical computing
- **jupyter** (>=1.0.0) - Interactive notebooks
- **tabulate** (>=0.9.0) - Pretty-print tabular data
- **matplotlib** (>=3.6.0) - Plotting and visualization
- **groq** (==1.1.0) - Groq API client
- **transformers** (>=4.30.0) - Hugging Face transformers library

## Project Structure

```
.
├── main.py
├── pyproject.toml
├── requirements.txt
├── README.md
└── week1/
    └── Day1/
        ├── Task1.ipynb
        └── Task2.ipynb
```

## Usage

Run the main script:

```bash
python main.py
```

Or explore the Jupyter notebooks in the `week1/Day1/` directory:

```bash
jupyter notebook
```

## Notes

- This project uses `pyproject.toml` for package configuration with Poetry-style dependency declarations
- A `requirements.txt` file is provided for compatibility with pip
- Both pip and UV package managers are supported
