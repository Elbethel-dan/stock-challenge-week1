import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from CSV, Excel, or JSON files based on file extension.
    Supported formats: .csv, .xlsx, .xls, .json
    """
    file_ext = file_path.lower().split('.')[-1]

    try:
        if file_ext == "csv":
            return pd.read_csv(file_path)
        elif file_ext in ("xlsx", "xls"):
            return pd.read_excel(file_path)
        elif file_ext == "json":
            return pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file type: .{file_ext}")
    except Exception as e:
        raise ValueError(f"Failed to load file '{file_path}': {e}")