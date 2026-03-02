from typing import Dict
import json
import pandas as pd
import re

def _sanitize_sheet_name(name: str) -> str:
    """
    Sanitize sheet name for Excel compatibility.
    Excel sheet names cannot contain: [ ] : * ? / \
    Max length: 31 characters
    """
    # Replace invalid characters with underscore
    invalid_chars = r'[\[\]:*?/\\]'
    sanitized = re.sub(invalid_chars, '_', name)

    # Truncate to 31 characters (Excel limit)
    if len(sanitized) > 31:
        sanitized = sanitized[:31]

    # Ensure not empty
    if not sanitized:
        sanitized = "Sheet"

    return sanitized

def save_eval_to_json(
    all_results: Dict[str, Dict[str, Dict[str, float]]],
    filename: str = "regression_results.json"
) -> None:
    """
    Save regression results to JSON file.
    """
    assert filename.endswith(".json"), "To save JSON file, filename must end with .json"
    with open(filename, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"Results saved to {filename}")

def load_eval_from_json(filename: str) -> Dict:
    """
    Load regression evaluation results from JSON file.
    """
    assert filename.endswith(".json"), "To load JSON file, filename must end with .json"
    with open(filename, 'r') as f:
        all_results = json.load(f)
    return all_results


def save_eval_to_excel(
    all_results: Dict[str, Dict[str, Dict[str, float]]],
    filename: str = "regression_results.xlsx"
) -> None:
    """
    Save regression results to Excel file with multiple sheets.
    Each sheet corresponds to one target variable.
    Rows = methods, Columns = metrics (MAPE_%, CVRMSE_%, MaxAPE_%, Pearson_r, R2)


    all_results format:
        {target1: {method1 : {metric1: value, ...}, method2: {...}}, target2: {...}, ...}
    """
    assert filename.endswith(".xlsx"), "To save excel file, filename must end with .xlsx"

    with pd.ExcelWriter(filename, engine='openpyxl') as writer:

        for target_name, methods_dict in all_results.items():
            # Create DataFrame: rows=methods, columns=metrics
            df_data = []

            for method, metrics in methods_dict.items():
                row = {'Method': method}
                row.update(metrics)
                df_data.append(row)

            df = pd.DataFrame(df_data)

            # Ensure Method is first column
            cols = ['Method'] + [col for col in df.columns if col != 'Method']
            df = df[cols]

            # Sanitize sheet name for Excel compatibility
            sheet_name = _sanitize_sheet_name(target_name)

            # Handle duplicate sheet names (if sanitization causes collision)
            original_sheet_name = sheet_name
            counter = 1
            while sheet_name in writer.sheets:
                suffix = f"_{counter}"
                sheet_name = original_sheet_name[:31-len(suffix)] + suffix
                counter += 1

            # Write to Excel
            df.to_excel(writer, sheet_name=sheet_name, index=False)

            # Auto-adjust column widths
            worksheet = writer.sheets[sheet_name]
            for idx, col in enumerate(df.columns):
                max_length = max(
                    df[col].astype(str).map(len).max(),
                    len(str(col))
                ) + 2
                # Handle more than 26 columns (AA, AB, etc.)
                if idx < 26:
                    col_letter = chr(65 + idx)
                else:
                    col_letter = chr(65 + (idx // 26) - 1) + chr(65 + (idx % 26))
                worksheet.column_dimensions[col_letter].width = min(max_length, 50)

    print(f"Results saved to {filename}")