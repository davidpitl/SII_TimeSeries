from datetime import datetime

import pandas as pd

# Ruta al archivo .xlsb
xlsb_file_path = 'data/BdD_Series_diarias_SII.xlsb'
sheet_name = 'Selección series diarias'
output_csv_path = 'data/sii_total_nocierre.csv'

def excel_to_datetime(value):
    try:
        return datetime.fromtimestamp((float(value) - 25569) * 86400).strftime('%Y-%m-%d')
    except ValueError:
        return None

df = pd.read_excel(xlsb_file_path, sheet_name=sheet_name, header=None, engine='pyxlsb', skiprows=401)
df.iloc[:, 1] = df.iloc[:, 1].apply(excel_to_datetime)
#df.iloc[:, 2] = df.iloc[:, 2]#*100000000

columns_to_extract = [1,2]
selected_data = df[columns_to_extract]
selected_data.columns = ['date', 'total']
selected_data.to_csv(output_csv_path, index=False)
