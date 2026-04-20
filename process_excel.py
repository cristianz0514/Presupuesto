import pandas as pd
import json
import numpy as np

def clean(n, default="Global"):
    if pd.isna(n) or str(n).strip() == "": return default
    return str(n).strip()

def process_data():
    df = pd.read_excel('Informe Presupuesto Ejecutado 2026 - Consolidado.xlsx')
    
    # Clean and fill
    df['Presupuesto 2026'] = df['Presupuesto 2026'].fillna(0)
    df['Ejecutado 2026'] = df['Ejecutado 2026'].fillna(0)
    df['presup'] = df['Presupuesto 2026'] / 1000000
    df['ejec'] = df['Ejecutado 2026'] / 1000000
    
    df['Dependencia'] = df['Dependencia'].apply(lambda x: clean(x, "Global"))
    df['Responsable'] = df['Responsable'].apply(lambda x: clean(x, "Global"))
    df['Grupo'] = df['Grupo'].apply(lambda x: clean(x, "(Sin Grupo)"))
    df['Nombre de Cuenta'] = df['Nombre de Cuenta'].apply(lambda x: clean(x, "(Sin Cuenta)"))
    df['Descripción'] = df['Descripción'].apply(lambda x: clean(x, "(Sin Descripción)"))

    homologations = {}
    try:
        with open('homologations.json', 'r', encoding='utf-8') as f:
            h_data = json.load(f)
            homologations = {h['original']: h['suggested'] for h in h_data if h['status'] == 'approved'}
    except: pass

    def get_final_item(row):
        d = row['Descripción']
        if d in homologations: return homologations[d]
        return d

    df['Item_Final'] = df.apply(get_final_item, axis=1)

    data = []
    for dep_idx, (dep_name, dep_df) in enumerate(df.groupby('Dependencia')):
        dep_id = f'd{dep_idx}'
        dep = {
            'id': dep_id,
            'name': dep_name,
            'responsable': dep_df['Responsable'].mode().iloc[0] if not dep_df['Responsable'].empty else "Global",
            'presup': round(dep_df['presup'].sum(), 2),
            'ejec': round(dep_df['ejec'].sum(), 2),
            'areas': []
        }
        
        for area_idx, (area_name, area_df) in enumerate(dep_df.groupby('Responsable')):
            area_id = f'{dep_id}_a{area_idx}'
            area = {
                'id': area_id,
                'name': area_name,
                'responsable': area_name,
                'presup': round(area_df['presup'].sum(), 2),
                'ejec': round(area_df['ejec'].sum(), 2),
                'groups': []
            }
            
            for grp_idx, (grp_name, grp_df) in enumerate(area_df.groupby('Grupo')):
                grp_id = f'{area_id}_g{grp_idx}'
                grp = {
                    'id': grp_id,
                    'name': grp_name,
                    'presup': round(grp_df['presup'].sum(), 2),
                    'ejec': round(grp_df['ejec'].sum(), 2),
                    'accounts': []
                }
                
                for acc_idx, (acc_name, acc_df) in enumerate(grp_df.groupby('Nombre de Cuenta')):
                    acc_id = f'{grp_id}_acc{acc_idx}'
                    acc = {
                        'id': acc_id,
                        'name': acc_name,
                        'presup': round(acc_df['presup'].sum(), 2),
                        'ejec': round(acc_df['ejec'].sum(), 2),
                        'items': []
                    }
                    
                    for item_name, item_df in acc_df.groupby('Item_Final'):
                        ip = item_df['presup'].sum()
                        ie = item_df['ejec'].sum()
                        if ip != 0 or ie != 0:
                            acc['items'].append({
                                'name': item_name,
                                'presup': round(ip, 2),
                                'ejec': round(ie, 2)
                            })
                    
                    acc['items'].sort(key=lambda x: x['presup'], reverse=True)
                    if acc['presup'] != 0 or acc['ejec'] != 0:
                        grp['accounts'].append(acc)
                
                if grp['presup'] != 0 or grp['ejec'] != 0:
                    area['groups'].append(grp)
            
            if area['presup'] != 0 or area['ejec'] != 0:
                dep['areas'].append(area)
        
        if dep['presup'] != 0 or dep['ejec'] != 0:
            data.append(dep)
            
    return data

if __name__ == "__main__":
    data = process_data()
    data.sort(key=lambda x: x['name'])
    with open('data.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print("DONE")
