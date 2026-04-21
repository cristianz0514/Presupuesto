import pandas as pd
import json
import numpy as np
import logging
import os
import textdistance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(filename='process_logs.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s', filemode='w', encoding='utf-8')

ALL_MONTHS = ['2026-01','2026-02','2026-03','2026-04','2026-05','2026-06',
              '2026-07','2026-08','2026-09','2026-10','2026-11','2026-12']
MONTH_LABELS = {'2026-01':'Ene','2026-02':'Feb','2026-03':'Mar','2026-04':'Abr',
                '2026-05':'May','2026-06':'Jun','2026-07':'Jul','2026-08':'Ago',
                '2026-09':'Sep','2026-10':'Oct','2026-11':'Nov','2026-12':'Dic'}

def clean(n, default="Global"):
    if pd.isna(n) or str(n).strip() == "": return default
    return str(n).strip()

def run_contextual_ai(df, threshold=0.8):
    suggestions = []
    grouped = df.groupby(['Responsable', 'Nombre de Cuenta'])
    for (area, cuenta), sub_df in grouped:
        budgeted = sub_df[sub_df['Presupuesto 2026'] > 0]['Descripcion'].unique().tolist()
        executed = sub_df[sub_df['Ejecutado 2026'] > 0]['Descripcion'].unique().tolist()
        if not budgeted or not executed: continue
        vectorizer = TfidfVectorizer().fit(budgeted + executed)
        b_tfidf = vectorizer.transform(budgeted)
        e_tfidf = vectorizer.transform(executed)
        cosine_sim = cosine_similarity(e_tfidf, b_tfidf)
        for i, ex_item in enumerate(executed):
            best_idx = np.argmax(cosine_sim[i])
            score = cosine_sim[i][best_idx]
            jaro = textdistance.jaro_winkler(ex_item.lower(), budgeted[best_idx].lower())
            confidence = (score * 0.4) + (jaro * 0.6)
            if confidence >= threshold and ex_item != budgeted[best_idx]:
                suggestions.append({"area": area, "cuenta": cuenta, "original": ex_item,
                    "suggested": budgeted[best_idx], "confidence": round(confidence * 100, 1),
                    "status": "pending"})
    return suggestions

def build_monthly_presup(df_sub):
    r = {}
    for m in ALL_MONTHS:
        v = round(float(df_sub[df_sub['Ano_Mes']==m]['presup'].sum()), 2)
        if v: r[m] = v
    return r

def build_monthly_ejec(df_sub):
    r = {}
    for m in ALL_MONTHS:
        v = round(float(df_sub[df_sub['Ano_Mes']==m]['ejec'].sum()), 2)
        if v: r[m] = v
    return r

def build_monthly_ejec_co(df_sub, companies):
    r = {}
    for co in companies:
        sub = df_sub[df_sub['Nombre Empresa']==co]
        if sub['ejec'].sum() == 0: continue
        co_data = {}
        for m in ALL_MONTHS:
            v = round(float(sub[sub['Ano_Mes']==m]['ejec'].sum()), 2)
            if v: co_data[m] = v
        if co_data: r[co] = co_data
    return r

def main():
    try:
        logging.info("Iniciando procesamiento...")
        df = pd.read_excel('Informe Presupuesto Ejecutado 2026 - Consolidado.xlsx')
        df['Presupuesto 2026'] = df['Presupuesto 2026'].fillna(0)
        df['Ejecutado 2026']   = df['Ejecutado 2026'].fillna(0)
        df['presup'] = df['Presupuesto 2026'] / 1_000_000
        df['ejec']   = df['Ejecutado 2026']   / 1_000_000
        df['Ano_Mes'] = df['Año Mes'].apply(lambda x: str(x).strip() if pd.notna(x) else '')
        df['Nombre Empresa'] = df['Nombre Empresa'].apply(lambda x: clean(x, ''))
        for col in ['Dependencia','Responsable','Grupo','Nombre de Cuenta','Descripción']:
            df[col] = df[col].apply(lambda x: clean(x, f'({col} N/A)'))
        df['Descripcion'] = df['Descripción']

        homo_decisions = []
        if os.path.exists('homologations.json'):
            with open('homologations.json', 'r', encoding='utf-8') as f:
                homo_decisions = json.load(f)
        ai_suggestions = run_contextual_ai(df)
        final_homo = {f"{h['area']}|{h['cuenta']}|{h['original']}": h for h in homo_decisions}
        for s in ai_suggestions:
            key = f"{s['area']}|{s['cuenta']}|{s['original']}"
            if key not in final_homo: final_homo[key] = s
        homo_list = list(final_homo.values())
        with open('homologations.json', 'w', encoding='utf-8') as f:
            json.dump(homo_list, f, indent=2, ensure_ascii=False)

        approved_map = {(h['area'], h['cuenta'], h['original']): h['suggested']
                        for h in homo_list if h['status'] == 'approved'}
        def apply_homo(row):
            key = (row['Responsable'], row['Nombre de Cuenta'], row['Descripcion'])
            return approved_map.get(key, row['Descripcion'])
        df['Desc_Final'] = df.apply(apply_homo, axis=1)

        companies = sorted([c for c in df[df['ejec']>0]['Nombre Empresa'].dropna().unique() if c])

        data = []
        for dep_n, d_df in df.groupby('Dependencia'):
            dep = {
                'name': dep_n,
                'presup': round(d_df['presup'].sum(), 2),
                'ejec':   round(d_df['ejec'].sum(), 2),
                'presup_m': build_monthly_presup(d_df),
                'ejec_m':   build_monthly_ejec(d_df),
                'ejec_m_co': build_monthly_ejec_co(d_df, companies),
                'areas': []
            }
            for resp_n, a_df in d_df.groupby('Responsable'):
                area = {
                    'name': resp_n,
                    'presup': round(a_df['presup'].sum(), 2),
                    'ejec':   round(a_df['ejec'].sum(), 2),
                    'presup_m': build_monthly_presup(a_df),
                    'ejec_m':   build_monthly_ejec(a_df),
                    'ejec_m_co': build_monthly_ejec_co(a_df, companies),
                    'groups': []
                }
                for grp_n, g_df in a_df.groupby('Grupo'):
                    grp = {
                        'name': grp_n,
                        'presup': round(g_df['presup'].sum(), 2),
                        'ejec':   round(g_df['ejec'].sum(), 2),
                        'presup_m': build_monthly_presup(g_df),
                        'ejec_m':   build_monthly_ejec(g_df),
                        'accounts': []   # no ejec_m_co at group level to keep JSON lean
                    }
                    for acc_n, acc_df in g_df.groupby('Nombre de Cuenta'):
                        acc = {
                            'name': acc_n,
                            'presup': round(acc_df['presup'].sum(), 2),
                            'ejec':   round(acc_df['ejec'].sum(), 2),
                            'items': []
                        }
                        for item_n, i_df in acc_df.groupby('Desc_Final'):
                            ip = round(i_df['presup'].sum(), 2)
                            ie = round(i_df['ejec'].sum(), 2)
                            if ip != 0 or ie != 0:
                                acc['items'].append({
                                    'name': clean(item_n),
                                    'presup': ip, 'ejec': ie
                                })
                        acc['items'].sort(key=lambda x: x['presup'], reverse=True)
                        grp['accounts'].append(acc)
                    area['groups'].append(grp)
                dep['areas'].append(area)
            data.append(dep)

        write_html(data, homo_list, companies)
        print("SUCCESS")
    except Exception as e:
        logging.error(f"FATAL: {e}", exc_info=True)
        print(f"ERROR: {e}")


def write_html(data, homologations, companies):
    js_data      = json.dumps(data, ensure_ascii=False)
    js_homo      = json.dumps(homologations, ensure_ascii=False)
    js_companies = json.dumps(companies, ensure_ascii=False)
    js_months    = json.dumps(ALL_MONTHS, ensure_ascii=False)
    js_labels    = json.dumps(MONTH_LABELS, ensure_ascii=False)

    CSS = """
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Inter', sans-serif; background: #F0F4F8; color: #1A2535; display: flex; min-height: 100vh; }
.sidebar { width: 56px; background: #0F172A; display: flex; flex-direction: column; position: fixed; height: 100vh; z-index: 200; overflow: hidden; transition: width 0.22s cubic-bezier(.4,0,.2,1); }
.sidebar:hover { width: 220px; }
.side-logo { padding: 16px 14px 14px; border-bottom: 1px solid #1E293B; display: flex; align-items: center; gap: 10px; min-height: 56px; }
.side-logo-icon { font-size: 15px; font-weight: 900; color: #3B82F6; flex-shrink: 0; width: 28px; text-align: center; }
.side-logo-text { opacity: 0; white-space: nowrap; transition: opacity 0.15s; }
.sidebar:hover .side-logo-text { opacity: 1; }
.side-logo-main { font-size: 13px; font-weight: 800; color: #F1F5F9; letter-spacing: .5px; }
.side-logo-sub  { font-size: 10px; color: #64748B; margin-top: 1px; }
.side-section   { padding: 16px 14px 4px; font-size: 9px; font-weight: 700; color: #475569; letter-spacing: .12em; text-transform: uppercase; white-space: nowrap; opacity: 0; transition: opacity 0.15s; }
.sidebar:hover .side-section { opacity: 1; }
.side-menu      { flex: 1; overflow-y: auto; overflow-x: hidden; padding: 4px 0 12px; }
.menu-item { padding: 9px 14px; cursor: pointer; display: flex; align-items: center; gap: 10px; color: #94A3B8; font-size: 13px; font-weight: 500; border-radius: 6px; margin: 1px 6px; transition: background .12s, color .12s; white-space: nowrap; }
.menu-item:hover  { color: #F1F5F9; background: #1E293B; }
.menu-item.active { background: #1E3A5F; color: #60A5FA; font-weight: 600; }
.menu-item.disabled { opacity: .25; cursor: default; pointer-events: none; }
.menu-icon { width: 20px; text-align: center; flex-shrink: 0; font-size: 14px; }
.menu-text { opacity: 0; transition: opacity 0.15s; }
.sidebar:hover .menu-text { opacity: 1; }
.side-footer { padding: 10px 14px; font-size: 9px; color: #334155; border-top: 1px solid #1E293B; white-space: nowrap; opacity: 0; transition: opacity 0.15s; }
.sidebar:hover .side-footer { opacity: 1; }
.main { flex: 1; margin-left: 56px; display: flex; flex-direction: column; min-height: 100vh; transition: margin-left 0.22s cubic-bezier(.4,0,.2,1); }
.top-bar { height: 52px; background: #fff; border-bottom: 1px solid #E4E8EF; display: flex; align-items: center; padding: 0 24px; justify-content: space-between; position: sticky; top: 0; z-index: 100; }
.top-title { font-weight: 700; font-size: 14px; color: #1A2535; }
.content { padding: 20px 24px; flex: 1; }
.hidden { display: none !important; }
.filter-bar { display: flex; align-items: center; gap: 10px; margin-bottom: 18px; flex-wrap: wrap; }
.filter-label { font-size: 11px; font-weight: 600; color: #8A99AA; text-transform: uppercase; letter-spacing: .06em; }
.co-tabs { display: flex; gap: 4px; flex-wrap: wrap; }
.co-tab { padding: 5px 12px; border-radius: 20px; font-size: 12px; font-weight: 500; cursor: pointer; border: 1px solid #E4E8EF; background: #fff; color: #6B7A8D; transition: all .12s; white-space: nowrap; }
.co-tab:hover  { border-color: #93C5FD; color: #185FA5; }
.co-tab.active { background: #EFF6FF; border-color: #3B82F6; color: #185FA5; font-weight: 600; }
.filter-sep { width: 1px; height: 22px; background: #E4E8EF; flex-shrink: 0; }
.period-select { padding: 5px 10px; border: 1px solid #E4E8EF; border-radius: 7px; font-size: 12px; font-family: inherit; color: #2E3A4A; background: #fff; cursor: pointer; }
.mode-toggle { display: flex; border: 1px solid #E4E8EF; border-radius: 7px; overflow: hidden; }
.mode-btn { padding: 5px 13px; font-size: 12px; font-weight: 500; cursor: pointer; background: #fff; color: #6B7A8D; border: none; transition: all .12s; }
.mode-btn.active { background: #EFF6FF; color: #185FA5; font-weight: 600; }
.dd-breadcrumb { display: flex; align-items: center; gap: 6px; font-size: 13px; color: #6B7A8D; margin-bottom: 18px; flex-wrap: wrap; }
.dd-bc-link   { color: #185FA5; cursor: pointer; font-weight: 500; }
.dd-bc-link:hover { text-decoration: underline; }
.dd-bc-sep    { color: #C0CCDA; }
.dd-bc-active { color: #1A2535; font-weight: 600; }
.tag { display: inline-block; padding: 2px 9px; border-radius: 20px; font-size: 10px; font-weight: 700; white-space: nowrap; }
.tag-g { background: #E1F5EE; color: #085041; }
.tag-y { background: #FAEEDA; color: #633806; }
.tag-r { background: #FCEBEB; color: #791F1F; }
.dep-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px,1fr)); gap: 16px; margin-bottom: 20px; }
.dep-card { background: #fff; border: 1px solid #E4E8EF; border-radius: 12px; padding: 18px 18px 14px; cursor: pointer; transition: box-shadow .15s, transform .12s; }
.dep-card:hover { box-shadow: 0 4px 16px rgba(0,0,0,.09); transform: translateY(-2px); }
.dep-card-header { display: flex; align-items: center; gap: 8px; margin-bottom: 12px; }
.dep-dot { width: 10px; height: 10px; border-radius: 3px; flex-shrink: 0; }
.dep-name { font-size: 13px; font-weight: 700; color: #1A2535; line-height: 1.3; }
.dep-kpi-row { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 10px; }
.dep-kpi { background: #F8FAFC; border-radius: 7px; padding: 8px 10px; }
.dep-kpi-lbl { font-size: 9px; font-weight: 700; color: #8A99AA; text-transform: uppercase; letter-spacing: .06em; }
.dep-kpi-val { font-size: 18px; font-weight: 700; color: #0F1924; line-height: 1.2; }
.prog-track { height: 6px; background: #E8EDF2; border-radius: 3px; overflow: hidden; margin: 8px 0 6px; }
.prog-fill  { height: 100%; border-radius: 3px; transition: width .4s; }
.dep-footer { display: flex; align-items: center; justify-content: space-between; margin-top: 4px; }
.dep-link { font-size: 11px; color: #185FA5; font-weight: 600; }
.area-preview { display: flex; flex-direction: column; gap: 3px; margin: 8px 0; border-top: 1px solid #EEF1F5; padding-top: 8px; }
.area-prev-row { display: flex; align-items: center; gap: 6px; font-size: 11px; color: #6B7A8D; }
.sem-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
.area-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px,1fr)); gap: 14px; margin-bottom: 20px; }
.area-card { background: #fff; border: 1px solid #E4E8EF; border-radius: 10px; padding: 15px 15px 12px; cursor: pointer; transition: box-shadow .15s, transform .12s; }
.area-card:hover { box-shadow: 0 4px 14px rgba(0,0,0,.10); transform: translateY(-2px); }
.area-name { font-size: 12px; font-weight: 700; color: #1A2535; margin-bottom: 8px; }
.area-kpi-row { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-bottom: 8px; }
.area-kpi { background: #F8FAFC; border-radius: 6px; padding: 6px 9px; }
.area-kpi-lbl { font-size: 9px; font-weight: 700; color: #8A99AA; text-transform: uppercase; }
.area-kpi-val { font-size: 15px; font-weight: 700; color: #0F1924; }
.grp-mini { display: flex; flex-direction: column; gap: 4px; margin-top: 8px; border-top: 1px solid #EEF1F5; padding-top: 8px; }
.grp-mini-row { display: flex; align-items: center; gap: 6px; }
.grp-mini-lbl { font-size: 10px; color: #6B7A8D; width: 90px; flex-shrink: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.grp-mini-track { flex: 1; height: 4px; background: #E8EDF2; border-radius: 2px; overflow: hidden; }
.grp-mini-fill  { height: 100%; border-radius: 2px; }
.grp-mini-pct { font-size: 10px; font-weight: 600; color: #4B5563; width: 28px; text-align: right; }
.area-footer { display: flex; align-items: center; justify-content: space-between; margin-top: 8px; }
.area-link { font-size: 11px; color: #185FA5; font-weight: 600; }
.chart-wrap { background: #F8FAFC; border: 1px solid #E4E8EF; border-radius: 10px; padding: 16px 20px; margin-bottom: 16px; }
.chart-title { font-size: 11px; font-weight: 700; color: #8A99AA; text-transform: uppercase; letter-spacing: .07em; margin-bottom: 12px; }
.detail-table { width: 100%; border-collapse: collapse; font-size: 12px; border: 1px solid #E4E8EF; border-radius: 10px; overflow: hidden; margin-bottom: 16px; }
.detail-table th { background: #F5F7FA; color: #8A99AA; font-size: 9px; font-weight: 700; letter-spacing: .07em; text-transform: uppercase; padding: 8px 12px; border-bottom: 1px solid #E4E8EF; text-align: left; }
.detail-table th.r { text-align: right; }
.row-grp td { background: #F0F4F8; color: #374151; font-size: 10px; font-weight: 700; letter-spacing: .04em; text-transform: uppercase; padding: 7px 12px; border-top: 2px solid #E4E8EF; border-bottom: 1px solid #E4E8EF; }
.row-item td { padding: 8px 12px 8px 28px; border-bottom: 1px solid #F3F4F6; color: #374151; vertical-align: middle; }
.row-item:hover td { background: #FAFBFC; }
.row-total td { padding: 9px 12px; background: #1A2535; color: #F1F5F9; font-weight: 700; font-size: 12px; }
.r { text-align: right; }
.vari-up   { color: #791F1F; font-weight: 600; }
.vari-down { color: #085041; font-weight: 600; }
.bar-cell  { width: 110px; }
.emb-track { height: 5px; background: #E8EDF2; border-radius: 3px; overflow: hidden; margin-top: 3px; }
.emb-fill  { height: 100%; border-radius: 3px; }
.badge-g { display: inline-block; background: #374151; color: #fff; font-size: 9px; font-weight: 700; border-radius: 3px; padding: 1px 5px; margin-right: 6px; }
.homo-island  { background: #fff; border-radius: 12px; padding: 20px 24px 24px; border: 1px solid #E4E8EF; }
.homo-header  { display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px; flex-wrap: wrap; gap: 8px; }
.homo-title   { font-size: 15px; font-weight: 700; color: #1A2535; }
.homo-filter-badge { background: #FEF3C7; color: #B45309; border: 1px solid #FDE68A; border-radius: 20px; padding: 3px 10px; font-size: 11px; font-weight: 600; }
.homo-sub     { font-size: 12px; color: #8A99AA; margin: 0 0 16px; }
.homo-table   { width: 100%; border-collapse: collapse; font-size: 12px; border: 1px solid #E4E8EF; border-radius: 10px; overflow: hidden; }
.homo-table th { background: #F5F7FA; color: #8A99AA; font-size: 10px; font-weight: 700; letter-spacing: .07em; text-transform: uppercase; padding: 9px 12px; border-bottom: 1px solid #E4E8EF; text-align: left; }
.homo-table td { padding: 9px 12px; border-bottom: 1px solid #F0F3F7; color: #2E3A4A; vertical-align: middle; }
.homo-table tr:last-child td { border-bottom: none; }
.homo-input { border: 1px solid #E4E8EF; border-radius: 5px; padding: 4px 8px; font-size: 12px; width: 160px; }
.btn-primary { background: #185FA5; color: #fff; border: none; padding: 8px 18px; border-radius: 7px; cursor: pointer; font-weight: 600; font-size: 13px; }
.btn-primary:hover { background: #135091; }
.btn-sm { padding: 3px 10px !important; font-size: 11px !important; }
.btn-ghost { background: transparent; color: #6B7A8D; border: 1px solid #E4E8EF; padding: 7px 14px; border-radius: 7px; cursor: pointer; font-size: 12px; font-weight: 500; }
.btn-ghost:hover { background: #F0F4F8; }
.btn-approve { background: #DCFCE7 !important; color: #166534 !important; border-color: #BBF7D0 !important; }
.btn-reject  { background: #FEE2E2 !important; color: #991B1B !important; border-color: #FECACA !important; }
.pill { padding: 2px 8px; border-radius: 20px; font-size: 11px; font-weight: 700; }
"""

    JS = ("""
const DATA        = """ + js_data + """;
const ALL_MONTHS  = """ + js_months + """;
const MONTH_LBLS  = """ + js_labels + """;
const COMPANIES   = """ + js_companies + """;
var   HOMOLOGATIONS = """ + js_homo + """;

var state = { level: 0, depIdx: 0, areaIdx: 0, cutMonth: '2026-02', mode: 'acum', co: 'all' };
var homoFilter = null;
var _cL1 = null, _cL2 = null;
var expandedGroups = {};   /* gi -> true/false; default collapsed */
function tglGrp(gi) { expandedGroups[gi] = !expandedGroups[gi]; render(); }

/* ── Colors ── */
var DEP_COLORS = ['#534AB7','#0F6E56','#BA7517','#B91C1C','#0369A1','#7C3AED','#0891B2','#D97706','#BE185D','#047857'];
function depColor(i) { return DEP_COLORS[i % DEP_COLORS.length]; }

/* ── Semaphore ── */
function semColor(p) { return p<=85?'#085041':p<=100?'#633806':'#791F1F'; }
function semBg(p)    { return p<=85?'#E1F5EE':p<=100?'#FAEEDA':'#FCEBEB'; }
function semDot(p)   { return p<=85?'#22C55E':p<=100?'#F59E0B':'#E24B4A'; }
function semTag(p) {
  var cls = p<=85?'tag-g':p<=100?'tag-y':'tag-r';
  var lbl = p<=85?'En rango':p<=100?'Al l\u00edmite':'Excedido';
  return '<span class="tag '+cls+'">'+p+'%\u00a0'+lbl+'</span>';
}

/* ── Formatters ── */
function fmtK(v) {
  var a = Math.abs(v||0), s = v<0?'-':'';
  if (a>=1e9) return s+'$'+(a/1e9).toFixed(1)+'MM';
  if (a>=1e6) return s+'$'+(a/1e6).toFixed(1)+'M';
  if (a>=1e3) return s+'$'+(a/1e3).toFixed(0)+'k';
  return s+'$'+Math.round(a);
}
function fmtFull(v) {
  var a = Math.abs(Math.round(v||0));
  return (v<0?'-':'')+'$'+a.toLocaleString('es-CO');
}

/* ── Progress bar ── */
function progBar(ejec, presup, color) {
  var r = presup>0 ? ejec/presup : (ejec>0?2:0);
  var w = Math.min(r,1)*100;
  var over = r>1;
  return '<div class="prog-track"><div class="prog-fill" style="width:'+w+'%;background:'+(over?'#E24B4A':color)+'"></div></div>';
}

/* ── Varianza ── */
function variCell(presup, ejec) {
  var v = presup - ejec;
  if (v >= 0) return '<span class="vari-down">\u25bc '+fmtFull(v*1e6)+'</span>';
  return '<span class="vari-up">\u25b2 +'+fmtFull(Math.abs(v)*1e6)+'</span>';
}

/* ── Cut-aware getters ── */
function getPresup(node) {
  if (!node.presup_m) return node.presup;
  if (state.mode === 'mensual') return Math.round((node.presup_m[state.cutMonth]||0)*100)/100;
  var t = 0;
  for (var m in node.presup_m) { if (m <= state.cutMonth) t += node.presup_m[m]; }
  return Math.round(t*100)/100;
}
function getEjec(node) {
  if (state.co !== 'all') {
    var cd = (node.ejec_m_co && node.ejec_m_co[state.co]) || {};
    if (state.mode === 'mensual') return Math.round((cd[state.cutMonth]||0)*100)/100;
    var t = 0;
    for (var m in cd) { if (m <= state.cutMonth) t += cd[m]; }
    return Math.round(t*100)/100;
  }
  if (!node.ejec_m) return node.ejec;
  if (state.mode === 'mensual') return Math.round((node.ejec_m[state.cutMonth]||0)*100)/100;
  var t = 0;
  for (var m in node.ejec_m) { if (m <= state.cutMonth) t += node.ejec_m[m]; }
  return Math.round(t*100)/100;
}
function pct(e, p) { return p===0?(e>0?999:0):Math.round(e/p*100); }

/* ── Filter bar ── */
function buildFilterBar() {
  var coH = '<div class="co-tab'+(state.co==='all'?' active':'')+'" onclick="setCo(0)">Grupo</div>';
  COMPANIES.forEach(function(co, ci) {
    var s = co.replace('C.I. Fortia Minerals S.A.S.','Fortia Minerals')
              .replace('Sociedad Portuaria Coalcorp S.A.','Coalcorp')
              .replace('Fortia Investments Colombia S.A.S.','Investments')
              .replace('Transcaribbean Chartering S.A.','Transcaribbean')
              .replace(' S.A.S.','').replace(' S.A.','').replace(' S. De R.L.','');
    coH += '<div class="co-tab'+(state.co===co?' active':'')+'" onclick="setCo('+(ci+1)+')">'+s+'</div>';
  });
  var pH = '<select class="period-select" onchange="setCut(this.selectedIndex)">';
  ALL_MONTHS.forEach(function(m){ pH += '<option'+(m===state.cutMonth?' selected':'')+'>Corte: '+MONTH_LBLS[m]+' 2026</option>'; });
  pH += '</select>';
  var mH = '<div class="mode-toggle">'
    +'<button class="mode-btn'+(state.mode==='acum'?' active':'')+'" onclick="setMode(0)">Acumulado</button>'
    +'<button class="mode-btn'+(state.mode==='mensual'?' active':'')+'" onclick="setMode(1)">Mensual</button>'
    +'</div>';
  return '<div class="filter-bar"><span class="filter-label">Empresa</span><div class="co-tabs">'+coH+'</div>'
    +'<div class="filter-sep"></div>'+pH+'<div class="filter-sep"></div>'+mH+'</div>';
}
function setCo(i)  { state.co = i===0?'all':COMPANIES[i-1]; render(); }
function setCut(i) { state.cutMonth = ALL_MONTHS[i]; render(); }
function setMode(m){ state.mode = m===0?'acum':'mensual'; render(); }

/* ── Breadcrumb ── */
function bc(parts) {
  var h = '<div class="dd-breadcrumb">';
  parts.forEach(function(p,i){
    if (i>0) h += '<span class="dd-bc-sep">\u203a</span>';
    h += p.fn ? '<span class="dd-bc-link" onclick="'+p.fn+'">'+p.t+'</span>'
               : '<span class="dd-bc-active">'+p.t+'</span>';
  });
  return h+'</div>';
}

/* ── Navigation ── */
function openWC(id) {
  document.getElementById('wc-dash').classList.toggle('hidden', id!=='dash');
  document.getElementById('wc-homo').classList.toggle('hidden', id!=='homo');
  document.getElementById('m-dash').classList.toggle('active', id==='dash');
  document.getElementById('m-homo').classList.toggle('active', id==='homo');
  document.getElementById('wc-title').textContent = id==='dash'?'Monitor Presupuestal':'Homologaciones IA';
  if (id==='homo') renderHomo(); else render();
}
function render() {
  var c = document.getElementById('wc-dash');
  if (state.level===0) renderL0(c);
  else if (state.level===1) renderL1(c);
  else renderL2(c);
}

/* ══ NIVEL 1 ══ */
function renderL0(c) {
  var tp = DATA.reduce(function(s,d){ return s+getPresup(d); },0);
  var te = DATA.reduce(function(s,d){ return s+getEjec(d); },0);
  var pp = pct(te, tp);
  var modeLbl = state.mode==='mensual'?'Mes '+MONTH_LBLS[state.cutMonth]:'Acum. hasta '+MONTH_LBLS[state.cutMonth];

  var kpiH = '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:12px;margin-bottom:20px">'
    +[{l:'Presupuesto Periodo',v:fmtK(tp*1e6),s:modeLbl},
      {l:'Total Ejecutado',v:fmtK(te*1e6),s:modeLbl},
      {l:'Saldo',v:fmtK((tp-te)*1e6),s:'Por ejecutar'},
      {l:'Ejecuci\u00f3n',v:pp+'%',tag:1}
    ].map(function(k){
      return '<div style="background:#F8FAFC;border:1px solid #E4E8EF;border-radius:8px;padding:10px 12px">'
        +'<div style="font-size:9px;font-weight:700;color:#8A99AA;text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px">'+k.l+'</div>'
        +'<div style="font-size:20px;font-weight:700;color:#0F1924;line-height:1.2">'+k.v+'</div>'
        +(k.tag?semTag(pp):'<div style="font-size:10px;color:#8A99AA;margin-top:2px">'+k.s+'</div>')
        +'</div>';
    }).join('')+'</div>';

  var cardsH = '<div class="dep-grid">'
    +DATA.map(function(d,i){
      var dp=getPresup(d), de=getEjec(d), p=pct(de,dp), color=depColor(i);
      var prevH = d.areas.slice(0,3).map(function(a){
        var p2=pct(getEjec(a),getPresup(a));
        var nm=a.name.length>26?a.name.substring(0,26)+'...':a.name;
        return '<div class="area-prev-row"><span class="sem-dot" style="background:'+semDot(p2)+'"></span>'+nm+'</div>';
      }).join('');
      if (d.areas.length>3) prevH+='<div style="font-size:10px;color:#B0BEC5;margin-top:2px">+'+(d.areas.length-3)+' m\u00e1s</div>';
      return '<div class="dep-card" onclick="state.level=1;state.depIdx='+i+';render()">'
        +'<div class="dep-card-header"><div class="dep-dot" style="background:'+color+'"></div>'
        +'<div class="dep-name">'+d.name+'</div></div>'
        +'<div class="dep-kpi-row">'
        +'<div class="dep-kpi"><div class="dep-kpi-lbl">Presupuesto</div><div class="dep-kpi-val">'+fmtK(dp*1e6)+'</div></div>'
        +'<div class="dep-kpi"><div class="dep-kpi-lbl">Ejecutado</div><div class="dep-kpi-val">'+fmtK(de*1e6)+'</div></div>'
        +'</div>'
        +progBar(de,dp,color)
        +'<div class="dep-footer">'+semTag(p)+'</div>'
        +'<div class="area-preview">'+prevH+'</div>'
        +'<div class="dep-footer"><span class="dep-link">Ver \u00e1reas \u2192</span></div>'
        +'</div>';
    }).join('')+'</div>';

  c.innerHTML = buildFilterBar()+kpiH+cardsH;
}

/* ══ NIVEL 2 ══ */
function renderL1(c) {
  var dep=DATA[state.depIdx], di=state.depIdx, color=depColor(di);

  var cardsH = '<div class="area-grid">'
    +dep.areas.map(function(a,i){
      var ap=getPresup(a), ae=getEjec(a), p=pct(ae,ap);
      var grpH = a.groups.slice(0,4).map(function(g){
        var gp=getPresup(g), ge=getEjec(g), gp2=pct(ge,gp);
        var lbl=g.name.replace(/_/g,' ');
        lbl=lbl.length>15?lbl.substring(0,15)+'...':lbl;
        return '<div class="grp-mini-row">'
          +'<div class="grp-mini-lbl">'+lbl+'</div>'
          +'<div class="grp-mini-track"><div class="grp-mini-fill" style="width:'+Math.min(gp2,100)+'%;background:'+(gp2>100?'#E24B4A':color)+'"></div></div>'
          +'<div class="grp-mini-pct">'+gp2+'%</div>'
          +'</div>';
      }).join('');
      return '<div class="area-card" onclick="state.level=2;state.areaIdx='+i+';expandedGroups={};render()">'
        +'<div class="area-name">'+a.name+'</div>'
        +'<div class="area-kpi-row">'
        +'<div class="area-kpi"><div class="area-kpi-lbl">Presupuesto</div><div class="area-kpi-val">'+fmtK(ap*1e6)+'</div></div>'
        +'<div class="area-kpi"><div class="area-kpi-lbl">Ejecutado</div><div class="area-kpi-val">'+fmtK(ae*1e6)+'</div></div>'
        +'</div>'
        +progBar(ae,ap,color)
        +'<div class="grp-mini">'+grpH+'</div>'
        +'<div class="area-footer">'+semTag(p)+'<span class="area-link">Ver detalle \u2192</span></div>'
        +'</div>';
    }).join('')+'</div>';

  var chartH2 = dep.areas.length * 34 + 50;
  var chartH = '<div class="chart-wrap">'
    +'<div class="chart-title">Presupuesto vs Ejecutado por \u00c1rea</div>'
    +'<div style="position:relative;height:'+chartH2+'px"><canvas id="chart-l1"></canvas></div></div>';

  c.innerHTML = buildFilterBar()
    +bc([{t:'Todas las dependencias',fn:'state.level=0;render()'},{t:dep.name}])
    +cardsH+chartH;

  setTimeout(function(){
    var el=document.getElementById('chart-l1');
    if (!el) return;
    if (_cL1) { _cL1.destroy(); _cL1=null; }
    _cL1 = new Chart(el.getContext('2d'),{
      type:'bar',
      data:{
        labels:dep.areas.map(function(a){return a.name;}),
        datasets:[
          {label:'Presupuesto',data:dep.areas.map(function(a){return getPresup(a)*1e6;}),backgroundColor:'rgba(0,0,0,0.09)',borderRadius:3},
          {label:'Ejecutado',data:dep.areas.map(function(a){return getEjec(a)*1e6;}),backgroundColor:color+'CC',borderRadius:3}
        ]
      },
      options:{
        indexAxis:'y',responsive:true,maintainAspectRatio:false,
        plugins:{legend:{labels:{font:{size:11}}}},
        scales:{
          x:{grid:{color:'#E8EDF2'},ticks:{font:{size:10},callback:function(v){return '$'+Math.round(v/1e6)+'M';}}},
          y:{grid:{display:false},ticks:{font:{size:10}}}
        }
      }
    });
  },0);
}

/* ══ NIVEL 3 ══ */
function renderL2(c) {
  var dep=DATA[state.depIdx], di=state.depIdx;
  var area=dep.areas[state.areaIdx];
  var color=depColor(di);
  var ap=getPresup(area), ae=getEjec(area), pp=pct(ae,ap), saldo=ap-ae;

  var kpiH = '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:10px;margin-bottom:16px">'
    +[{l:'Presupuesto',v:fmtK(ap*1e6)},{l:'Ejecutado',v:fmtK(ae*1e6)},
      {l:'Saldo',v:fmtK(saldo*1e6),neg:saldo<0},{l:'Ejecuci\u00f3n',tag:1}
    ].map(function(k){
      return '<div style="background:#F8FAFC;border:1px solid #E4E8EF;border-radius:8px;padding:9px 12px">'
        +'<div style="font-size:9px;font-weight:700;color:#8A99AA;text-transform:uppercase;letter-spacing:.06em;margin-bottom:3px">'+k.l+'</div>'
        +(k.tag?semTag(pp):'<div style="font-size:17px;font-weight:700;color:'+(k.neg?'#791F1F':'#0F1924')+'">'+k.v+'</div>')
        +'</div>';
    }).join('')+'</div>';

  /* collect totals for chart */
  var allItems=[], totP=0, totE=0;
  area.groups.forEach(function(g){
    g.accounts.forEach(function(acc){
      acc.items.forEach(function(item){
        if (item.presup===0 && item.ejec===0) return;
        totP+=item.presup; totE+=item.ejec;
        allItems.push({name:item.name,presup:item.presup,ejec:item.ejec});
      });
    });
  });

  /* expandable table */
  var rows='';
  area.groups.forEach(function(g, gi){
    var gp=getPresup(g), ge=getEjec(g), gpct=pct(ge,gp);
    var lbl=g.name.replace(/_/g,' ');
    var open = expandedGroups[gi] === true;
    var icon = open ? '\u25bc' : '\u25ba';

    /* collapsed: show item name chips as preview */
    var preview='';
    if (!open) {
      var chips=[];
      g.accounts.forEach(function(acc){ acc.items.forEach(function(item){ if(item.presup||item.ejec) chips.push(item.name); }); });
      if (chips.length) {
        var shown=chips.slice(0,4).map(function(n){ return '<span style="background:#E8EDF2;border-radius:4px;padding:1px 6px;font-size:10px;color:#374151;white-space:nowrap">'+n+'</span>'; }).join(' ');
        var more=chips.length>4?' <span style="font-size:10px;color:#94A3B8">+'+(chips.length-4)+' m\u00e1s</span>':'';
        preview='<div style="display:flex;flex-wrap:wrap;gap:4px;margin-top:6px">'+shown+more+'</div>';
      }
    }

    rows+='<tr class="row-grp" style="cursor:pointer" onclick="tglGrp('+gi+')">'
      +'<td colspan="5">'
      +'<div style="display:flex;align-items:center;justify-content:space-between;gap:8px">'
      +'<div><span style="font-size:9px;margin-right:5px;color:#6B7A8D">'+icon+'</span><span class="badge-g">G</span>'+lbl+'</div>'
      +'<div style="display:flex;gap:8px;align-items:center;flex-shrink:0">'+fmtFull(gp*1e6)+' / '+fmtFull(ge*1e6)+'\u00a0'+semTag(gpct)+'</div>'
      +'</div>'+(preview||'')+'</td></tr>';

    if (open) {
      g.accounts.forEach(function(acc){
        acc.items.forEach(function(item){
          if (item.presup===0 && item.ejec===0) return;
          var ip=item.presup, ie=item.ejec, ipct=pct(ie,ip);
          rows+='<tr class="row-item">'
            +'<td>'+item.name+'</td>'
            +'<td class="r">'+fmtFull(ip*1e6)+'</td>'
            +'<td class="r">'+fmtFull(ie*1e6)+'</td>'
            +'<td class="r">'+variCell(ip,ie)+'</td>'
            +'<td class="bar-cell r">'
            +'<div style="font-size:11px;font-weight:600;color:'+semColor(ipct)+'">'+ipct+'%</div>'
            +'<div class="emb-track"><div class="emb-fill" style="width:'+Math.min(ipct,100)+'%;background:'+(ipct>100?'#E24B4A':color)+'"></div></div>'
            +semTag(ipct)+'</td></tr>';
        });
      });
    }
  });

  var tpct=pct(totE,totP);
  rows+='<tr class="row-total"><td>TOTAL \u00c1REA</td>'
    +'<td class="r">'+fmtFull(totP*1e6)+'</td>'
    +'<td class="r">'+fmtFull(totE*1e6)+'</td>'
    +'<td class="r">'+variCell(totP,totE)+'</td>'
    +'<td class="r">'+tpct+'%</td></tr>';

  var top20=allItems.slice().sort(function(a,b){return b.ejec-a.ejec;}).slice(0,20);
  var tableH='<table class="detail-table"><thead><tr>'
    +'<th>Descripci\u00f3n <span style="font-weight:400;color:#B0BEC5">(clic en \u25ba para expandir)</span></th>'
    +'<th class="r">Presupuesto</th><th class="r">Ejecutado</th>'
    +'<th class="r">Varianza</th><th class="r bar-cell">Ejecuci\u00f3n</th>'
    +'</tr></thead><tbody>'+rows+'</tbody></table>';

  var chartPxH=top20.length*28+50;
  var chartH='<div class="chart-wrap"><div class="chart-title">Ejecuci\u00f3n por \u00cdtem \u2014 Top 20</div>'
    +'<div style="position:relative;height:'+chartPxH+'px"><canvas id="chart-l2"></canvas></div></div>';

  var hBtn='<button style="background:#EFF6FF;color:#185FA5;border:1px solid #BFDBFE;border-radius:7px;padding:7px 14px;font-size:12px;font-weight:600;cursor:pointer;margin-bottom:16px" onclick="showAreaHomo()">Ver Homologaciones del \u00c1rea</button>';

  c.innerHTML=buildFilterBar()
    +bc([{t:'Todas las dependencias',fn:'state.level=0;render()'},{t:dep.name,fn:'state.level=1;render()'},{t:area.name}])
    +hBtn+kpiH+tableH+chartH;

  setTimeout(function(){
    var el=document.getElementById('chart-l2');
    if (!el) return;
    if (_cL2) { _cL2.destroy(); _cL2=null; }
    _cL2 = new Chart(el.getContext('2d'),{
      type:'bar',
      data:{
        labels:top20.map(function(it){return it.name;}),
        datasets:[{
          label:'Ejecutado',
          data:top20.map(function(it){return it.ejec*1e6;}),
          backgroundColor:top20.map(function(it){return it.ejec>it.presup?'#E24B4A':color+'CC';}),
          borderRadius:3
        }]
      },
      options:{
        indexAxis:'y',responsive:true,maintainAspectRatio:false,
        plugins:{legend:{display:false}},
        scales:{
          x:{grid:{color:'#E8EDF2'},ticks:{font:{size:10},callback:function(v){return '$'+Math.round(v/1e6)+'M';}}},
          y:{grid:{display:false},ticks:{font:{size:9}}}
        }
      }
    });
  },0);
}

/* ── Homo ── */
function showAreaHomo() {
  homoFilter = DATA[state.depIdx].areas[state.areaIdx].name;
  openWC('homo');
}
var _homoFormOpen = false;

function renderHomo() {
  var items = homoFilter ? HOMOLOGATIONS.filter(function(h){return h.area===homoFilter;}) : HOMOLOGATIONS;
  var badge = homoFilter
    ? '<span class="homo-filter-badge">Filtrado: '+homoFilter+'</span>'
      +'<button class="btn-ghost" style="margin-left:8px" onclick="homoFilter=null;renderHomo()">Ver todas</button>'
    : '';
  document.getElementById('homo-header-extra').innerHTML = badge;

  /* ── Manual entry form ── */
  var areaDefault = homoFilter || '';
  var formH = _homoFormOpen
    ? '<div id="homo-form" style="background:#F8FAFC;border:1px solid #E4E8EF;border-radius:10px;padding:16px 18px;margin-bottom:16px">'
      +'<div style="font-size:12px;font-weight:700;color:#1A2535;margin-bottom:12px">Nueva homologaci\u00f3n manual</div>'
      +'<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:10px">'
      +'<div><label style="font-size:10px;font-weight:700;color:#8A99AA;text-transform:uppercase;display:block;margin-bottom:3px">\u00c1rea</label>'
      +'<input id="hf-area" class="homo-input" style="width:100%" placeholder="\u00c1rea responsable" value="'+areaDefault+'"/></div>'
      +'<div><label style="font-size:10px;font-weight:700;color:#8A99AA;text-transform:uppercase;display:block;margin-bottom:3px">Cuenta</label>'
      +'<input id="hf-cuenta" class="homo-input" style="width:100%" placeholder="Nombre de cuenta"/></div>'
      +'</div>'
      +'<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:12px">'
      +'<div><label style="font-size:10px;font-weight:700;color:#8A99AA;text-transform:uppercase;display:block;margin-bottom:3px">Texto ejecutado (original)</label>'
      +'<input id="hf-orig" class="homo-input" style="width:100%" placeholder="Descripci\u00f3n tal como aparece en ejecuci\u00f3n"/></div>'
      +'<div><label style="font-size:10px;font-weight:700;color:#8A99AA;text-transform:uppercase;display:block;margin-bottom:3px">Mapear a (sugerencia)</label>'
      +'<input id="hf-sugg" class="homo-input" style="width:100%" placeholder="Descripci\u00f3n presupuestada a la que debe imputarse"/></div>'
      +'</div>'
      +'<div style="display:flex;gap:8px">'
      +'<button class="btn-primary btn-sm" onclick="addManualHomo()" style="padding:6px 16px;font-size:12px">Guardar homologaci\u00f3n</button>'
      +'<button class="btn-ghost btn-sm" onclick="_homoFormOpen=false;renderHomo()" style="padding:6px 14px;font-size:12px">Cancelar</button>'
      +'</div></div>'
    : '';

  document.getElementById('homo-form-wrap').innerHTML = formH;

  var b = document.getElementById('homo-body');
  if (!items.length) {
    b.innerHTML='<tr><td colspan="6" style="text-align:center;color:#8A99AA;padding:24px">Sin homologaciones — agrega una manualmente con el bot\u00f3n de arriba.</td></tr>';
    return;
  }
  b.innerHTML = items.map(function(h){
    var ri=HOMOLOGATIONS.indexOf(h), sc='';
    if (h.status==='approved') {
      sc='<span style="font-weight:600;color:#059669">&#10003; Aprobada</span> <button class="btn-ghost btn-sm btn-reject" onclick="updH('+ri+',2)">Rechazar</button>';
    } else if (h.status==='rejected') {
      sc='<span style="font-weight:600;color:#DC2626">&#10007; Rechazada</span> <button class="btn-ghost btn-sm btn-approve" onclick="updH('+ri+',1)">Aprobar</button>';
    } else {
      sc='<button class="btn-primary btn-sm btn-approve" onclick="updH('+ri+',1)">Aprobar</button> <button class="btn-ghost btn-sm btn-reject" onclick="updH('+ri+',2)">Rechazar</button>';
    }
    return '<tr>'
      +'<td style="font-size:11px"><b>'+h.area+'</b><br><span style="color:#8A99AA">'+h.cuenta+'</span></td>'
      +'<td>'+h.original+'</td>'
      +'<td style="color:#185FA5;font-weight:600">'+h.suggested+'</td>'
      +'<td><input class="homo-input" value="'+h.suggested+'" onchange="setManual('+ri+',this.value)"/></td>'
      +'<td><span class="pill" style="background:#E6F1FB;color:#185FA5">'+h.confidence+'%</span></td>'
      +'<td>'+sc+'</td></tr>';
  }).join('');
}

function addManualHomo() {
  var area  = (document.getElementById('hf-area')  ||{}).value||'';
  var cta   = (document.getElementById('hf-cuenta') ||{}).value||'';
  var orig  = (document.getElementById('hf-orig')   ||{}).value||'';
  var sugg  = (document.getElementById('hf-sugg')   ||{}).value||'';
  if (!area || !orig || !sugg) { alert('Completa al menos \u00c1rea, Texto original y Sugerencia.'); return; }
  HOMOLOGATIONS.push({ area:area, cuenta:cta||'(Manual)', original:orig, suggested:sugg, confidence:100, status:'approved' });
  _homoFormOpen = false;
  renderHomo();
}

function updH(idx,s){ HOMOLOGATIONS[idx].status=s===1?'approved':s===2?'rejected':'pending'; renderHomo(); }
function setManual(idx,val){ HOMOLOGATIONS[idx].suggested=val; }
function approveAllArea(){
  var a=homoFilter;
  HOMOLOGATIONS.forEach(function(h){if(!a||h.area===a)h.status='approved';});
  renderHomo();
}
function saveHomo(){
  var blob=new Blob([JSON.stringify(HOMOLOGATIONS,null,2)],{type:'application/json'});
  var a=document.createElement('a'); a.href=URL.createObjectURL(blob); a.download='homologations.json'; a.click();
  alert('Descargado. Reemplaza el archivo y re-ejecuta el script para aplicar los cambios.');
}

render();
""")

    HTML = (
        '<!DOCTYPE html>\n<html lang="es">\n<head>\n'
        '<meta charset="UTF-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Fortia \u00b7 Monitor Presupuestal 2026</title>\n'
        '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">\n'
        '<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>\n'
        '<style>' + CSS + '</style>\n</head>\n<body>\n'
        '<div class="sidebar">\n'
        '  <div class="side-logo">'
        '<div class="side-logo-icon">F</div>'
        '<div class="side-logo-text">'
        '<div class="side-logo-main">FORTIA <span style="color:#3B82F6">\u00b7</span> ERP</div>'
        '<div class="side-logo-sub">Gesti\u00f3n Presupuestal 2026</div>'
        '</div></div>\n'
        '  <div class="side-menu">\n'
        '    <div class="side-section">An\u00e1lisis</div>\n'
        '    <div class="menu-item active" id="m-dash" onclick="openWC(\'dash\')">'
        '<span class="menu-icon">&#9673;</span><span class="menu-text">Monitor Presupuesto</span></div>\n'
        '    <div class="side-section">Modificaciones</div>\n'
        '    <div class="menu-item" id="m-homo" onclick="openWC(\'homo\')">'
        '<span class="menu-icon">&#10038;</span><span class="menu-text">Homologaci\u00f3n IA</span></div>\n'
        '    <div class="side-section">Pr\u00f3ximamente</div>\n'
        '    <div class="menu-item disabled"><span class="menu-icon">&#9675;</span><span class="menu-text">Monitor Ingresos</span></div>\n'
        '    <div class="menu-item disabled"><span class="menu-icon">&#9675;</span><span class="menu-text">Usuarios y Permisos</span></div>\n'
        '  </div>\n'
        '  <div class="side-footer">v3.1 \u00b7 Fortia Minerals 2026</div>\n'
        '</div>\n'
        '<div class="main">\n'
        '  <div class="top-bar">\n'
        '    <div id="wc-title" class="top-title">Monitor Presupuestal</div>\n'
        '    <div style="font-size:11px;color:#8A99AA">CI Fortia Minerals S.A.S.</div>\n'
        '  </div>\n'
        '  <div class="content">\n'
        '    <div id="wc-dash"></div>\n'
        '    <div id="wc-homo" class="hidden">\n'
        '      <div class="homo-island">\n'
        '        <div class="homo-header">\n'
        '          <div>\n'
        '            <p class="homo-title">Homologaciones IA</p>\n'
        '            <p class="homo-sub">Comparaci\u00f3n contextual por \u00c1rea + Cuenta. Aprueba, rechaza o edita manualmente.</p>\n'
        '          </div>\n'
        '          <div id="homo-header-extra" style="display:flex;align-items:center;gap:6px"></div>\n'
        '        </div>\n'
        '        <div id="homo-form-wrap"></div>\n'
        '        <div style="overflow-x:auto">\n'
        '          <table class="homo-table">\n'
        '            <thead><tr><th>\u00c1rea / Cuenta</th><th>Texto Ejecutado</th>'
        '<th>Sugerencia IA</th><th>Editar</th><th>Confianza</th><th>Acci\u00f3n</th></tr></thead>\n'
        '            <tbody id="homo-body"></tbody>\n'
        '          </table>\n'
        '        </div>\n'
        '        <div style="margin-top:16px;display:flex;gap:10px;flex-wrap:wrap;align-items:center">\n'
        '          <button class="btn-primary" onclick="saveHomo()">&#8659; Exportar Homologaciones</button>\n'
        '          <button class="btn-ghost" onclick="approveAllArea()">Aprobar todas visibles</button>\n'
        '          <button class="btn-ghost" style="border-color:#BFDBFE;color:#185FA5" onclick="_homoFormOpen=true;renderHomo()">&#43; Agregar manual</button>\n'
        '        </div>\n'
        '      </div>\n'
        '    </div>\n'
        '  </div>\n'
        '</div>\n'
        '<script>' + JS + '</script>\n'
        '</body></html>'
    )

    with open('presupuesto_ejecucion.html', 'w', encoding='utf-8') as f:
        f.write(HTML)
    logging.info("HTML generado.")


if __name__ == "__main__": main()
