import pandas as pd
import json
import numpy as np
import logging
import os
import textdistance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(filename='process_logs.log', level=logging.INFO, format='%(asctime)s - %(message)s', filemode='w', encoding='utf-8')

def clean(n, default="Global"):
    if pd.isna(n) or str(n).strip() == "": return default
    return str(n).strip()

def run_contextual_ai(df, threshold=0.8):
    suggestions = []
    grouped = df.groupby(['Responsable', 'Nombre de Cuenta'])
    for (area, cuenta), sub_df in grouped:
        budgeted = sub_df[sub_df['Presupuesto 2026'] > 0]['Descripción'].unique().tolist()
        executed = sub_df[sub_df['Ejecutado 2026'] > 0]['Descripción'].unique().tolist()
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
                    "suggested": budgeted[best_idx], "confidence": round(confidence * 100, 1), "status": "pending"})
    return suggestions

def main():
    try:
        logging.info("Iniciando procesamiento de Work Centers...")
        df = pd.read_excel('Informe Presupuesto Ejecutado 2026 - Consolidado.xlsx')
        df['Presupuesto 2026'] = df['Presupuesto 2026'].fillna(0)
        df['Ejecutado 2026'] = df['Ejecutado 2026'].fillna(0)
        df['presup'] = df['Presupuesto 2026'] / 1000000
        df['ejec'] = df['Ejecutado 2026'] / 1000000
        for col in ['Dependencia', 'Responsable', 'Grupo', 'Nombre de Cuenta', 'Descripción']:
            df[col] = df[col].apply(lambda x: clean(x, f"({col} N/A)"))

        homo_decisions = []
        if os.path.exists('homologations.json'):
            with open('homologations.json', 'r', encoding='utf-8') as f: homo_decisions = json.load(f)
        ai_suggestions = run_contextual_ai(df)
        final_homo = {f"{h['area']}|{h['cuenta']}|{h['original']}": h for h in homo_decisions}
        for s in ai_suggestions:
            key = f"{s['area']}|{s['cuenta']}|{s['original']}"
            if key not in final_homo: final_homo[key] = s
        homo_list = list(final_homo.values())
        with open('homologations.json', 'w', encoding='utf-8') as f: json.dump(homo_list, f, indent=2, ensure_ascii=False)

        approved_map = {(h['area'], h['cuenta'], h['original']): h['suggested'] for h in homo_list if h['status'] == 'approved'}
        def apply_homo(row):
            key = (row['Responsable'], row['Nombre de Cuenta'], row['Descripción'])
            return approved_map.get(key, row['Descripción'])
        df['Desc_Final'] = df.apply(apply_homo, axis=1)

        data = []
        for dep_n, d_df in df.groupby('Dependencia'):
            dep = {'name': dep_n, 'presup': round(d_df['presup'].sum(), 2), 'ejec': round(d_df['ejec'].sum(), 2), 'areas': []}
            for resp_n, a_df in d_df.groupby('Responsable'):
                area = {'name': resp_n, 'presup': round(a_df['presup'].sum(), 2), 'ejec': round(a_df['ejec'].sum(), 2), 'groups': []}
                for grp_n, g_df in a_df.groupby('Grupo'):
                    grp = {'name': grp_n, 'presup': round(g_df['presup'].sum(), 2), 'ejec': round(g_df['ejec'].sum(), 2), 'accounts': []}
                    for acc_n, acc_df in g_df.groupby('Nombre de Cuenta'):
                        acc = {'name': acc_n, 'presup': round(acc_df['presup'].sum(), 2), 'ejec': round(acc_df['ejec'].sum(), 2), 'items': []}
                        for item_n, i_df in acc_df.groupby('Desc_Final'):
                            acc['items'].append({'name': clean(item_n), 'presup': round(i_df['presup'].sum(), 2), 'ejec': round(i_df['ejec'].sum(), 2)})
                        acc['items'].sort(key=lambda x: x['presup'], reverse=True)
                        grp['accounts'].append(acc)
                    area['groups'].append(grp)
                dep['areas'].append(area)
            data.append(dep)

        def add_ids(node, prefix):
            node['id'] = prefix
            for k in ['areas', 'groups', 'accounts']:
                if k in node:
                    for i, child in enumerate(node[k]): add_ids(child, f"{prefix}_{k[0]}{i}")
        for i, d in enumerate(data): add_ids(d, f"d{i}")

        write_workcenter_html(data, homo_list)
        print("SUCCESS")
    except Exception as e:
        logging.error(f"FATAL: {e}", exc_info=True)
        print(f"ERROR: {e}")


def write_workcenter_html(data, homologations):
    js_data = json.dumps(data, ensure_ascii=False)
    js_homo = json.dumps(homologations, ensure_ascii=False)

    CSS = """
*, *::before, *::after { box-sizing: border-box; }
body { font-family: 'Inter', sans-serif; background: #F0F4F8; color: #1A2535; margin: 0; display: flex; min-height: 100vh; }

/* Sidebar */
.sidebar { width: 232px; background: #FFFFFF; border-right: 1px solid #E4E8EF; display: flex; flex-direction: column; position: fixed; height: 100vh; }
.side-logo { padding: 18px 20px; font-size: 14px; font-weight: 800; color: #1A2535; border-bottom: 1px solid #E4E8EF; letter-spacing: .5px; }
.side-logo span { color: #185FA5; }
.side-section { padding: 14px 16px 4px; font-size: 10px; font-weight: 700; color: #B0BEC5; letter-spacing: .1em; text-transform: uppercase; }
.side-menu { flex: 1; padding: 6px 0; overflow-y: auto; }
.menu-item { padding: 9px 16px; cursor: pointer; display: flex; align-items: center; gap: 9px; color: #6B7A8D; font-weight: 500; font-size: 13px; border-radius: 6px; margin: 2px 8px; transition: background .12s, color .12s; }
.menu-item:hover { color: #1A2535; background: #F0F4F8; }
.menu-item.active { background: #EFF6FF; color: #185FA5; font-weight: 600; }
.menu-item.disabled { opacity: 0.3; cursor: default; }
.side-footer { padding: 14px 20px; font-size: 10px; color: #C0CCDA; border-top: 1px solid #E4E8EF; }

/* Main */
.main { flex: 1; margin-left: 232px; display: flex; flex-direction: column; min-height: 100vh; }
.top-bar { height: 52px; background: #FFFFFF; border-bottom: 1px solid #E4E8EF; display: flex; align-items: center; padding: 0 24px; justify-content: space-between; position: sticky; top: 0; z-index: 100; }
.top-title { font-weight: 700; font-size: 14px; color: #1A2535; }
.top-meta  { font-size: 11px; color: #8A99AA; }
.content { padding: 20px 24px; flex: 1; }
.hidden { display: none; }

/* Island */
.dd-island { background: #FFFFFF; border-radius: 12px; padding: 20px 24px 24px; box-shadow: 0 1px 4px rgba(0,0,0,.07); border: 1px solid #E4E8EF; }

/* Breadcrumb */
.dd-breadcrumb { display: flex; align-items: center; gap: 6px; font-size: 13px; color: #6B7A8D; margin-bottom: 18px; flex-wrap: wrap; }
.dd-bc-link   { color: #185FA5; cursor: pointer; font-weight: 500; }
.dd-bc-link:hover { text-decoration: underline; }
.dd-bc-sep    { color: #C0CCDA; }
.dd-bc-active { color: #1A2535; font-weight: 600; }

/* Grids */
.dd-kpi-grid   { display: grid; grid-template-columns: repeat(auto-fill, minmax(155px,1fr)); gap: 12px; margin-bottom: 18px; }
.dd-cards-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(210px,1fr)); gap: 14px; margin-bottom: 20px; }
.dd-cards-sm   { grid-template-columns: repeat(auto-fill, minmax(180px,1fr)); }

/* Cards */
.dd-card { background: #FFFFFF; border: 1px solid #E4E8EF; border-radius: 10px; padding: 16px 14px 12px; cursor: pointer; transition: box-shadow .15s, transform .12s; display: flex; flex-direction: column; gap: 5px; box-shadow: 0 1px 3px rgba(0,0,0,.06); }
.dd-card:hover { box-shadow: 0 4px 14px rgba(0,0,0,.10); transform: translateY(-2px); }
.dd-card-sm  { padding: 13px 13px 11px; }
.dd-card-kpi { cursor: default; background: #F8FAFC; }
.dd-card-kpi:hover { transform: none; box-shadow: 0 1px 3px rgba(0,0,0,.06); }
.dd-card-header { display: flex; align-items: center; gap: 7px; margin-bottom: 1px; }
.dd-dot { width: 9px; height: 9px; border-radius: 3px; flex-shrink: 0; }
.dd-card-name { font-size: 12px; font-weight: 600; color: #1A2535; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.dd-card-amount { font-size: 22px; font-weight: 700; color: #0F1924; line-height: 1.15; }
.dd-card-sm .dd-card-amount { font-size: 18px; }
.dd-card-sub { font-size: 11px; color: #8A99AA; }
.dd-bar-wrap { height: 3px; background: #E8EDF2; border-radius: 2px; overflow: hidden; margin: 2px 0 3px; }
.dd-bar { height: 100%; border-radius: 2px; transition: width .4s; }
.dd-card-subitems { display: flex; flex-direction: column; gap: 3px; border-top: 1px solid #EEF1F5; padding-top: 7px; margin-top: 1px; }
.dd-card-subrow { display: flex; justify-content: space-between; align-items: center; font-size: 11px; }
.dd-card-subname { color: #6B7A8D; }
.dd-card-subpct  { color: #2E3A4A; font-weight: 600; }
.dd-card-link { font-size: 11px; color: #185FA5; margin-top: 5px; font-weight: 500; }
.dd-card:hover .dd-card-link { text-decoration: underline; }

/* Chart */
.dd-chart-section { background: #F8FAFC; border: 1px solid #E4E8EF; border-radius: 10px; padding: 16px 20px 12px; margin-top: 4px; }

/* L3 area detail */
.dd-area-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px; flex-wrap: wrap; gap: 10px; }
.dd-area-title  { font-size: 16px; font-weight: 700; color: #1A2535; }
.dd-homo-btn { background: #EFF6FF; color: #185FA5; border: 1px solid #BFDBFE; border-radius: 7px; padding: 7px 14px; font-size: 12px; font-weight: 600; cursor: pointer; }
.dd-homo-btn:hover { background: #DBEAFE; }
.dd-section-title { font-size: 11px; font-weight: 700; color: #8A99AA; letter-spacing: .07em; text-transform: uppercase; margin: 18px 0 10px; }

/* Summary boxes */
.dd-l3-summary { display: grid; grid-template-columns: repeat(auto-fill, minmax(120px,1fr)); gap: 10px; margin-bottom: 4px; }
.dd-l3-sum-item { background: #F8FAFC; border: 1px solid #E4E8EF; border-radius: 8px; padding: 10px 12px; }
.dd-l3-sum-label { font-size: 10px; font-weight: 700; color: #8A99AA; letter-spacing: .05em; text-transform: uppercase; margin-bottom: 3px; }
.dd-l3-sum-val   { font-size: 17px; font-weight: 700; color: #0F1924; }
.dd-pos { color: #059669; }
.dd-neg { color: #DC2626; }

/* Detail table */
.dd-detail-table { width: 100%; border-collapse: collapse; font-size: 13px; border: 1px solid #E4E8EF; border-radius: 10px; overflow: hidden; }
.dd-detail-table th { background: #F5F7FA; color: #8A99AA; font-size: 10px; font-weight: 700; letter-spacing: .07em; text-transform: uppercase; padding: 9px 12px; border-bottom: 1px solid #E4E8EF; text-align: left; }
.dd-group-row td { background: #EFF6FF; color: #185FA5; font-size: 10px; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; padding: 6px 12px; border-top: 2px solid #BFDBFE; border-bottom: 1px solid #BFDBFE; }
.dd-acc-row td   { background: #F8FAFC; color: #6B7A8D; font-size: 11px; font-weight: 600; padding: 5px 12px 5px 24px; border-bottom: 1px solid #EEF1F5; }
.dd-item-row td  { padding: 9px 12px 9px 32px; border-bottom: 1px solid #F0F3F7; color: #2E3A4A; vertical-align: middle; }
.dd-item-row:last-child td { border-bottom: none; }
.dd-item-row:hover td { background: #FAFBFC; }
.dd-val-cell { text-align: right; font-weight: 600; color: #0F1924; white-space: nowrap; }
.dd-pct-cell { text-align: right; font-size: 11px; color: #8A99AA; white-space: nowrap; }
.dd-bar-cell { width: 120px; padding-right: 12px !important; }

/* Homo */
.homo-island { background: #FFFFFF; border-radius: 12px; padding: 20px 24px 24px; box-shadow: 0 1px 4px rgba(0,0,0,.07); border: 1px solid #E4E8EF; }
.homo-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px; flex-wrap: wrap; gap: 8px; }
.homo-title  { font-size: 15px; font-weight: 700; color: #1A2535; margin: 0; }
.homo-filter-badge { background: #FEF3C7; color: #B45309; border: 1px solid #FDE68A; border-radius: 20px; padding: 3px 10px; font-size: 11px; font-weight: 600; }
.homo-sub    { font-size: 12px; color: #8A99AA; margin: 0 0 16px; }
.homo-table  { width: 100%; border-collapse: collapse; font-size: 12px; border: 1px solid #E4E8EF; border-radius: 10px; overflow: hidden; }
.homo-table th { background: #F5F7FA; color: #8A99AA; font-size: 10px; font-weight: 700; letter-spacing: .07em; text-transform: uppercase; padding: 9px 12px; border-bottom: 1px solid #E4E8EF; text-align: left; }
.homo-table td { padding: 9px 12px; border-bottom: 1px solid #F0F3F7; color: #2E3A4A; vertical-align: middle; }
.homo-table tr:last-child td { border-bottom: none; }
.btn-primary { background: #185FA5; color: #fff; border: none; padding: 8px 18px; border-radius: 7px; cursor: pointer; font-weight: 600; font-size: 13px; }
.btn-primary:hover { background: #1352914; }
.btn-ghost   { background: transparent; color: #6B7A8D; border: 1px solid #E4E8EF; padding: 7px 14px; border-radius: 7px; cursor: pointer; font-size: 12px; font-weight: 500; }
.btn-ghost:hover { background: #F0F4F8; }
.pill { padding: 2px 8px; border-radius: 20px; font-size: 11px; font-weight: 700; }
"""

    JS = ("""
const DATA = """ + js_data + """;
const HOMOLOGATIONS = """ + js_homo + """;
let state = { level: 0, depIdx: 0, areaIdx: 0 };
let homoFilter = null;

const DD_PALETTE = [
  { color: '#185FA5' }, { color: '#0F6E56' }, { color: '#B45309' },
  { color: '#7C3AED' }, { color: '#B91C1C' }, { color: '#0369A1' }
];

function _ddShort(v) {
  var a = Math.abs(v || 0);
  if (a >= 1e9) return '$' + (a/1e9).toFixed(1) + 'MM';
  if (a >= 1e6) return '$' + (a/1e6).toFixed(0) + 'M';
  if (a >= 1e3) return '$' + (a/1e3).toFixed(0) + 'k';
  return '$0';
}

function _svgBarChart(items) {
  if (!items || !items.length) return '';
  var PL=150, PR=10, PT=10, PB=24, ROW=34, W=520;
  var H = items.length * ROW + PT + PB;
  var IW = W - PL - PR;
  var maxV = Math.max.apply(null, items.map(function(i){return Math.abs(i.value||0);}));
  if(maxV===0) maxV=1;
  var nTicks=5, raw=maxV/(nTicks-1), mag=Math.pow(10,Math.floor(Math.log10(raw)||0));
  var tick=Math.ceil(raw/mag)*mag, axMax=tick*(nTicks-1);
  var grid='', xLbls='', bars='';
  for(var t=0;t<nTicks;t++){
    var x=PL+(t/(nTicks-1))*IW, val=t*tick;
    grid  += '<line x1="'+x+'" y1="'+PT+'" x2="'+x+'" y2="'+(PT+items.length*ROW)+'" stroke="#E8EDF2" stroke-width="0.5"/>';
    xLbls += '<text x="'+x+'" y="'+(PT+items.length*ROW+14)+'" text-anchor="middle" fill="#B0BEC5" font-size="9" font-family="system-ui">'+_ddShort(val)+'</text>';
  }
  for(var i=0;i<items.length;i++){
    var it=items[i], y=PT+i*ROW, bH=18, bY=y+(ROW-bH)/2;
    var bW=axMax>0?(Math.abs(it.value||0)/axMax)*IW:0;
    var lbl=it.label.length>22?it.label.substring(0,22)+'...':it.label;
    bars += '<text x="'+(PL-8)+'" y="'+(bY+bH/2+4)+'" text-anchor="end" fill="#4B5563" font-size="11" font-family="system-ui">'+lbl+'</text>'
          + '<rect x="'+PL+'" y="'+bY+'" width="'+IW+'" height="'+bH+'" rx="3" fill="#F0F4F8"/>'
          + '<rect x="'+PL+'" y="'+bY+'" width="'+Math.max(bW,2)+'" height="'+bH+'" rx="3" fill="'+it.color+'" opacity="0.85"/>';
  }
  return '<div class="dd-chart-section"><svg viewBox="0 0 '+W+' '+H+'" style="width:100%;max-width:'+W+'px;display:block;overflow:visible">'+grid+bars+xLbls+'</svg></div>';
}

var pct = function(e,p){ return p===0?(e>0?100:0):Math.round(e/p*100); };
var barC = function(p){ return p<50?'#3B82F6':p<90?'#F59E0B':p<=100?'#059669':'#DC2626'; };
var pal = function(i){ return DD_PALETTE[i%DD_PALETTE.length].color; };

function openWC(id) {
  document.getElementById('wc-dash').classList.toggle('hidden', id!=='dash');
  document.getElementById('wc-homo').classList.toggle('hidden', id!=='homo');
  document.getElementById('m-dash').classList.toggle('active', id==='dash');
  document.getElementById('m-homo').classList.toggle('active', id==='homo');
  document.getElementById('wc-title').textContent = id==='dash' ? 'Monitor Presupuestal' : 'Homologaciones IA';
  if(id==='homo') renderHomo(); else render();
}

function render() {
  var c = document.getElementById('wc-dash');
  if(state.level===0) renderL0(c);
  else if(state.level===1) renderL1(c);
  else renderL2(c);
}

/* ── L0: dependencias ── */
function renderL0(c) {
  var tp=DATA.reduce(function(s,d){return s+d.presup;},0);
  var te=DATA.reduce(function(s,d){return s+d.ejec;},0);
  var pp=pct(te,tp);
  var kpis=[
    {lbl:'Presupuesto 2026', val:_ddShort(tp*1e6), sub:'Total aprobado'},
    {lbl:'Total Ejecutado',  val:_ddShort(te*1e6), sub:'Acumulado año'},
    {lbl:'Saldo Disponible', val:_ddShort((tp-te)*1e6), sub:'Por ejecutar'},
    {lbl:'Ejecución',        val:pp+'%', bar:pp}
  ];
  var kpiH = kpis.map(function(k){
    return '<div class="dd-card dd-card-kpi">'
      +'<div class="dd-card-sub">'+k.lbl+'</div>'
      +'<div class="dd-card-amount" style="font-size:20px">'+k.val+'</div>'
      +(k.sub?'<div class="dd-card-sub">'+k.sub+'</div>':'')
      +(k.bar!==undefined?'<div class="dd-bar-wrap"><div class="dd-bar" style="width:'+Math.min(k.bar,100)+'%;background:'+barC(k.bar)+'"></div></div>':'')
      +'</div>';
  }).join('');
  var depH = DATA.map(function(d,i){
    var p=pct(d.ejec,d.presup);
    return '<div class="dd-card" onclick="state.level=1;state.depIdx='+i+';render()">'
      +'<div class="dd-card-header"><div class="dd-dot" style="background:'+pal(i)+'"></div>'
      +'<div class="dd-card-name" title="'+d.name+'">'+d.name+'</div></div>'
      +'<div class="dd-card-amount">'+_ddShort(d.ejec*1e6)+'</div>'
      +'<div class="dd-card-sub">de '+_ddShort(d.presup*1e6)+'</div>'
      +'<div class="dd-bar-wrap"><div class="dd-bar" style="width:'+Math.min(p,100)+'%;background:'+barC(p)+'"></div></div>'
      +'<div class="dd-card-subrow"><span class="dd-card-subname">Ejecuci\u00f3n</span><span class="dd-card-subpct" style="color:'+barC(p)+'">'+p+'%</span></div>'
      +'<div class="dd-card-link">Ver \u00e1reas \u2192</div>'
      +'</div>';
  }).join('');
  var chartItems=DATA.map(function(d,i){return{label:d.name,value:d.ejec*1e6,color:pal(i)};});
  c.innerHTML='<div class="dd-island"><div class="dd-kpi-grid">'+kpiH+'</div>'
    +'<div class="dd-cards-grid">'+depH+'</div>'+_svgBarChart(chartItems)+'</div>';
}

/* ── L1: áreas/responsables ── */
function renderL1(c) {
  var dep=DATA[state.depIdx], di=state.depIdx;
  var areaH=dep.areas.map(function(a,i){
    var p=pct(a.ejec,a.presup);
    return '<div class="dd-card dd-card-sm" onclick="state.level=2;state.areaIdx='+i+';render()">'
      +'<div class="dd-card-header"><div class="dd-dot" style="background:'+pal(di+i+1)+'"></div>'
      +'<div class="dd-card-name" title="'+a.name+'">'+a.name+'</div></div>'
      +'<div class="dd-card-amount">'+_ddShort(a.ejec*1e6)+'</div>'
      +'<div class="dd-card-sub">de '+_ddShort(a.presup*1e6)+'</div>'
      +'<div class="dd-bar-wrap"><div class="dd-bar" style="width:'+Math.min(p,100)+'%;background:'+barC(p)+'"></div></div>'
      +'<div class="dd-card-subitems">'
      +'<div class="dd-card-subrow"><span class="dd-card-subname">Ejecuci\u00f3n</span><span class="dd-card-subpct" style="color:'+barC(p)+'">'+p+'%</span></div>'
      +'<div class="dd-card-subrow"><span class="dd-card-subname">Grupos</span><span class="dd-card-subpct">'+a.groups.length+'</span></div>'
      +'</div><div class="dd-card-link">Ver detalle \u2192</div></div>';
  }).join('');
  var chartItems=dep.areas.map(function(a,i){return{label:a.name,value:a.ejec*1e6,color:pal(di+i+1)};});
  c.innerHTML='<div class="dd-island">'
    +'<div class="dd-breadcrumb">'
    +'<span class="dd-bc-link" onclick="state.level=0;render()">Inicio</span>'
    +'<span class="dd-bc-sep">\u203a</span>'
    +'<span class="dd-bc-active">'+dep.name+'</span>'
    +'</div>'
    +'<div class="dd-cards-grid dd-cards-sm">'+areaH+'</div>'
    +_svgBarChart(chartItems)+'</div>';
}

/* ── L2: detalle de área ── */
function renderL2(c) {
  var dep=DATA[state.depIdx], di=state.depIdx;
  var area=dep.areas[state.areaIdx], ai=state.areaIdx;
  var saldo=area.presup-area.ejec, pp=pct(area.ejec,area.presup);

  /* grupo cards */
  var grpCards=area.groups.map(function(g,i){
    var p=pct(g.ejec,g.presup);
    var lbl=g.name.replace(/_/g,' ');
    return '<div class="dd-card dd-card-sm dd-card-kpi">'
      +'<div class="dd-card-header"><div class="dd-dot" style="background:'+pal(di+ai+i+2)+'"></div>'
      +'<div class="dd-card-name" title="'+lbl+'">'+lbl+'</div></div>'
      +'<div class="dd-card-amount">'+_ddShort(g.ejec*1e6)+'</div>'
      +'<div class="dd-card-sub">de '+_ddShort(g.presup*1e6)+'</div>'
      +'<div class="dd-bar-wrap"><div class="dd-bar" style="width:'+Math.min(p,100)+'%;background:'+barC(p)+'"></div></div>'
      +'<div class="dd-card-subrow"><span class="dd-card-subname">Ejecuci\u00f3n</span><span class="dd-card-subpct" style="color:'+barC(p)+'">'+p+'%</span></div>'
      +'</div>';
  }).join('');

  var chartGrp=area.groups.map(function(g,i){return{label:g.name.replace(/_/g,' '),value:g.ejec*1e6,color:pal(di+ai+i+2)};});

  /* tabla jerárquica */
  var rows='';
  area.groups.forEach(function(g){
    var grpLbl=g.name.replace(/_/g,' ');
    rows+='<tr class="dd-group-row"><td colspan="5">'+grpLbl+'</td></tr>';
    g.accounts.forEach(function(acc){
      rows+='<tr class="dd-acc-row"><td colspan="5">'+acc.name+'</td></tr>';
      acc.items.forEach(function(item){
        var ip=pct(item.ejec,item.presup);
        rows+='<tr class="dd-item-row">'
          +'<td>'+item.name+'</td>'
          +'<td class="dd-val-cell">'+_ddShort(item.presup*1e6)+'</td>'
          +'<td class="dd-val-cell">'+_ddShort(item.ejec*1e6)+'</td>'
          +'<td class="dd-bar-cell"><div class="dd-bar-wrap"><div class="dd-bar" style="width:'+Math.min(ip,100)+'%;background:'+barC(ip)+'"></div></div></td>'
          +'<td class="dd-pct-cell">'+ip+'%</td>'
          +'</tr>';
      });
    });
  });

  c.innerHTML='<div class="dd-island">'
    +'<div class="dd-breadcrumb">'
    +'<span class="dd-bc-link" onclick="state.level=0;render()">Inicio</span>'
    +'<span class="dd-bc-sep">\u203a</span>'
    +'<span class="dd-bc-link" onclick="state.level=1;render()">'+dep.name+'</span>'
    +'<span class="dd-bc-sep">\u203a</span>'
    +'<span class="dd-bc-active">'+area.name+'</span>'
    +'</div>'
    +'<div class="dd-area-header">'
    +'<span class="dd-area-title">'+area.name+'</span>'
    +'<button class="dd-homo-btn" onclick="showAreaHomo()">Ver Homologaciones del \u00c1rea</button>'
    +'</div>'
    +'<div class="dd-l3-summary">'
    +'<div class="dd-l3-sum-item"><div class="dd-l3-sum-label">Presupuesto</div><div class="dd-l3-sum-val">'+_ddShort(area.presup*1e6)+'</div></div>'
    +'<div class="dd-l3-sum-item"><div class="dd-l3-sum-label">Ejecutado</div><div class="dd-l3-sum-val">'+_ddShort(area.ejec*1e6)+'</div></div>'
    +'<div class="dd-l3-sum-item"><div class="dd-l3-sum-label">Saldo</div><div class="dd-l3-sum-val '+(saldo<0?'dd-neg':'dd-pos')+'">'+_ddShort(saldo*1e6)+'</div></div>'
    +'<div class="dd-l3-sum-item"><div class="dd-l3-sum-label">Ejecuci\u00f3n</div><div class="dd-l3-sum-val" style="color:'+barC(pp)+'">'+pp+'%</div></div>'
    +'</div>'
    +'<div class="dd-section-title">Informe por Grupo de Cuenta</div>'
    +'<div class="dd-cards-grid dd-cards-sm">'+grpCards+'</div>'
    +_svgBarChart(chartGrp)
    +'<div class="dd-section-title" style="margin-top:20px">\u00cdtems Detallados</div>'
    +'<table class="dd-detail-table">'
    +'<thead><tr>'
    +'<th>Descripci\u00f3n</th>'
    +'<th style="text-align:right">Presupuesto</th>'
    +'<th style="text-align:right">Ejecutado</th>'
    +'<th class="dd-bar-cell">Progreso</th>'
    +'<th style="text-align:right">%</th>'
    +'</tr></thead><tbody>'+rows+'</tbody></table>'
    +'</div>';
}

/* ── Homologaciones ── */
function showAreaHomo() {
  var area=DATA[state.depIdx].areas[state.areaIdx];
  homoFilter=area.name;
  openWC('homo');
}

function renderHomo() {
  var items=homoFilter
    ? HOMOLOGATIONS.filter(function(h){return h.area===homoFilter;})
    : HOMOLOGATIONS;
  var badge=homoFilter
    ? '<span class="homo-filter-badge">Filtrado: '+homoFilter+'</span>'
      +'<button class="btn-ghost" style="margin-left:8px" onclick="homoFilter=null;renderHomo()">Ver todas</button>'
    : '';
  document.getElementById('homo-header-extra').innerHTML=badge;
  var b=document.getElementById('homo-body');
  b.innerHTML=items.length===0
    ? '<tr><td colspan="5" style="text-align:center;color:#8A99AA;padding:24px">Sin homologaciones para esta \u00e1rea</td></tr>'
    : items.map(function(h,idx){
        var realIdx=HOMOLOGATIONS.indexOf(h);
        return '<tr>'
          +'<td style="font-size:11px"><b>'+h.area+'</b><br><span style="color:#8A99AA">'+h.cuenta+'</span></td>'
          +'<td>'+h.original+'</td>'
          +'<td style="color:#185FA5;font-weight:600">'+h.suggested+'</td>'
          +'<td><span class="pill" style="background:#E6F1FB;color:#185FA5">'+h.confidence+'%</span></td>'
          +'<td>'+(h.status==='pending'
            ?'<button class="btn-primary" style="padding:3px 10px;font-size:11px" onclick="updH('+realIdx+')">Aprobar</button>'
            :'<span style="font-weight:600;color:#059669">&#10003; Aprobada</span>')
          +'</td></tr>';
      }).join('');
}

function updH(idx){ HOMOLOGATIONS[idx].status='approved'; renderHomo(); }
function saveHomo(){
  var blob=new Blob([JSON.stringify(HOMOLOGATIONS,null,2)],{type:'application/json'});
  var a=document.createElement('a'); a.href=URL.createObjectURL(blob); a.download='homologations.json'; a.click();
  alert('Descargado. Reemplaza el archivo y vuelve a ejecutar el .BAT para aplicar los cambios.');
}

render();
""")

    html = (
        '<!DOCTYPE html>\n<html lang="es">\n<head>\n'
        '<meta charset="UTF-8"><title>Fortia \u00b7 Monitor Presupuestal 2026</title>\n'
        '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap" rel="stylesheet">\n'
        '<style>' + CSS + '</style>\n</head>\n<body>\n'
        '<div class="sidebar">\n'
        '  <div class="side-logo">FORTIA <span>\u00b7</span> ERP</div>\n'
        '  <div class="side-menu">\n'
        '    <div class="side-section">Gesti\u00f3n</div>\n'
        '    <div class="menu-item active" id="m-dash" onclick="openWC(\'dash\')">&#9673; Monitor Presupuesto</div>\n'
        '    <div class="menu-item" id="m-homo" onclick="openWC(\'homo\')">&#10038; Homologaci\u00f3n IA</div>\n'
        '    <div class="side-section">Pr\u00f3ximamente</div>\n'
        '    <div class="menu-item disabled">&#9675; Monitor Ingresos</div>\n'
        '    <div class="menu-item disabled">&#9675; Usuarios y Permisos</div>\n'
        '  </div>\n'
        '  <div class="side-footer">v2.2 \u00b7 Fortia Minerals 2026</div>\n'
        '</div>\n'
        '<div class="main">\n'
        '  <div class="top-bar">\n'
        '    <div id="wc-title" class="top-title">Monitor Presupuestal</div>\n'
        '    <div class="top-meta">Gesti\u00f3n Presupuestal 2026</div>\n'
        '  </div>\n'
        '  <div class="content">\n'
        '    <div id="wc-dash"></div>\n'
        '    <div id="wc-homo" class="hidden">\n'
        '      <div class="homo-island">\n'
        '        <div class="homo-header">\n'
        '          <div>\n'
        '            <p class="homo-title">Homologaciones IA</p>\n'
        '            <p class="homo-sub">Comparaci\u00f3n contextual por \u00c1rea + Cuenta para m\u00e1xima precisi\u00f3n.</p>\n'
        '          </div>\n'
        '          <div id="homo-header-extra" style="display:flex;align-items:center;gap:6px"></div>\n'
        '        </div>\n'
        '        <div style="overflow-x:auto">\n'
        '          <table class="homo-table">\n'
        '            <thead><tr><th>\u00c1rea / Cuenta</th><th>Ejecuci\u00f3n (Texto)</th>'
        '<th>Sugerencia Presupuesto</th><th>Confianza</th><th>Acci\u00f3n</th></tr></thead>\n'
        '            <tbody id="homo-body"></tbody>\n'
        '          </table>\n'
        '        </div>\n'
        '        <div style="margin-top:16px;display:flex;gap:10px">\n'
        '          <button class="btn-primary" onclick="saveHomo()">Sincronizar y Aplicar</button>\n'
        '        </div>\n'
        '      </div>\n'
        '    </div>\n'
        '  </div>\n'
        '</div>\n'
        '<script>' + JS + '</script>\n'
        '</body></html>'
    )

    with open('presupuesto_ejecucion.html', 'w', encoding='utf-8') as f:
        f.write(html)


if __name__ == "__main__": main()
