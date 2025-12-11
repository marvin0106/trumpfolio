import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px

# --- Imports und Setup ---
try:
    from pypfopt import risk_models, EfficientFrontier, black_litterman, objective_functions
except ImportError:
    st.error("Fehler: 'PyPortfolioOpt' fehlt. Bitte pip install PyPortfolioOpt ausführen.")
    st.stop()

st.set_page_config(page_title="Trumpfolio Dashboard", layout="wide")
st.title("📈 Trumpfolio Theory - Interactive Optimizer")

# --- Daten-Konfiguration ---
DATA_DIR = "data"
FILES_CONFIG = {
    "returns": {"filename": "daily_returns_full.xlsx", "index_col": 0}, 
    "desc":    {"filename": "Datensatz_Workshop_neu.xlsx", "index_col": None}, 
    "market":  {"filename": "descriptives_full_02.xlsx", "index_col": 1}
}

# --- Caching & Loading ---
@st.cache_data(show_spinner=False)
def load_data_file(key, uploaded_file=None):
    conf = FILES_CONFIG[key]
    path = os.path.join(DATA_DIR, conf["filename"])
    
    df = None
    if uploaded_file:
        df = pd.read_excel(uploaded_file, index_col=conf["index_col"])
    elif os.path.exists(path):
        df = pd.read_excel(path, index_col=conf["index_col"])
    
    if df is not None:
        if df.index.dtype == 'object':
            df.index = df.index.str.strip()
        if key == "returns": 
            df.columns = df.columns.str.strip()
            df.index = pd.to_datetime(df.index)
            
    return df

# --- Sidebar: Daten laden ---
st.sidebar.header("1. Datenquellen")
dfs = {}

for key, conf in FILES_CONFIG.items():
    local_path = os.path.join(DATA_DIR, conf["filename"])
    if os.path.exists(local_path):
        st.sidebar.success(f"Gefunden: {conf['filename']}")
        dfs[key] = load_data_file(key)
    else:
        up = st.sidebar.file_uploader(f"Upload {conf['filename']}", type=["xlsx"])
        if up:
            dfs[key] = load_data_file(key, up)
        else:
            dfs[key] = None

st.sidebar.markdown("---")
st.sidebar.header("2. Globale Parameter")
risk_free_rate = st.sidebar.number_input("Risk Free Rate (Rf)", value=0.02, step=0.005, format="%.3f")
threshold_es = st.sidebar.slider("Emissions-Score Threshold (Quantil)", 0.1, 0.9, 0.3, 0.05)


# --- NEU: Sidebar Constraints ---
st.sidebar.markdown("---")
st.sidebar.header("3. Portfolio Beschränkungen")

# A. Max Weight per Asset
max_weight_asset = st.sidebar.slider(
    "Max. Allokation pro Aktie", 
    min_value=0.05, max_value=1.0, value=1.0, step=0.05,
    help="Wie viel Prozent darf eine einzelne Aktie maximal einnehmen?"
)

# B. Regionen Constraints (Dynamisch)
region_constraints = {}
available_regions = []

if dfs["desc"] is not None:
    # Wir holen uns die Regionen aus dem Datensatz, um die Slider zu bauen
    # Achtung: Wir nehmen hier alle Regionen, auch wenn sie nachher evtl. rausgefiltert werden
    available_regions = dfs["desc"]['RegionofHeadquarters'].dropna().unique()
    available_regions = sorted([str(x) for x in available_regions])

    with st.sidebar.expander("🌍 Regionale Limits (Min/Max)", expanded=False):
        st.caption("Lege fest, wie viel % des Portfolios aus einer Region kommen muss/darf.")
        for reg in available_regions:
            # Slider liefert Tuple (min, max)
            # Default: (0.0, 1.0) also keine Einschränkung
            c_min, c_max = st.slider(f"{reg}", 0.0, 1.0, (0.0, 1.0), step=0.05)
            
            # Wir speichern nur Beschränkungen, die vom Standard (0,1) abweichen
            if c_min > 0.0 or c_max < 1.0:
                region_constraints[reg] = (c_min, c_max)

# --- Logik Funktionen ---

def get_filtered_universe(df_desc, thresh_quantile):
    """Logik analog Notebook V2."""
    df_clean = df_desc.copy()
    df_clean = df_clean.dropna(subset=['EmissionsScore', 'SinRevenue'])
    df_clean['SinRevenue'] = pd.to_numeric(df_clean['SinRevenue'], errors='coerce')
    
    df_sin = df_clean[df_clean['SinRevenue'] == 1]
    thresh_val = df_sin['EmissionsScore'].quantile(thresh_quantile)
    df_final = df_sin[df_sin['EmissionsScore'] <= thresh_val]
    
    valid_isins = df_final['isin'].unique()
    valid_isins = [str(x).strip() for x in valid_isins] 
    return valid_isins

def optimize_portfolio(df_ret, df_mkt, df_desc, valid_isins, rf, max_single_weight, region_limits):
    """
    Erweitert um Constraints und Mapping für Regionen.
    """
    # 1. Datenvorbereitung (Notebook Logik)
    common_index = df_ret.columns.intersection(df_mkt.index)
    if len(common_index) < 10:
        return None, "Zu wenig Übereinstimmungen Datasets.", None

    mkt_returns_sync = df_ret[common_index]
    mkt_caps = df_mkt.loc[common_index, 'Mcap']
    w_mkt = mkt_caps / mkt_caps.sum()
    
    weighted_returns = mkt_returns_sync.mul(w_mkt, axis=1)
    market_return_series = weighted_returns.sum(axis=1)
    
    expected_market_return = market_return_series.mean() * 252
    market_variance = market_return_series.var() * 252
    delta = (expected_market_return - rf) / market_variance
    
    cov_matrix_market = mkt_returns_sync.cov() * 252
    
    prior_returns = black_litterman.market_implied_prior_returns(
        market_caps=w_mkt,
        risk_aversion=delta,
        cov_matrix=cov_matrix_market,
        risk_free_rate=rf
    )
    
    # 2. Filter Sin-Universe
    available_isins = [isin for isin in valid_isins if isin in df_ret.columns]
    final_universe = [isin for isin in available_isins if isin in prior_returns.index]
    
    if len(final_universe) == 0:
        return None, "Keine Aktien im Universum übrig.", None

    er_filtered = prior_returns.loc[final_universe]
    returns_filtered = df_ret[final_universe]
    
    # Cleaning
    returns_safe = returns_filtered.astype('float64').clip(lower=-0.9, upper=5.0).fillna(0.0)
    common_idx = er_filtered.index.intersection(returns_safe.columns)
    er_final = er_filtered.loc[common_idx]
    returns_final = returns_safe[common_idx]
    
    # Kovarianz
    try:
        cov_matrix_final = risk_models.CovarianceShrinkage(returns_final).ledoit_wolf()
    except:
        cov_matrix_temp = returns_final.cov() * 252
        cov_matrix_final = risk_models.fix_nonpositive_semidefinite(cov_matrix_temp)
        
    # --- 3. Optimierung mit Constraints ---
    
    # Bounds: (0, max_single_weight) für jede Aktie
    # Standard ist (0, 1)
    ef = EfficientFrontier(er_final, cov_matrix_final, weight_bounds=(0, max_single_weight))
    
    # Regionen Constraints anwenden
# Regionen Constraints anwenden
    if region_limits:
        # 1. Mapping erstellen {ISIN: Region}
        subset_desc = df_desc[df_desc['isin'].isin(common_idx)]
        isin_region_map = subset_desc.drop_duplicates('isin').set_index('isin')['RegionofHeadquarters'].to_dict()
        
        # Mapper für PyPortfolioOpt (Fallback "Other" falls Daten fehlen)
        sector_mapper = {isin: isin_region_map.get(isin, "Other") for isin in common_idx}
        
        # 2. Limits aufsplitten in Lower und Upper Bounds
        # PyPortfolioOpt braucht zwei getrennte Dicts: eins für Min, eins für Max
        sector_lower = {reg: limits[0] for reg, limits in region_limits.items()}
        sector_upper = {reg: limits[1] for reg, limits in region_limits.items()}
        
        # 3. Constraints hinzufügen
        # Hier übergeben wir jetzt explizit (Mapper, Lower, Upper)
        ef.add_sector_constraints(sector_mapper, sector_lower, sector_upper)

    try:
        # Hier lag evtl. der Unterschied: Notebook nutzt ef.max_sharpe() ohne args -> nutzt default rf (0.02)
        weights = ef.max_sharpe(risk_free_rate=rf)
        cleaned_weights = ef.clean_weights()
        
        # Performance holen für Ausgabe im Dashboard
        perf = ef.portfolio_performance(verbose=False, risk_free_rate=rf)
        
        return cleaned_weights, perf, sector_mapper if region_limits else None
        
    except Exception as e:
        return None, f"Optimierung gescheitert. Sind die Constraints möglich?\nFehler: {str(e)}", None


# --- Main UI ---

if all(df is not None for df in dfs.values()):
    if st.button("🚀 Kalkulation starten"):
        with st.spinner("Berechne Portfolio..."):
            
            # 1. Filterung durchführen
            valid_isins = get_filtered_universe(dfs["desc"], threshold_es)
            
            # Safety Check: Apple
            if "US0378331005" in valid_isins:
                st.error("🚨 ALARM: Apple ist im Filter enthalten! Die Filterlogik muss geprüft werden.")
            else:
                # 2. Optimierung starten
                res, perf, mapper = optimize_portfolio(
                    dfs["returns"], 
                    dfs["market"], 
                    dfs["desc"], # Übergeben für Regionen-Mapping
                    valid_isins, 
                    risk_free_rate,
                    max_weight_asset,
                    region_constraints
                )
                
                if res is None:
                    st.error(perf) # Fehlermeldung aus der Funktion anzeigen
                else:
                    # --- DASHBOARD AUFBAU ---
                    st.success("Optimierung erfolgreich!")
                    
                    # Umwandeln in DataFrame für die Anzeige
                    df_w = pd.DataFrame.from_dict(res, orient='index', columns=['Weight'])
                    # Nur echte Positionen behalten
                    df_active = df_w[df_w['Weight'] > 0.0001].copy()
                    
                    # ---------------------------------------------------------
                    # 🧬 4-STUFEN FUNNEL VISUALISIERUNG (KORRIGIERT)
                    # ---------------------------------------------------------
                    st.markdown("### 🧬 Der Selektions-Trichter (Funnel)")
                    
                    # 1. Daten für die 4 Stufen berechnen
                    
                    # Stufe 1: Gesamtes Universum (aus Market File)
                    # KORREKTUR: Zugriff auf dfs["market"] statt dfs["desc"]
                    # Wir nehmen die Länge des Index (Annahme: 1 Zeile pro Aktie im Market File)
                    count_total = len(dfs["market"])
                    
                    # Stufe 2: Nach Sin & ES Filter (Deine valid_isins aus dem Workshop File)
                    count_filtered = len(valid_isins)
                    
                    # Stufe 3: Verfügbare Rendite-Daten (Match mit Returns File)
                    isins_with_data = [i for i in valid_isins if i in dfs["returns"].columns]
                    count_data = len(isins_with_data)
                    
                    # Stufe 4: Finales Portfolio (Gewichtung > 0)
                    count_final = len(df_active)

                    # --- CRITICAL CHECK (Sparring Partner) ---
                    # Falls der Filter-Datensatz mehr Aktien hat als der Markt-Datensatz, stimmt was nicht.
                    if count_filtered > count_total:
                         st.warning(f"⚠️ **Daten-Inkonsistenz:** Dein gefiltertes Universum ({count_filtered}) ist größer als der Gesamtmarkt ({count_total}). Prüfe, ob 'market' und 'desc' zusammenpassen!")

                    # 2. DataFrame für Plotly
                    df_funnel = pd.DataFrame({
                        'Phase': [
                            '1. Gesamtmarkt', 
                            '2. Nach Nachhaltigkeits-Filter', 
                            '3. Mit Rendite-Daten (Quality Check)', 
                            '4. Finales Portfolio'
                        ],
                        'Anzahl': [count_total, count_filtered, count_data, count_final]
                    })

                    # 3. Plotly Chart erstellen
                    fig_funnel = px.funnel(
                        df_funnel, 
                        x='Anzahl', 
                        y='Phase', 
                        title='Selektionsprozess',
                        color='Phase',
                        # Farben: Grau (Markt), Gold (Filter), Blau (Daten), Grün (Portfolio)
                        color_discrete_sequence=['#E0E0E0', '#FFD700', '#87CEEB', '#228B22'] 
                    )
                    
                    fig_funnel.update_traces(textinfo="value+percent previous") 
                    fig_funnel.update_layout(showlegend=False, height=400)

                    # 4. Anzeige (Split Layout)
                    c_fun1, c_fun2 = st.columns([2, 1])
                    
                    with c_fun1:
                        st.plotly_chart(fig_funnel, use_container_width=True)
                    
                    with c_fun2:
                        st.info("💡 **Analyse:**")
                        
                        # Berechnung Drop-Off
                        if count_filtered > 0:
                            drop_stage3 = 1 - (count_data / count_filtered)
                        else:
                            drop_stage3 = 0
                        st.markdown(f"**Daten-Verfügbarkeitsverlust (Stage 2 → 3):** {drop_stage3:.4%}")
                            
                        # Selektionsquote (Markt -> Portfolio)
                        if count_total > 0:
                            selection_rate = count_final / count_total
                            st.markdown(f"**Gesamt-Selektionsquote:** {selection_rate:.4%}")
                        
                        st.caption("Zeigt, wie selektiv der Algorithmus arbeitet.")
                    
                    st.divider()
                    # ---------------------------------------------------------

                    # ---------------------------------------------------------
                    # DATEN-MERGE & VISUALISIERUNG
                    # ---------------------------------------------------------

                    # Merge mit Meta-Daten (Regionen & Sektoren) für Anzeige
                    # WICHTIG: Hier 'TRBCBusinessSectorName' explizit mit auswählen!
                    meta_cols = ['RegionofHeadquarters', 'TRBCBusinessSectorName', 'CountryofHeadquarters']
                    
                    # Wir holen Metadaten aus dem Descriptive File und entfernen Duplikate
                    meta_clean = dfs["desc"].drop_duplicates('isin').set_index('isin')[meta_cols]
                    
                    # Left Join: Portfolio-Daten mit Metadaten anreichern
                    df_display = df_active.join(meta_clean, how='left')
                    
                    # Fehlende Werte auffüllen (kosmetisch)
                    df_display['RegionofHeadquarters'] = df_display['RegionofHeadquarters'].fillna('Unknown')
                    df_display['TRBCBusinessSectorName'] = df_display['TRBCBusinessSectorName'].fillna('Other Sector')

                    # KPI Row Anzeige
                    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                    kpi1.metric("Anzahl Aktien", len(df_active))
                    kpi2.metric("Max Allokation", f"{df_active['Weight'].max():.2%}", df_active['Weight'].idxmax())
                    kpi3.metric("Min Allokation", f"{df_active['Weight'].min():.2%}", df_active['Weight'].idxmin())
                    kpi4.metric("Sharpe Ratio", f"{perf[2]:.2f}")
                    
                    st.markdown("---")
                    
                    # 2. Charts & Tables Area (Split Layout)
                    col_left, col_right = st.columns([1, 1])
                    
                    # --- LINKE SPALTE: TOP 10 ---
                    with col_left:
                        st.subheader("🏆 Top 10 Positionen")
                        top10 = df_display.sort_values('Weight', ascending=False).head(10)
                        
                        # Tabelle anzeigen (mit Sektor Info für mehr Details)
                        st.dataframe(
                            top10[['Weight', 'TRBCBusinessSectorName', 'CountryofHeadquarters']].style.format({'Weight': '{:.2%}'}),
                            use_container_width=True
                        )
                        
                        st.caption(f"Gesamtrendite (Exp.): {perf[0]:.2%} | Volatilität: {perf[1]:.2%}")

                    # --- RECHTE SPALTE: STRUKTUR (TABS) ---
                    with col_right:
                        st.subheader("📊 Portfolio Struktur")
                        
                        if not df_display.empty:
                            # Tabs für saubere Trennung
                            tab_reg, tab_sec = st.tabs(["🌍 Regionen", "🏭 Sektoren"])
                            
                            # --- TAB 1: REGIONEN ---
                            with tab_reg:
                                df_region = df_display.groupby('RegionofHeadquarters')['Weight'].sum().reset_index()
                                fig_reg = px.pie(
                                    df_region, 
                                    values='Weight', 
                                    names='RegionofHeadquarters', 
                                    hole=0.4,
                                    color_discrete_sequence=px.colors.qualitative.Pastel
                                )
                                fig_reg.update_layout(margin=dict(t=20, b=20, l=20, r=20))
                                st.plotly_chart(fig_reg, use_container_width=True)

                            # --- TAB 2: SEKTOREN ---
                            with tab_sec:
                                df_sector = df_display.groupby('TRBCBusinessSectorName')['Weight'].sum().reset_index()
                                # Sortieren, damit die größten Sektoren oben stehen (für die Legende)
                                df_sector = df_sector.sort_values('Weight', ascending=False)
                                
                                fig_sec = px.pie(
                                    df_sector, 
                                    values='Weight', 
                                    names='TRBCBusinessSectorName', 
                                    hole=0.4,
                                    color_discrete_sequence=px.colors.qualitative.Set3 # Andere Farbpalette zur Unterscheidung
                                )
                                fig_sec.update_layout(margin=dict(t=20, b=20, l=20, r=20))
                                st.plotly_chart(fig_sec, use_container_width=True)
                        else:
                            st.warning("Keine Positionen für Diagramme.")

                    # 3. Alle Positionen (Expander)
                    with st.expander("Vollständiges Portfolio ansehen"):
                        st.dataframe(df_display.sort_values('Weight', ascending=False).style.format({'Weight': '{:.4%}'}))
else:
    st.info("Bitte lade alle Dateien hoch oder platziere sie im 'data' Ordner (returns, desc, market).")