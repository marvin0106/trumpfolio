import streamlit as st
import pandas as pd
import numpy as np
import os

# --- Imports und Setup ---
try:
    from pypfopt import risk_models, EfficientFrontier, black_litterman
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

# --- Sidebar: Daten Laden ---
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
risk_free_rate = st.sidebar.number_input("Risk Free Rate", value=0.02, step=0.005, format="%.3f")
threshold_es = st.sidebar.slider("Emissions-Score Threshold (Quantil)", 0.1, 0.9, 0.3, 0.05)


# --- Kern-Logik (Filter & Optimierung) ---

def get_filtered_universe_and_regions(df_desc, thresh_quantile):
    """
    1. Filtert das Universum exakt wie im Notebook (Sin-Subset Quantil).
    2. Erstellt zusätzlich ein Mapping {ISIN: Region} für die Constraints.
    """
    df_clean = df_desc.copy()
    
    # Cleaning wie im Notebook Zelle 4
    df_clean = df_clean.dropna(subset=['EmissionsScore', 'SinRevenue', 'RegionofHeadquarters'])
    df_clean['SinRevenue'] = pd.to_numeric(df_clean['SinRevenue'], errors='coerce')
    
    # Filter: SinRevenue == 1
    df_sin = df_clean[df_clean['SinRevenue'] == 1]
    
    # Threshold auf dem SIN-Subset berechnen (Notebook Logik)
    thresh_val = df_sin['EmissionsScore'].quantile(thresh_quantile)
    
    # Finaler Cut
    df_final = df_sin[df_sin['EmissionsScore'] <= thresh_val]
    
    # ISIN Liste
    valid_isins = [str(x).strip() for x in df_final['isin'].unique()]
    
    # Regionen Mapping erstellen (ISIN -> Region)
    # Da Panel-Daten vorliegen, nehmen wir den letzten verfügbaren Eintrag pro ISIN
    df_unique_meta = df_final.drop_duplicates(subset=['isin'], keep='last')
    region_map = pd.Series(df_unique_meta.RegionofHeadquarters.values, index=df_unique_meta.isin).to_dict()
    
    # Liste aller verfügbaren Regionen für die Sidebar
    available_regions = sorted(list(set(region_map.values())))
    
    return valid_isins, region_map, available_regions

def optimize_portfolio(df_ret, df_mkt, valid_isins, rf, region_limits, region_mapper, max_single_weight):
    """
    Führt Black-Litterman und Mean-Variance Optimierung durch.
    Beachtet dabei regionale Constraints und max. Einzelgewichtung.
    """
    
    # --- 1. Vorbereitung Gesamtmarkt (BL) ---
    common_index = df_ret.columns.intersection(df_mkt.index)
    if len(common_index) < 10: return None, "Zu wenig Übereinstimmungen (Returns <-> Market)."

    mkt_returns_sync = df_ret[common_index]
    mkt_caps = df_mkt.loc[common_index, 'Mcap']
    w_mkt = mkt_caps / mkt_caps.sum()
    
    # Marktparameter berechnen
    weighted_returns = mkt_returns_sync.mul(w_mkt, axis=1)
    market_return_series = weighted_returns.sum(axis=1)
    
    expected_market_return = market_return_series.mean() * 252
    market_variance = market_return_series.var() * 252
    delta = (expected_market_return - rf) / market_variance
    
    cov_matrix_market = mkt_returns_sync.cov() * 252
    
    # BL Prior Returns
    prior_returns = black_litterman.market_implied_prior_returns(
        market_caps=w_mkt,
        risk_aversion=delta,
        cov_matrix=cov_matrix_market,
        risk_free_rate=rf
    )
    
    # --- 2. Reduktion auf Sin-Universum ---
    available_isins = [i for i in valid_isins if i in df_ret.columns]
    final_universe = [i for i in available_isins if i in prior_returns.index]
    
    if len(final_universe) == 0: return None, "Keine Aktien nach Schnittmenge übrig."

    er_filtered = prior_returns.loc[final_universe]
    returns_filtered = df_ret[final_universe]
    
    # Cleaning für Matrix
    returns_safe = returns_filtered.astype('float64').clip(lower=-0.9, upper=5.0).fillna(0.0)
    
    # Sync Indizes
    common_idx = er_filtered.index.intersection(returns_safe.columns)
    er_final = er_filtered.loc[common_idx]
    returns_final = returns_safe[common_idx]
    
    # Kovarianzmatrix
    try:
        cov_matrix_final = risk_models.CovarianceShrinkage(returns_final).ledoit_wolf()
    except:
        cov_matrix_final = risk_models.fix_nonpositive_semidefinite(returns_final.cov() * 252)
        
    # --- 3. Optimierung mit Constraints ---
    
    # Wir setzen weight_bounds=(0, max_single_weight) -> Keine Shortpositionen, Max Gewicht pro Aktie
    ef = EfficientFrontier(er_final, cov_matrix_final, weight_bounds=(0, max_single_weight))
    
    # Regionale Constraints hinzufügen
    # PyPortfolioOpt braucht einen Mapper, der nur die Assets im finalen Universum enthält
    # Wir filtern den globalen region_mapper auf die finalen ISINs
    final_region_mapper = {isin: region_mapper.get(isin, "Other") for isin in er_final.index}
    
    # Wir müssen prüfen, ob Limits verletzt werden könnten (z.B. wenn Limit < Summe der Min-Weights)
    # Hier nehmen wir Sector Constraints
    # Achtung: Wenn ein Limit auf 0 gesetzt wird, fliegt der Sektor raus.
    
    # Erstelle das Limits-Dictionary für PyPortfolioOpt
    # Format: {'Asia': 0.5, 'Europe': 0.3}
    active_sector_limits = {reg: lim for reg, lim in region_limits.items() if reg in list(final_region_mapper.values())}
    
    # Constraints anwenden
    if active_sector_limits:
        ef.add_sector_constraints(final_region_mapper, active_sector_limits)
        
    try:
        weights = ef.max_sharpe(risk_free_rate=rf)
        cleaned_weights = ef.clean_weights()
        perf = ef.portfolio_performance(verbose=False, risk_free_rate=rf)
        return cleaned_weights, perf
    except Exception as e:
        return None, f"Optimierung fehlgeschlagen (Constraints zu strikt?): {str(e)}"


# --- UI Logik ---

if all(df is not None for df in dfs.values()):
    
    # 1. Daten vorverarbeiten um Regionen zu erhalten
    valid_isins, region_map, available_regions = get_filtered_universe_and_regions(dfs["desc"], threshold_es)
    
    # --- Sidebar: Constraints ---
    st.sidebar.markdown("---")
    st.sidebar.header("3. Constraints (Regeln)")
    
    # A. Einzelaktien Limit
    max_single_asset = st.sidebar.slider(
        "Max. Gewichtung pro Aktie", 
        min_value=0.05, max_value=1.0, value=1.0, step=0.05,
        help="Keine einzelne Aktie darf mehr als diesen Prozentsatz ausmachen."
    )
    
    # B. Regionale Limits
    st.sidebar.subheader("Max. Allokation nach Region")
    region_limits = {}
    
    # Expander für Ordnung
    with st.sidebar.expander("Regionale Limits anpassen", expanded=True):
        for region in available_regions:
            # Default 1.0 = 100% erlaubt (keine Einschränkung)
            val = st.slider(f"{region}", 0.0, 1.0, 1.0, 0.05)
            region_limits[region] = val

    # --- Hauptbereich ---

    if st.button("🚀 Kalkulation starten", type="primary"):
        with st.spinner("Optimiere Portfolio..."):
            
            # Apple Check
            if "US0378331005" in valid_isins:
                st.error("🚨 ALARM: Apple ist fälschlicherweise im Datensatz!")
            else:
                # Optimierung aufrufen
                res, perf = optimize_portfolio(
                    dfs["returns"], dfs["market"], valid_isins, risk_free_rate, 
                    region_limits, region_map, max_single_asset
                )
                
                if res is None:
                    st.error(perf) # Fehlermeldung anzeigen
                else:
                    # --- Ergebnisse Anzeigen (UX) ---
                    
                    # 1. Performance KPIs
                    st.markdown("### 📊 Portfolio Performance")
                    kpi1, kpi2, kpi3 = st.columns(3)
                    kpi1.metric("Erwartete Rendite (p.a.)", f"{perf[0]:.2%}")
                    kpi2.metric("Volatilität (p.a.)", f"{perf[1]:.2%}")
                    kpi3.metric("Sharpe Ratio", f"{perf[2]:.2f}")
                    
                    st.divider()
                    
                    # Datenaufbereitung für Tabelle & Stats
                    df_w = pd.DataFrame.from_dict(res, orient='index', columns=['Weight'])
                    # Filter auf relevante Positionen
                    portfolio_active = df_w[df_w['Weight'] > 0.0001].copy()
                    portfolio_active = portfolio_active.sort_values('Weight', ascending=False)
                    
                    # Region hinzufügen zur Tabelle
                    portfolio_active['Region'] = portfolio_active.index.map(region_map)
                    
                    # 2. Portfolio Statistiken (User Wunsch)
                    st.markdown("### 🧩 Portfolio Zusammensetzung")
                    
                    stat1, stat2, stat3, stat4 = st.columns(4)
                    
                    # Anzahl
                    count = len(portfolio_active)
                    stat1.metric("Anzahl Aktien", count)
                    
                    # Max Allokation
                    if not portfolio_active.empty:
                        max_stock = portfolio_active['Weight'].idxmax()
                        max_val = portfolio_active['Weight'].max()
                        stat2.metric("Größte Position", f"{max_val:.2%}", help=max_stock)
                        
                        min_stock = portfolio_active['Weight'].idxmin()
                        min_val = portfolio_active['Weight'].min()
                        stat3.metric("Kleinste Position", f"{min_val:.2%}", help=min_stock)
                        
                        # Summe (Check)
                        stat4.metric("Investitionsquote", f"{portfolio_active['Weight'].sum():.1%}")
                    
                    # 3. Top 10 Tabelle
                    st.subheader("🏆 Top 10 Positionen")
                    
                    top10 = portfolio_active.head(10)
                    
                    # Styling der Tabelle
                    st.dataframe(
                        top10.style.format({'Weight': '{:.2%}'}).background_gradient(subset=['Weight'], cmap="Greens"),
                        use_container_width=True
                    )
                    
                    # 4. Volle Liste im Expander
                    with st.expander("Vollständige Positionsliste anzeigen"):
                        st.dataframe(portfolio_active.style.format({'Weight': '{:.4f}'}), use_container_width=True)

else:
    st.info("👋 Willkommen! Bitte lade links deine Excel-Dateien hoch oder lege sie in den 'data' Ordner.")