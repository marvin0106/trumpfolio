import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px

# --- Imports und Setup ---
try:
    from pypfopt import risk_models, EfficientFrontier, black_litterman
except ImportError:
    st.error("Fehler: 'PyPortfolioOpt' fehlt. Bitte pip install PyPortfolioOpt ausführen.")
    st.stop()

st.set_page_config(page_title="Trumpfolio Dashboard", layout="wide", page_icon="⚖️")
st.title("⚖️ Trumpfolio Theory - Comparative Optimizer")

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
    if uploaded_file:
        df = pd.read_excel(uploaded_file, index_col=FILES_CONFIG[key]["index_col"])
    else:
        path = os.path.join(DATA_DIR, FILES_CONFIG[key]["filename"])
        if os.path.exists(path):
            df = pd.read_excel(path, index_col=FILES_CONFIG[key]["index_col"])
        else:
            return None
    
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

# --- Sidebar: Parameter ---
st.sidebar.markdown("---")
st.sidebar.header("2. Globale Parameter")

risk_free_rate = st.sidebar.number_input("Risk Free Rate (Rf)", value=0.02, step=0.005, format="%.3f")

# Slider gilt für beide Richtungen symmetrisch
selection_pct = st.sidebar.slider(
    "Selektions-Intensität (Top/Bottom %)", 
    min_value=0.1, max_value=0.9, value=0.3, step=0.05,
    help="Definiert die Schärfe des Filters. 0.30 bedeutet: Die schlechtesten 30% (Brown) vs. die besten 30% (Green)."
)

st.sidebar.markdown("---")
st.sidebar.header("3. Constraints")
max_weight_asset = st.sidebar.slider("Max. Allokation pro Aktie", 0.05, 1.0, 1.0, 0.05)

region_constraints = {}
available_regions = []
if dfs["desc"] is not None:
    available_regions = sorted([str(x) for x in dfs["desc"]['RegionofHeadquarters'].dropna().unique()])
    with st.sidebar.expander("🌍 Regionale Limits", expanded=False):
        for reg in available_regions:
            c_min, c_max = st.slider(f"{reg}", 0.0, 1.0, (0.0, 1.0), step=0.05)
            if c_min > 0.0 or c_max < 1.0:
                region_constraints[reg] = (c_min, c_max)

# --- Logik Funktionen ---

def get_universe_by_strategy(df_desc, selection_percentage, strategy="green"):
    """
    Filtert das Universum für eine spezifische Strategie.
    strategy: 'green' oder 'brown'
    """
    df_clean = df_desc.copy()
    df_clean = df_clean.dropna(subset=['EmissionsScore', 'SinRevenue'])
    df_clean['SinRevenue'] = pd.to_numeric(df_clean['SinRevenue'], errors='coerce')
    
    if strategy == "brown":
        # Brown: Nur Sin-Stocks (1) UND schlechte Scores (Bottom Quantile)
        df_subset = df_clean[df_clean['SinRevenue'] == 1]
        threshold_val = df_subset['EmissionsScore'].quantile(selection_percentage)
        df_final = df_subset[df_subset['EmissionsScore'] <= threshold_val]
        operator_label = "<="
        
    else: # green
        # Green: Keine Sin-Stocks (0) UND gute Scores (Top Quantile)
        df_subset = df_clean[df_clean['SinRevenue'] == 0]
        # Wir invertieren das Quantil (z.B. 30% Intensity -> Top 30% -> ab 70% Quantil)
        target_quantile = 1.0 - selection_percentage
        threshold_val = df_subset['EmissionsScore'].quantile(target_quantile)
        df_final = df_subset[df_subset['EmissionsScore'] >= threshold_val]
        operator_label = ">="
    
    valid_isins = [str(x).strip() for x in df_final['isin'].unique()]
    return valid_isins, threshold_val, operator_label

def run_optimization(df_ret, df_mkt, df_desc, valid_isins, rf, max_weight, reg_limits):
    """
    Führt die Optimierung für eine gegebene Liste an ISINs durch.
    """
    # 1. Markt-Prior (Black-Litterman Basis)
    common_index = df_ret.columns.intersection(df_mkt.index)
    if len(common_index) < 10:
        return None, "Zu wenig Markt-Daten matches."

    mkt_returns_sync = df_ret[common_index]
    mkt_caps = df_mkt.loc[common_index, 'Mcap']
    w_mkt = mkt_caps / mkt_caps.sum()
    
    # Implizite Rendite
    market_return_series = mkt_returns_sync.mul(w_mkt, axis=1).sum(axis=1)
    delta = (market_return_series.mean()*252 - rf) / (market_return_series.var()*252)
    cov_market = mkt_returns_sync.cov() * 252
    
    prior_returns = black_litterman.market_implied_prior_returns(
        market_caps=w_mkt, risk_aversion=delta, cov_matrix=cov_market, risk_free_rate=rf
    )
    
    # 2. Filtern auf Strategie-Universum
    available_isins = [i for i in valid_isins if i in df_ret.columns]
    final_universe = [i for i in available_isins if i in prior_returns.index]
    
    if not final_universe:
        return None, "Leeres Universum nach Data-Matching."

    # 3. Inputs vorbereiten
    er_final = prior_returns.loc[final_universe]
    returns_final = df_ret[final_universe].astype('float64').clip(-0.9, 5.0).fillna(0.0)
    
    # Index Sync
    idx = er_final.index.intersection(returns_final.columns)
    er_final = er_final.loc[idx]
    returns_final = returns_final[idx]
    
    # 4. Kovarianz (Ledoit Wolf mit Fallback)
    try:
        cov_matrix = risk_models.CovarianceShrinkage(returns_final).ledoit_wolf()
    except:
        cov_matrix = risk_models.fix_nonpositive_semidefinite(returns_final.cov() * 252)
        
    # 5. Efficient Frontier
    ef = EfficientFrontier(er_final, cov_matrix, weight_bounds=(0, max_weight))
    
    # Constraints
    if reg_limits:
        subset = df_desc[df_desc['isin'].isin(idx)].drop_duplicates('isin').set_index('isin')
        reg_map = subset['RegionofHeadquarters'].to_dict()
        mapper = {i: reg_map.get(i, "Other") for i in idx}
        ef.add_sector_constraints(mapper, 
                                  {r: l[0] for r, l in reg_limits.items()}, 
                                  {r: l[1] for r, l in reg_limits.items()})

    try:
        weights = ef.max_sharpe(risk_free_rate=rf)
        cleaned = ef.clean_weights()
        perf = ef.portfolio_performance(verbose=False, risk_free_rate=rf)
        return cleaned, perf
    except Exception as e:
        return None, str(e)

def display_portfolio_column(title, emoji, strategy_color, weights, perf, df_desc, df_mkt):
    """
    Hilfsfunktion, um eine Portfolio-Spalte (Brown oder Green) anzuzeigen.
    """
    st.markdown(f"### {emoji} {title}")
    
    if weights is None:
        st.error(f"Optimierung fehlgeschlagen: {perf}")
        return
    
    # 1. Daten aufbereiten
    df_w = pd.DataFrame.from_dict(weights, orient='index', columns=['Weight'])
    df_active = df_w[df_w['Weight'] > 0.0001].copy()
    
    # Metadaten holen (Sektor & Region)
    meta = df_desc.drop_duplicates('isin').set_index('isin')[['RegionofHeadquarters', 'TRBCBusinessSectorName']]
    df_show = df_active.join(meta, how='left')

    # --- FIX: Namen Mapping sicher machen ---
    # Schritt 1: Erst mappen (ergibt Index-Objekt)
    mapped_names = df_show.index.map(df_mkt['Name'])
    # Schritt 2: Zuweisen (Pandas macht daraus eine Series)
    df_show['Name'] = mapped_names
    # Schritt 3: fillna (jetzt sicher, da Series auf Series trifft)
    df_show['Name'] = df_show['Name'].fillna(df_show.index.to_series())
    
    # Kosmetik für NaN Sektoren
    df_show['RegionofHeadquarters'] = df_show['RegionofHeadquarters'].fillna('Unknown')
    df_show['TRBCBusinessSectorName'] = df_show['TRBCBusinessSectorName'].fillna('Other Sector')
    
    # 2. KPIs
    c1, c2 = st.columns(2)
    c1.metric("Rendite (Erw.)", f"{perf[0]:.2%}")
    c2.metric("Volatilität", f"{perf[1]:.2%}")
    st.metric("Sharpe Ratio", f"{perf[2]:.2f}")
    st.caption(f"Aktien im Portfolio: {len(df_active)}")
    
    st.divider()
    
    # 3. Top 5 Positionen (Kompakt)
    st.markdown("**Top 5 Positionen**")
    top5 = df_show.sort_values('Weight', ascending=False).head(5)
    
    st.dataframe(
        top5[['Name', 'Weight']],
        column_config={
            "Name": st.column_config.TextColumn("Name", width="medium"),
            "Weight": st.column_config.NumberColumn("Gewicht", format="%.2f %%")
        },
        use_container_width=True,
        hide_index=True
    )
    
    # 4. Sektor Chart
    st.markdown("**Sektoren**")
    grp = df_show.groupby('TRBCBusinessSectorName')['Weight'].sum().reset_index()
    
    # Farbschema: Rot/Blau für Brown, Grün-Töne für Green
    colors = px.colors.sequential.RdBu if strategy_color=='brown' else px.colors.sequential.Greens_r
    
    fig = px.pie(grp, values='Weight', names='TRBCBusinessSectorName', hole=0.5, 
                 color_discrete_sequence=colors)
    fig.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0), height=200)
    st.plotly_chart(fig, use_container_width=True)

    # 5. --- WIEDER DA: Vollständige Liste ---
    with st.expander("📄 Alle Positionen anzeigen", expanded=False):
        # Wir sortieren nach Gewichtung und zeigen relevante Spalten
        cols_full = ['Name', 'Weight', 'RegionofHeadquarters', 'TRBCBusinessSectorName']
        
        st.dataframe(
            df_show[cols_full].sort_values('Weight', ascending=False),
            column_config={
                "Name": st.column_config.TextColumn("Name"),
                "Weight": st.column_config.NumberColumn("Gewicht", format="%.4f %%"), # 4 Nachkommastellen für Details
                "RegionofHeadquarters": "Region",
                "TRBCBusinessSectorName": "Sektor"
            },
            use_container_width=True
        )

    return perf # Rückgabe der Performance für den Vergleich

# --- MAIN APP UI ---

if all(df is not None for df in dfs.values()):
    
    # Großer "Berechnen" Button über die volle Breite
    if st.button("⚖️ Portfolios Berechnen & Vergleichen", type="primary", use_container_width=True):
        
        # --- 1. Filterung (Universen bilden) ---
        isins_brown, thresh_brown, op_brown = get_universe_by_strategy(dfs["desc"], selection_pct, "brown")
        isins_green, thresh_green, op_green = get_universe_by_strategy(dfs["desc"], selection_pct, "green")
        
        # --- 2. Berechnung (Optimierung) ---
        with st.spinner("Optimiere Brown & Green Portfolios parallel..."):
            # Brown Optimierung
            res_brown, perf_brown = run_optimization(
                dfs["returns"], dfs["market"], dfs["desc"], 
                isins_brown, risk_free_rate, max_weight_asset, region_constraints
            )
            
            # Green Optimierung
            res_green, perf_green = run_optimization(
                dfs["returns"], dfs["market"], dfs["desc"], 
                isins_green, risk_free_rate, max_weight_asset, region_constraints
            )

        # --- 3. Anzeige (Side-by-Side) ---
        
        # Info Box über Filter-Logik
        with st.expander("ℹ️ Details zur Filter-Logik ansehen", expanded=False):
            c_info1, c_info2 = st.columns(2)
            c_info1.info(f"**Brown Filter:** Sin=Yes & Score {op_brown} {thresh_brown:.1f}\n\n(Universum: {len(isins_brown)} Aktien)")
            c_info2.success(f"**Green Filter:** Sin=No & Score {op_green} {thresh_green:.1f}\n\n(Universum: {len(isins_green)} Aktien)")

        # Haupt-Spalten
        col_brown, col_sep, col_green = st.columns([1, 0.1, 1])
        
        with col_brown:
            p_brown = display_portfolio_column(
                "Maximum Brown", "🟤", "brown", 
                res_brown, perf_brown, dfs["desc"], dfs["market"]
            )

        with col_green:
            p_green = display_portfolio_column(
                "Maximum Green", "🟢", "green", 
                res_green, perf_green, dfs["desc"], dfs["market"]
            )
            
        # --- 4. Vergleich / Delta ---
        if p_brown and p_green:
            st.markdown("---")
            st.subheader("📊 Performance Vergleich (Delta)")
            
            d_col1, d_col2, d_col3 = st.columns(3)
            
            # Delta Berechnung (Green - Brown)
            delta_ret = p_green[0] - p_brown[0]
            delta_vol = p_green[1] - p_brown[1]
            delta_sharpe = p_green[2] - p_brown[2]
            
            d_col1.metric("Rendite Spread (Green - Brown)", f"{delta_ret:+.2%}", delta_color="normal")
            d_col2.metric("Vola Spread (Green - Brown)", f"{delta_vol:+.2%}", delta_color="inverse")
            d_col3.metric("Sharpe Spread", f"{delta_sharpe:+.2f}", delta_color="normal")
            
            if delta_sharpe > 0:
                st.success("✅ **Fazit:** Das grüne Portfolio liefert eine bessere risikoadjustierte Rendite.")
            else:
                st.warning("⚠️ **Fazit:** Das braune Portfolio liefert aktuell eine bessere risikoadjustierte Rendite.")

else:
    st.info("Bitte lade die Excel-Dateien hoch.")