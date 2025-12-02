"""
Portfolio Optimizer Pro - Streamlit Version
Busca dados reais via yfinance e BCB, calcula volatilidades e correlações automaticamente.
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
from typing import Tuple, Dict, List, Optional
import matplotlib.pyplot as plt
import datetime as _dt

st.set_page_config(
    page_title="Portfolio Optimizer Pro",
    page_icon="📈",
    layout="wide"
)

# =====================
# Optional libs check
# =====================
try:
    from sklearn.covariance import LedoitWolf
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

try:
    import cvxpy as cp
    CVXPY_OK = True
except ImportError:
    CVXPY_OK = False

try:
    import yfinance as yf
    YF_OK = True
except ImportError:
    YF_OK = False

try:
    import requests as _rq
    REQUESTS_OK = True
except ImportError:
    REQUESTS_OK = False

# =====================
# Styling
# =====================
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #334155;
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 14px;
        color: #94a3b8;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)

# =====================
# Header
# =====================
st.title("📈 Portfolio Optimizer Pro")
st.caption("Otimização de portfólios com dados reais de mercado (BCB + yfinance) | Fronteira Eficiente | Markowitz")

# =====================
# Data Fetching Functions
# =====================
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_yf_prices(ticker: str, start: str, end: Optional[str] = None) -> Optional[pd.Series]:
    """Busca preços históricos via yfinance"""
    if not YF_OK:
        st.error("yfinance não instalado. Adicione ao requirements.txt")
        return None
    try:
        if end is None:
            end = _dt.date.today().isoformat()
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if df is None or len(df) == 0:
            return None
        
        # Handle MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            if 'Close' in df.columns.get_level_values(0):
                s = df['Close']
            elif 'Adj Close' in df.columns.get_level_values(0):
                s = df['Adj Close']
            else:
                return None
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
        else:
            if 'Close' in df.columns:
                s = df['Close']
            elif 'Adj Close' in df.columns:
                s = df['Adj Close']
            else:
                return None
        
        s = pd.to_numeric(s, errors='coerce').dropna()
        s.index = pd.to_datetime(s.index, errors='coerce')
        s = s[~s.index.isna()]
        s.name = ticker
        return s
    except Exception as e:
        st.warning(f"Erro ao buscar {ticker}: {e}")
        return None

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_bcb_series(code: int, start: str) -> Optional[pd.Series]:
    """Busca séries do Banco Central (SGS)"""
    if not REQUESTS_OK:
        return None
    try:
        start_br = pd.to_datetime(start).strftime("%d/%m/%Y")
        end_br = _dt.date.today().strftime("%d/%m/%Y")
        base = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados"
        url = f"{base}?formato=json&dataInicial={start_br}&dataFinal={end_br}"
        
        r = _rq.get(url, timeout=30)
        if not r.ok:
            # Fallback: últimos 5000 pontos
            url2 = f"{base}/ultimos/5000?formato=json"
            r = _rq.get(url2, timeout=30)
            r.raise_for_status()
        
        js = r.json()
        df = pd.DataFrame(js)
        df['data'] = pd.to_datetime(df['data'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['data']).set_index('data')
        df['valor'] = pd.to_numeric(df['valor'], errors='coerce')
        s = df['valor'].dropna()
        s.name = f"BCB_{code}"
        return s
    except Exception as e:
        st.warning(f"Erro ao buscar BCB {code}: {e}")
        return None

def rate_to_price_index(rate_pct: pd.Series) -> pd.Series:
    """Converte taxa anual % para índice de preços"""
    if rate_pct.empty:
        return rate_pct
    r_daily = rate_pct / 100.0 / 252.0
    idx = 100.0 * (1.0 + r_daily).cumprod()
    idx.name = rate_pct.name
    return idx

# =====================
# Default Asset Mapping
# =====================
DEFAULT_MAPPING = pd.DataFrame({
    "Classe": [
        "RF (CDI/Selic)",
        "Inflação (IMA-B 5)",
        "Pré-fixado (IRF-M)",
        "Crédito Privado",
        "Renda Variável BR",
        "Renda Variável EXT",
        "Multimercado"
    ],
    "Fonte": ["bcb", "yfinance", "yfinance", "yfinance", "yfinance", "yfinance", "yfinance"],
    "Identificador": ["4389", "IMAB11.SA", "IRFM11.SA", "CPTI11.SA", "BOVA11.SA", "IVVB11.SA", "DIVO11.SA"],
    "Retorno Esperado (%)": [11.75, 10.5, 11.0, 12.5, 14.0, 12.0, 10.0],
    "Min (%)": [0, 0, 0, 0, 0, 0, 0],
    "Max (%)": [100, 30, 30, 20, 40, 20, 20]
})

# =====================
# Calculation Functions
# =====================
def calc_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calcula retornos logarítmicos"""
    rets = np.log(prices / prices.shift(1)).dropna()
    return rets.replace([np.inf, -np.inf], np.nan).dropna()

def calc_annualized_stats(returns: pd.DataFrame, periods_per_year: int = 252) -> Tuple[pd.Series, pd.DataFrame]:
    """Calcula média e covariância anualizadas"""
    mu = returns.mean() * periods_per_year
    sigma = returns.cov() * periods_per_year
    return mu, sigma

def ledoit_wolf_cov(returns: pd.DataFrame) -> pd.DataFrame:
    """Covariância com shrinkage Ledoit-Wolf"""
    if not SKLEARN_OK:
        return returns.cov()
    lw = LedoitWolf().fit(returns.values)
    return pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns)

def regularize_cov(Sigma: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
    """Regulariza matriz para ser positiva definida"""
    try:
        np.linalg.cholesky(Sigma.values)
        return Sigma
    except np.linalg.LinAlgError:
        A = Sigma.values.copy()
        boost = eps
        for _ in range(12):
            try:
                np.linalg.cholesky(A + boost * np.eye(A.shape[0]))
                return pd.DataFrame(A + boost * np.eye(A.shape[0]), 
                                   index=Sigma.index, columns=Sigma.columns)
            except np.linalg.LinAlgError:
                boost *= 10.0
        return pd.DataFrame(A + boost * np.eye(A.shape[0]), 
                           index=Sigma.index, columns=Sigma.columns)

def portfolio_metrics(w: np.ndarray, mu: np.ndarray, Sigma: np.ndarray, rf: float = 0) -> Dict:
    """Calcula métricas do portfólio"""
    ret = float(w @ mu)
    vol = float(np.sqrt(w @ Sigma @ w))
    sharpe = (ret - rf) / vol if vol > 0 else 0
    return {"return": ret, "volatility": vol, "sharpe": sharpe}

# =====================
# Optimization Functions
# =====================
def optimize_min_variance(mu: np.ndarray, Sigma: np.ndarray, 
                          lb: np.ndarray, ub: np.ndarray) -> Optional[np.ndarray]:
    """Otimiza para mínima variância"""
    if not CVXPY_OK:
        st.error("cvxpy não instalado")
        return None
    
    n = len(mu)
    w = cp.Variable(n)
    
    constraints = [
        cp.sum(w) == 1,
        w >= lb,
        w <= ub
    ]
    
    objective = cp.Minimize(cp.quad_form(w, Sigma))
    prob = cp.Problem(objective, constraints)
    
    try:
        prob.solve(solver=cp.OSQP, verbose=False)
        if w.value is None:
            prob.solve(solver=cp.SCS, verbose=False)
    except Exception:
        return None
    
    return np.array(w.value).ravel() if w.value is not None else None

def optimize_max_sharpe(mu: np.ndarray, Sigma: np.ndarray, rf: float,
                        lb: np.ndarray, ub: np.ndarray) -> Optional[np.ndarray]:
    """Otimiza para máximo Sharpe via grid search"""
    if not CVXPY_OK:
        return None
    
    n = len(mu)
    mu_min, mu_max = float(np.min(mu)), float(np.max(mu))
    targets = np.linspace(mu_min, mu_max, 30)
    
    best_w, best_sharpe = None, -1e18
    
    for target in targets:
        w = cp.Variable(n)
        constraints = [
            cp.sum(w) == 1,
            w >= lb,
            w <= ub,
            mu @ w >= target
        ]
        
        objective = cp.Minimize(cp.quad_form(w, Sigma))
        prob = cp.Problem(objective, constraints)
        
        try:
            prob.solve(solver=cp.OSQP, verbose=False)
            if w.value is None:
                continue
        except Exception:
            continue
        
        wv = np.array(w.value).ravel()
        port = portfolio_metrics(wv, mu, Sigma, rf)
        
        if port["sharpe"] > best_sharpe:
            best_sharpe = port["sharpe"]
            best_w = wv
    
    return best_w

def compute_frontier(mu: np.ndarray, Sigma: np.ndarray,
                     lb: np.ndarray, ub: np.ndarray, n_points: int = 30) -> List[Dict]:
    """Calcula fronteira eficiente"""
    if not CVXPY_OK:
        return []
    
    n = len(mu)
    mu_min, mu_max = float(np.min(mu)), float(np.max(mu))
    targets = np.linspace(mu_min, mu_max, n_points)
    
    frontier = []
    for target in targets:
        w = cp.Variable(n)
        constraints = [
            cp.sum(w) == 1,
            w >= lb,
            w <= ub,
            mu @ w >= target
        ]
        
        objective = cp.Minimize(cp.quad_form(w, Sigma))
        prob = cp.Problem(objective, constraints)
        
        try:
            prob.solve(solver=cp.OSQP, verbose=False)
            if w.value is None:
                continue
        except Exception:
            continue
        
        wv = np.array(w.value).ravel()
        port = portfolio_metrics(wv, mu, Sigma)
        frontier.append({
            "weights": wv,
            "return": port["return"],
            "volatility": port["volatility"],
            "sharpe": port["sharpe"]
        })
    
    return frontier

# =====================
# Build Dataset Function
# =====================
def build_dataset(mapping: pd.DataFrame, start: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Monta dataset de preços a partir do mapeamento"""
    cols = []
    failures = []
    
    progress = st.progress(0, text="Buscando dados...")
    total = len(mapping)
    
    for i, (_, row) in enumerate(mapping.iterrows()):
        classe = str(row.get('Classe', '')).strip()
        fonte = str(row.get('Fonte', '')).strip().lower()
        ident = str(row.get('Identificador', '')).strip()
        
        progress.progress((i + 1) / total, text=f"Buscando {classe}...")
        
        if not classe or not fonte or not ident:
            continue
        
        try:
            if fonte == 'yfinance':
                s = fetch_yf_prices(ident, start=start)
                if s is None or s.empty:
                    failures.append((classe, fonte, ident, 'Sem dados'))
                else:
                    cols.append(s.rename(classe))
            
            elif fonte == 'bcb':
                try:
                    code = int(float(ident))
                except ValueError:
                    failures.append((classe, fonte, ident, 'Código BCB inválido'))
                    continue
                
                s = fetch_bcb_series(code, start=start)
                if s is None or s.empty:
                    failures.append((classe, fonte, ident, 'Sem dados BCB'))
                else:
                    s_idx = rate_to_price_index(s).rename(classe)
                    cols.append(s_idx)
            else:
                failures.append((classe, fonte, ident, 'Fonte desconhecida'))
        
        except Exception as e:
            failures.append((classe, fonte, ident, str(e)))
    
    progress.empty()
    
    if not cols:
        return pd.DataFrame(), pd.DataFrame(failures, columns=['Classe', 'Fonte', 'ID', 'Erro'])
    
    # Combinar séries
    panel = pd.concat(cols, axis=1).sort_index()
    panel = panel.ffill().dropna()
    
    return panel, pd.DataFrame(failures, columns=['Classe', 'Fonte', 'ID', 'Erro'])

# =====================
# Sidebar Configuration
# =====================
st.sidebar.header("⚙️ Configuração")

# Date range
st.sidebar.subheader("📅 Período de Análise")
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Início", value=_dt.date(2022, 1, 1))
with col2:
    end_date = st.date_input("Fim", value=_dt.date.today())

# Estimation window
window_days = st.sidebar.selectbox(
    "Janela de estimação",
    options=[252, 504, 756],
    format_func=lambda x: f"{x//252} ano(s) (~{x} dias)",
    index=1
)

# Risk-free rate
rf = st.sidebar.number_input(
    "Taxa livre de risco (% a.a.)",
    min_value=0.0, max_value=30.0, value=11.75, step=0.25
) / 100

# Covariance method
cov_method = st.sidebar.selectbox(
    "Método de covariância",
    ["Amostral", "Ledoit-Wolf (shrinkage)"],
    index=1
)

# =====================
# Asset Mapping Editor
# =====================
st.sidebar.subheader("📊 Mapeamento de Ativos")

if "asset_mapping" not in st.session_state:
    st.session_state["asset_mapping"] = DEFAULT_MAPPING.copy()

with st.sidebar.expander("Editar ativos", expanded=False):
    st.session_state["asset_mapping"] = st.data_editor(
        st.session_state["asset_mapping"],
        hide_index=True,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "Classe": st.column_config.TextColumn("Classe"),
            "Fonte": st.column_config.SelectboxColumn("Fonte", options=["yfinance", "bcb"]),
            "Identificador": st.column_config.TextColumn("Ticker/Código"),
            "Retorno Esperado (%)": st.column_config.NumberColumn("μ (%)", min_value=-50, max_value=100, step=0.5),
            "Min (%)": st.column_config.NumberColumn("Min (%)", min_value=0, max_value=100, step=1),
            "Max (%)": st.column_config.NumberColumn("Max (%)", min_value=0, max_value=100, step=1),
        }
    )

# Fetch data button
fetch_data = st.sidebar.button("🔄 Buscar Dados de Mercado", type="primary", use_container_width=True)

# =====================
# Main Content
# =====================
tab_data, tab_params, tab_opt, tab_frontier = st.tabs([
    "📊 Dados", "📈 Parâmetros", "🎯 Otimização", "📉 Fronteira Eficiente"
])

# =====================
# Tab: Data
# =====================
with tab_data:
    st.subheader("Dados de Mercado")
    
    if fetch_data or "prices" in st.session_state:
        if fetch_data:
            with st.spinner("Buscando dados de mercado..."):
                prices, failures = build_dataset(
                    st.session_state["asset_mapping"],
                    start=start_date.strftime("%Y-%m-%d")
                )
                
                if not prices.empty:
                    st.session_state["prices"] = prices
                    st.session_state["failures"] = failures
                    st.success(f"✅ Dados carregados: {len(prices)} dias, {len(prices.columns)} ativos")
                else:
                    st.error("❌ Não foi possível carregar os dados")
        
        if "prices" in st.session_state:
            prices = st.session_state["prices"]
            
            # Show failures if any
            if "failures" in st.session_state and not st.session_state["failures"].empty:
                with st.expander("⚠️ Alguns ativos falharam"):
                    st.dataframe(st.session_state["failures"], use_container_width=True)
            
            # Price chart
            st.markdown("### 📈 Evolução dos Preços (Normalizado)")
            prices_norm = prices / prices.iloc[0] * 100
            st.line_chart(prices_norm)
            
            # Sample data
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Primeiras linhas")
                st.dataframe(prices.head(), use_container_width=True)
            with col2:
                st.markdown("### Últimas linhas")
                st.dataframe(prices.tail(), use_container_width=True)
            
            # Download
            st.download_button(
                "📥 Baixar preços (CSV)",
                data=prices.to_csv().encode(),
                file_name="precos_historicos.csv",
                mime="text/csv"
            )
    else:
        st.info("👆 Clique em 'Buscar Dados de Mercado' na barra lateral para começar.")

# =====================
# Tab: Parameters
# =====================
with tab_params:
    st.subheader("Parâmetros Estimados")
    
    if "prices" not in st.session_state:
        st.warning("⚠️ Carregue os dados primeiro na aba 'Dados'")
        st.stop()
    
    prices = st.session_state["prices"]
    
    # Calculate returns
    returns = calc_log_returns(prices).tail(window_days)
    
    # Calculate statistics
    mu_hist, Sigma_raw = calc_annualized_stats(returns)
    
    # Apply covariance method
    if cov_method == "Ledoit-Wolf (shrinkage)":
        Sigma = ledoit_wolf_cov(returns) * 252
    else:
        Sigma = Sigma_raw
    
    Sigma = regularize_cov(Sigma)
    
    # Store in session
    st.session_state["mu_hist"] = mu_hist
    st.session_state["Sigma"] = Sigma
    st.session_state["returns"] = returns
    
    # Display
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Retornos Anualizados (Histórico)")
        mu_df = pd.DataFrame({
            "Ativo": mu_hist.index,
            "Retorno Histórico (%)": (mu_hist * 100).round(2),
        })
        st.dataframe(mu_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("### 📉 Volatilidades Anualizadas")
        vol = np.sqrt(np.diag(Sigma.values)) * 100
        vol_df = pd.DataFrame({
            "Ativo": Sigma.index,
            "Volatilidade (%)": vol.round(2)
        })
        st.dataframe(vol_df, use_container_width=True, hide_index=True)
    
    # Correlation matrix
    st.markdown("### 🔗 Matriz de Correlação")
    corr = returns.corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1)
    
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)
    
    # Add correlation values
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", 
                   color="black" if abs(corr.iloc[i, j]) < 0.5 else "white", fontsize=10)
    
    plt.colorbar(im, ax=ax, label="Correlação")
    plt.title("Matriz de Correlação (Calculada)")
    plt.tight_layout()
    st.pyplot(fig)
    
    # Download
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("📥 Volatilidades (CSV)", 
                          data=vol_df.to_csv(index=False).encode(),
                          file_name="volatilidades.csv")
    with col2:
        st.download_button("📥 Correlações (CSV)",
                          data=corr.to_csv().encode(),
                          file_name="correlacoes.csv")

# =====================
# Tab: Optimization
# =====================
with tab_opt:
    st.subheader("Otimização de Portfólio")
    
    if "Sigma" not in st.session_state:
        st.warning("⚠️ Calcule os parâmetros primeiro na aba 'Parâmetros'")
        st.stop()
    
    Sigma = st.session_state["Sigma"]
    mapping = st.session_state["asset_mapping"]
    assets = list(Sigma.index)
    
    # Get expected returns (user-defined or historical)
    use_custom_mu = st.checkbox("Usar retornos esperados personalizados", value=True)
    
    if use_custom_mu:
        mu_custom = mapping.set_index("Classe")["Retorno Esperado (%)"].reindex(assets).fillna(10) / 100
        mu = mu_custom.values
    else:
        mu = st.session_state["mu_hist"].values
    
    # Get bounds
    lb = mapping.set_index("Classe")["Min (%)"].reindex(assets).fillna(0).values / 100
    ub = mapping.set_index("Classe")["Max (%)"].reindex(assets).fillna(100).values / 100
    
    # Optimization objective
    objective = st.radio(
        "Objetivo",
        ["Máximo Sharpe", "Mínima Variância"],
        horizontal=True
    )
    
    if st.button("🚀 Otimizar Portfólio", type="primary"):
        with st.spinner("Otimizando..."):
            Sigma_np = Sigma.values
            
            if objective == "Máximo Sharpe":
                w = optimize_max_sharpe(mu, Sigma_np, rf, lb, ub)
            else:
                w = optimize_min_variance(mu, Sigma_np, lb, ub)
            
            if w is not None:
                st.session_state["optimal_weights"] = w
                
                # Metrics
                port = portfolio_metrics(w, mu, Sigma_np, rf)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📈 Retorno Esperado", f"{port['return']*100:.2f}%")
                with col2:
                    st.metric("📉 Volatilidade", f"{port['volatility']*100:.2f}%")
                with col3:
                    st.metric("⭐ Sharpe Ratio", f"{port['sharpe']:.2f}")
                
                # Weights table
                st.markdown("### 📊 Alocação Ótima")
                weights_df = pd.DataFrame({
                    "Ativo": assets,
                    "Peso (%)": (w * 100).round(2)
                }).sort_values("Peso (%)", ascending=False)
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.dataframe(weights_df, use_container_width=True, hide_index=True)
                
                with col2:
                    # Pie chart
                    fig, ax = plt.subplots(figsize=(8, 8))
                    weights_nonzero = weights_df[weights_df["Peso (%)"] > 0.5]
                    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(weights_nonzero)))
                    ax.pie(weights_nonzero["Peso (%)"], labels=weights_nonzero["Ativo"],
                          autopct='%1.1f%%', colors=colors, startangle=90)
                    ax.set_title("Distribuição do Portfólio")
                    st.pyplot(fig)
                
                # Download
                st.download_button(
                    "📥 Baixar pesos (CSV)",
                    data=weights_df.to_csv(index=False).encode(),
                    file_name="pesos_otimos.csv"
                )
            else:
                st.error("❌ Não foi possível encontrar solução. Ajuste os limites.")

# =====================
# Tab: Frontier
# =====================
with tab_frontier:
    st.subheader("Fronteira Eficiente")
    
    if "Sigma" not in st.session_state:
        st.warning("⚠️ Calcule os parâmetros primeiro na aba 'Parâmetros'")
        st.stop()
    
    Sigma = st.session_state["Sigma"]
    mapping = st.session_state["asset_mapping"]
    assets = list(Sigma.index)
    
    # Get parameters
    mu_custom = mapping.set_index("Classe")["Retorno Esperado (%)"].reindex(assets).fillna(10) / 100
    mu = mu_custom.values
    lb = mapping.set_index("Classe")["Min (%)"].reindex(assets).fillna(0).values / 100
    ub = mapping.set_index("Classe")["Max (%)"].reindex(assets).fillna(100).values / 100
    
    n_points = st.slider("Número de pontos na fronteira", 10, 50, 30)
    
    if st.button("📈 Calcular Fronteira Eficiente", type="primary"):
        with st.spinner("Calculando fronteira..."):
            frontier = compute_frontier(mu, Sigma.values, lb, ub, n_points)
            
            if frontier:
                st.session_state["frontier"] = frontier
                
                # Plot
                fig, ax = plt.subplots(figsize=(12, 8))
                
                rets = [p["return"] * 100 for p in frontier]
                vols = [p["volatility"] * 100 for p in frontier]
                sharpes = [p["sharpe"] for p in frontier]
                
                # Frontier line
                scatter = ax.scatter(vols, rets, c=sharpes, cmap="viridis", s=100, zorder=5)
                ax.plot(vols, rets, 'b-', alpha=0.5, linewidth=2, zorder=4)
                
                # Individual assets
                asset_vols = np.sqrt(np.diag(Sigma.values)) * 100
                asset_rets = mu * 100
                ax.scatter(asset_vols, asset_rets, c='red', s=150, marker='^', zorder=6, label='Ativos individuais')
                
                for i, asset in enumerate(assets):
                    ax.annotate(asset, (asset_vols[i], asset_rets[i]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=9)
                
                # Find special points
                min_var_idx = np.argmin(vols)
                max_sharpe_idx = np.argmax(sharpes)
                
                ax.scatter(vols[min_var_idx], rets[min_var_idx], 
                          c='green', s=250, marker='*', zorder=7, label='Mínima Variância')
                ax.scatter(vols[max_sharpe_idx], rets[max_sharpe_idx],
                          c='gold', s=250, marker='*', zorder=7, label='Máximo Sharpe')
                
                ax.set_xlabel("Volatilidade (%)", fontsize=12)
                ax.set_ylabel("Retorno Esperado (%)", fontsize=12)
                ax.set_title("Fronteira Eficiente de Markowitz", fontsize=14)
                ax.legend(loc='upper left')
                ax.grid(True, alpha=0.3)
                
                plt.colorbar(scatter, ax=ax, label="Sharpe Ratio")
                plt.tight_layout()
                st.pyplot(fig)
                
                # Save plot
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                st.download_button("📥 Baixar Fronteira (PNG)",
                                  data=buf.getvalue(),
                                  file_name="fronteira_eficiente.png",
                                  mime="image/png")
                
                # Summary table
                st.markdown("### 📋 Pontos da Fronteira")
                frontier_df = pd.DataFrame({
                    "Retorno (%)": [p["return"] * 100 for p in frontier],
                    "Volatilidade (%)": [p["volatility"] * 100 for p in frontier],
                    "Sharpe": [p["sharpe"] for p in frontier]
                }).round(2)
                st.dataframe(frontier_df, use_container_width=True, hide_index=True)
            else:
                st.error("❌ Não foi possível calcular a fronteira. Ajuste os parâmetros.")

# =====================
# Footer
# =====================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64748b; font-size: 12px;'>
    📊 Portfolio Optimizer Pro | Dados: BCB, Yahoo Finance | Otimização: CVXPY<br>
    Volatilidades e correlações calculadas a partir de séries históricas reais
</div>
""", unsafe_allow_html=True)
