import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import io
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="MITS-AppsFlyer Campaign Analytics Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    /* Main background and text */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #1f2937;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        color: #6b7280;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Headers */
    h1 {
        color: #1f2937;
        font-weight: 800;
        letter-spacing: -0.5px;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #3b82f6;
    }
    
    h2 {
        color: #374151;
        font-weight: 700;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    h3 {
        color: #4b5563;
        font-weight: 600;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3a8a 0%, #1e40af 100%);
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #e5e7eb;
    }
    
    [data-testid="stSidebar"] label {
        color: #e5e7eb !important;
        font-weight: 600;
    }
    
    /* Cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border-left: 4px solid #3b82f6;
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: white;
        padding: 10px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #f3f4f6;
        border-radius: 8px;
        color: #374151;
        font-weight: 600;
        padding: 0 24px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
        color: white;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(59, 130, 246, 0.4);
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 10px;
        border-left: 4px solid #3b82f6;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: white;
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* Significance badge */
    .sig-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .sig-high { background: #10b981; color: white; }
    .sig-medium { background: #f59e0b; color: white; }
    .sig-low { background: #ef4444; color: white; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA LOADING & PROCESSING
# ============================================================================
@st.cache_data
def load_data(file, platform):
    """Load and preprocess data with platform label"""
    df = pd.read_csv(file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # Add platform column
    df['Platform'] = platform
    
    # Fill NaN with 0
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    # Add derived columns
    df['Day_of_Week'] = df['Date'].dt.day_name()
    df['Week_Number'] = df['Date'].dt.isocalendar().week
    
    # Identify Media Source Type (Paid/Organic)
    if 'Media Source' in df.columns:
        df['Source_Type'] = df['Media Source'].apply(lambda x: 'Paid' if 'Paid' in str(x) else 'Organic')
    else:
        df['Source_Type'] = 'Unknown'
    
    # Calculate conversion rates
    df['Install_to_Account_Rate'] = np.where(df['Installs'] > 0, 
                                              df['account number (Unique users)'] / df['Installs'] * 100, 0)
    
    df['Install_to_OTP_Rate'] = np.where(df['Installs'] > 0,
                                          df['confirm_mobileotp (Unique users)'] / df['Installs'] * 100, 0)
    
    df['Install_to_KYC_Rate'] = np.where(df['Installs'] > 0, 
                                          df['verify_kyc (Unique users)'] / df['Installs'] * 100, 0)
    
    df['KYC_to_Account_Rate'] = np.where(df['verify_kyc (Unique users)'] > 0,
                                          df['account number (Unique users)'] / df['verify_kyc (Unique users)'] * 100, 0)
    
    df['Account_to_Trade_Rate'] = np.where(df['account number (Unique users)'] > 0,
                                            df['any_trade_s2s (Unique users)'] / df['account number (Unique users)'] * 100, 0)
    
    df['Session_per_Install'] = np.where(df['Installs'] > 0, df['Sessions'] / df['Installs'], 0)
    df['Loyal_User_Rate'] = df['Loyal Users/Installs'] * 100
    
    # CTR and Conversion Rate
    df['CTR'] = np.where(df['Impressions'] > 0, df['Clicks'] / df['Impressions'] * 100, 0)
    df['Click_to_Install_Rate'] = np.where(df['Clicks'] > 0, df['Installs'] / df['Clicks'] * 100, 0)
    
    return df

@st.cache_data
def load_spends_data(file):
    """Load daily spends data"""
    df = pd.read_csv(file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # Ensure Spends column exists
    if 'Spends' not in df.columns and 'Spend' in df.columns:
        df['Spends'] = df['Spend']
    elif 'Spends' not in df.columns and 'Cost' in df.columns:
        df['Spends'] = df['Cost']
    
    return df[['Date', 'Spends']]

def combine_platform_data(ios_file, android_file):
    """Combine iOS and Android data"""
    dfs = []
    
    if ios_file is not None:
        ios_df = load_data(ios_file, 'iOS')
        dfs.append(ios_df)
    
    if android_file is not None:
        android_df = load_data(android_file, 'Android')
        dfs.append(android_df)
    
    if len(dfs) == 0:
        return None
    
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df

def aggregate_daily_data(df):
    """Aggregate data by date for time series analysis"""
    daily_agg = df.groupby('Date').agg({
        'Impressions': 'sum',
        'Clicks': 'sum',
        'Installs': 'sum',
        'Sessions': 'sum',
        'account number (Unique users)': 'sum',
        'confirm_mobileotp (Unique users)': 'sum',
        'verify_kyc (Unique users)': 'sum',
        'any_trade_s2s (Unique users)': 'sum',
        'Loyal Users': 'sum'
    }).reset_index()
    
    # Recalculate rates
    daily_agg['CTR'] = np.where(daily_agg['Impressions'] > 0, 
                                 daily_agg['Clicks'] / daily_agg['Impressions'] * 100, 0)
    daily_agg['Click_to_Install_Rate'] = np.where(daily_agg['Clicks'] > 0,
                                                    daily_agg['Installs'] / daily_agg['Clicks'] * 100, 0)
    daily_agg['Install_to_Account_Rate'] = np.where(daily_agg['Installs'] > 0,
                                                      daily_agg['account number (Unique users)'] / daily_agg['Installs'] * 100, 0)
    daily_agg['Install_to_OTP_Rate'] = np.where(daily_agg['Installs'] > 0,
                                                  daily_agg['confirm_mobileotp (Unique users)'] / daily_agg['Installs'] * 100, 0)
    daily_agg['Install_to_KYC_Rate'] = np.where(daily_agg['Installs'] > 0,
                                                  daily_agg['verify_kyc (Unique users)'] / daily_agg['Installs'] * 100, 0)
    
    # ADD THESE MISSING COLUMNS:
    daily_agg['KYC_to_Account_Rate'] = np.where(daily_agg['verify_kyc (Unique users)'] > 0,
                                                  daily_agg['account number (Unique users)'] / daily_agg['verify_kyc (Unique users)'] * 100, 0)
    
    daily_agg['Account_to_Trade_Rate'] = np.where(daily_agg['account number (Unique users)'] > 0,
                                                    daily_agg['any_trade_s2s (Unique users)'] / daily_agg['account number (Unique users)'] * 100, 0)
    
    return daily_agg

def merge_with_spends(daily_df, spends_df):
    """Merge daily aggregated data with spends"""
    merged = daily_df.merge(spends_df, on='Date', how='left')
    merged['Spends'] = merged['Spends'].fillna(0)
    
    # Calculate cost metrics
    merged['CPI'] = np.where(merged['Installs'] > 0, merged['Spends'] / merged['Installs'], 0)
    merged['CPC'] = np.where(merged['Clicks'] > 0, merged['Spends'] / merged['Clicks'], 0)
    merged['CPM'] = np.where(merged['Impressions'] > 0, 
                             (merged['Spends'] / merged['Impressions']) * 1000, 0)
    
    return merged

def aggregate_by_source_type(df):
    """Aggregate data by source type (Paid/Organic) and date"""
    if 'Source_Type' not in df.columns:
        return None
    
    source_agg = df.groupby(['Date', 'Source_Type']).agg({
        'Impressions': 'sum',
        'Clicks': 'sum',
        'Installs': 'sum',
        'Sessions': 'sum',
        'account number (Unique users)': 'sum',
        'confirm_mobileotp (Unique users)': 'sum',
        'verify_kyc (Unique users)': 'sum',
        'any_trade_s2s (Unique users)': 'sum'
    }).reset_index()
    
    # Recalculate rates
    source_agg['Install_to_Account_Rate'] = np.where(source_agg['Installs'] > 0,
                                                       source_agg['account number (Unique users)'] / source_agg['Installs'] * 100, 0)
    source_agg['Install_to_OTP_Rate'] = np.where(source_agg['Installs'] > 0,
                                                   source_agg['confirm_mobileotp (Unique users)'] / source_agg['Installs'] * 100, 0)
    source_agg['Install_to_KYC_Rate'] = np.where(source_agg['Installs'] > 0,
                                                   source_agg['verify_kyc (Unique users)'] / source_agg['Installs'] * 100, 0)
    
    return source_agg

# ============================================================================
# CORRELATION & STATISTICAL ANALYSIS
# ============================================================================
def calculate_correlation_with_lag(df, x_col, y_col, max_lag=7):
    """Calculate correlation with lag analysis (0 to max_lag days)"""
    results = []
    
    for lag in range(max_lag + 1):
        if lag == 0:
            x_series = df[x_col]
            y_series = df[y_col]
        else:
            x_series = df[x_col].shift(lag)
            y_series = df[y_col]
        
        # Remove NaN values
        valid_mask = ~(x_series.isna() | y_series.isna())
        x_clean = x_series[valid_mask]
        y_clean = y_series[valid_mask]
        
        if len(x_clean) > 2:
            pearson_corr, pearson_p = pearsonr(x_clean, y_clean)
            spearman_corr, spearman_p = spearmanr(x_clean, y_clean)
            
            results.append({
                'Lag (days)': lag,
                'Pearson Correlation': pearson_corr,
                'Pearson p-value': pearson_p,
                'Spearman Correlation': spearman_corr,
                'Spearman p-value': spearman_p,
                'Significant': 'Yes' if pearson_p < 0.05 else 'No'
            })
    
    return pd.DataFrame(results)

def calculate_lift(paid_df, organic_df, metric):
    """Calculate lift of paid over organic"""
    paid_mean = paid_df[metric].mean()
    organic_mean = organic_df[metric].mean()
    
    if organic_mean > 0:
        lift = ((paid_mean - organic_mean) / organic_mean) * 100
    else:
        lift = 0
    
    # Statistical test
    if len(paid_df) > 1 and len(organic_df) > 1:
        t_stat, p_value = stats.ttest_ind(paid_df[metric].dropna(), organic_df[metric].dropna())
    else:
        t_stat, p_value = 0, 1
    
    return {
        'Paid Mean': paid_mean,
        'Organic Mean': organic_mean,
        'Lift %': lift,
        'T-statistic': t_stat,
        'P-value': p_value,
        'Significant': 'Yes' if p_value < 0.05 else 'No'
    }

def calculate_correlation_matrix(df, metrics):
    """Calculate correlation matrix with p-values"""
    corr_matrix = df[metrics].corr()
    
    # Calculate p-values
    p_values = pd.DataFrame(np.zeros_like(corr_matrix), columns=metrics, index=metrics)
    
    for i, metric1 in enumerate(metrics):
        for j, metric2 in enumerate(metrics):
            if i != j and len(df[[metric1, metric2]].dropna()) > 2:
                _, p_val = pearsonr(df[metric1].dropna(), df[metric2].dropna())
                p_values.iloc[i, j] = p_val
    
    return corr_matrix, p_values

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================
def plot_correlation_heatmap(corr_matrix, p_values, title):
    """Plot correlation heatmap with significance markers"""
    # Create significance annotations
    annotations = []
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            p_val = p_values.iloc[i, j]
            
            # Add asterisks for significance
            sig_marker = ''
            if i != j:
                if p_val < 0.001:
                    sig_marker = '***'
                elif p_val < 0.01:
                    sig_marker = '**'
                elif p_val < 0.05:
                    sig_marker = '*'
            
            text = f'{corr_val:.3f}{sig_marker}'
            
            annotations.append(
                dict(
                    x=j,
                    y=i,
                    text=text,
                    showarrow=False,
                    font=dict(
                        color='white' if abs(corr_val) > 0.5 else 'black', 
                        size=11,
                        family='Arial, sans-serif'
                    )
                )
            )
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu_r',
        zmid=0,
        colorbar=dict(title="Correlation"),
        hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>',
        showscale=True
    ))
    
    fig.update_layout(
        title=title,
        height=600,
        xaxis={'side': 'bottom'},
        annotations=annotations
    )
    
    return fig
def plot_lag_correlation(lag_df, metric_name):
    """Plot correlation vs lag days"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=lag_df['Lag (days)'],
        y=lag_df['Pearson Correlation'],
        mode='lines+markers',
        name='Pearson',
        line=dict(color='#3b82f6', width=3),
        marker=dict(size=10)
    ))
    
    fig.add_trace(go.Scatter(
        x=lag_df['Lag (days)'],
        y=lag_df['Spearman Correlation'],
        mode='lines+markers',
        name='Spearman',
        line=dict(color='#10b981', width=3),
        marker=dict(size=10)
    ))
    
    # Add significance threshold
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    fig.update_layout(
        title=f"Lagged Correlation: {metric_name}",
        xaxis_title="Lag (Days)",
        yaxis_title="Correlation Coefficient",
        height=400,
        hovermode='x unified'
    )
    
    return fig

def plot_scatter_with_regression(df, x_col, y_col, x_label, y_label, color_col=None):
    """Create scatter plot with regression line"""
    if color_col and color_col in df.columns:
        fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                        labels={x_col: x_label, y_col: y_label},
                        trendline="ols",
                        hover_data=['Date'] if 'Date' in df.columns else None)
    else:
        fig = px.scatter(df, x=x_col, y=y_col,
                        labels={x_col: x_label, y_col: y_label},
                        trendline="ols",
                        hover_data=['Date'] if 'Date' in df.columns else None)
    
    # Calculate R-squared
    if len(df[[x_col, y_col]].dropna()) > 2:
        corr, p_val = pearsonr(df[x_col].dropna(), df[y_col].dropna())
        r_squared = corr ** 2
        
        fig.add_annotation(
            text=f'R² = {r_squared:.3f}<br>p-value = {p_val:.4f}',
            xref='paper', yref='paper',
            x=0.05, y=0.95,
            showarrow=False,
            bgcolor='white',
            bordercolor='black',
            borderwidth=1
        )
    
    fig.update_layout(height=450)
    return fig

def plot_paid_vs_organic_comparison(source_agg, metric, metric_label):
    """Plot paid vs organic comparison over time"""
    fig = go.Figure()
    
    paid_data = source_agg[source_agg['Source_Type'] == 'Paid']
    organic_data = source_agg[source_agg['Source_Type'] == 'Organic']
    
    fig.add_trace(go.Scatter(
        x=paid_data['Date'],
        y=paid_data[metric],
        mode='lines+markers',
        name='Paid',
        line=dict(color='#3b82f6', width=3),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=organic_data['Date'],
        y=organic_data[metric],
        mode='lines+markers',
        name='Organic',
        line=dict(color='#10b981', width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title=f"{metric_label}: Paid vs Organic",
        xaxis_title="Date",
        yaxis_title=metric_label,
        height=450,
        hovermode='x unified'
    )
    
    return fig

def plot_funnel_metrics(df, show_spends=False):
    """Plot funnel metrics with optional spends overlay"""
    fig = go.Figure()
    
    # Funnel metrics
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Impressions'],
        mode='lines',
        name='Impressions',
        line=dict(color='#93c5fd', width=2),
        yaxis='y'
    ))
    
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Clicks'],
        mode='lines',
        name='Clicks',
        line=dict(color='#60a5fa', width=2),
        yaxis='y'
    ))
    
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Installs'],
        mode='lines+markers',
        name='Installs',
        line=dict(color='#3b82f6', width=3),
        marker=dict(size=8),
        yaxis='y'
    ))
    
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['verify_kyc (Unique users)'],
        mode='lines+markers',
        name='KYC Verified',
        line=dict(color='#10b981', width=3),
        marker=dict(size=8),
        yaxis='y'
    ))
    
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['account number (Unique users)'],
        mode='lines+markers',
        name='Accounts Created',
        line=dict(color='#f59e0b', width=3),
        marker=dict(size=8),
        yaxis='y'
    ))
    
    if show_spends and 'Spends' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['Spends'],
            mode='lines',
            name='Spends (INR)',
            line=dict(color='#ef4444', width=3, dash='dash'),
            yaxis='y2'
        ))
        
        fig.update_layout(
            yaxis2=dict(
                title="Spends (INR)",
                overlaying='y',
                side='right'
            )
        )
    
    fig.update_layout(
        title="Campaign Funnel Metrics Over Time",
        xaxis_title="Date",
        yaxis_title="Count",
        height=500,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def plot_conversion_rates_over_time(df):
    """Plot conversion rates over time"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Install_to_OTP_Rate'],
        mode='lines+markers',
        name='Install → OTP %',
        line=dict(color='#3b82f6', width=2),
        marker=dict(size=6)
    ))
    
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Install_to_KYC_Rate'],
        mode='lines+markers',
        name='Install → KYC %',
        line=dict(color='#10b981', width=2),
        marker=dict(size=6)
    ))
    
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Install_to_Account_Rate'],
        mode='lines+markers',
        name='Install → Account %',
        line=dict(color='#f59e0b', width=2),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title="Conversion Rates Over Time",
        xaxis_title="Date",
        yaxis_title="Conversion Rate (%)",
        height=450,
        hovermode='x unified'
    )
    
    return fig


def aggregate_by_platform(df):
    """Aggregate data by platform (iOS/Android) and date"""
    if 'Platform' not in df.columns:
        return None
    
    platform_agg = df.groupby(['Date', 'Platform']).agg({
        'Impressions': 'sum',
        'Clicks': 'sum',
        'Installs': 'sum',
        'Sessions': 'sum',
        'account number (Unique users)': 'sum',
        'confirm_mobileotp (Unique users)': 'sum',
        'verify_kyc (Unique users)': 'sum',
        'any_trade_s2s (Unique users)': 'sum'
    }).reset_index()
    
    # Calculate conversion rates using actual conversion rate formula
    platform_agg['Install_to_KYC_Rate'] = np.where(
        platform_agg['Installs'] > 0,
        (platform_agg['verify_kyc (Unique users)'] / platform_agg['Installs']) * 100, 0
    )
    
    platform_agg['KYC_to_Account_Rate'] = np.where(
        platform_agg['verify_kyc (Unique users)'] > 0,
        (platform_agg['account number (Unique users)'] / platform_agg['verify_kyc (Unique users)']) * 100, 0
    )
    
    platform_agg['Account_to_Trade_Rate'] = np.where(
        platform_agg['account number (Unique users)'] > 0,
        (platform_agg['any_trade_s2s (Unique users)'] / platform_agg['account number (Unique users)']) * 100, 0
    )
    
    platform_agg['Install_to_Account_Rate'] = np.where(
        platform_agg['Installs'] > 0,
        (platform_agg['account number (Unique users)'] / platform_agg['Installs']) * 100, 0
    )
    
    platform_agg['Install_to_Trade_Rate'] = np.where(
        platform_agg['Installs'] > 0,
        (platform_agg['any_trade_s2s (Unique users)'] / platform_agg['Installs']) * 100, 0
    )
    
    return platform_agg

def get_platform_summary(df, platform_name):
    """Get summary statistics for a specific platform"""
    platform_df = df[df['Platform'] == platform_name]
    
    total_installs = platform_df['Installs'].sum()
    total_kyc = platform_df['verify_kyc (Unique users)'].sum()
    total_accounts = platform_df['account number (Unique users)'].sum()
    total_trades = platform_df['any_trade_s2s (Unique users)'].sum()
    
    # Calculate conversion rates as actual rates, not means
    install_to_kyc = (total_kyc / total_installs * 100) if total_installs > 0 else 0
    kyc_to_account = (total_accounts / total_kyc * 100) if total_kyc > 0 else 0
    account_to_trade = (total_trades / total_accounts * 100) if total_accounts > 0 else 0
    install_to_account = (total_accounts / total_installs * 100) if total_installs > 0 else 0
    install_to_trade = (total_trades / total_installs * 100) if total_installs > 0 else 0
    
    return {
        'Installs': int(total_installs),
        'KYC Verified': int(total_kyc),
        'Accounts': int(total_accounts),
        'Traders': int(total_trades),
        'Install→KYC %': install_to_kyc,
        'KYC→Account %': kyc_to_account,
        'Account→Trade %': account_to_trade,
        'Install→Account %': install_to_account,
        'Install→Trade %': install_to_trade
    }

def plot_platform_comparison_bars(platform_agg):
    """Create side-by-side bar chart for platform conversion rates"""
    # Calculate overall conversion rates per platform
    ios_data = platform_agg[platform_agg['Platform'] == 'iOS']
    android_data = platform_agg[platform_agg['Platform'] == 'Android']
    
    # Calculate actual conversion rates
    metrics = []
    ios_values = []
    android_values = []
    
    # Install → KYC
    if ios_data['Installs'].sum() > 0:
        ios_kyc = (ios_data['verify_kyc (Unique users)'].sum() / ios_data['Installs'].sum()) * 100
    else:
        ios_kyc = 0
    
    if android_data['Installs'].sum() > 0:
        android_kyc = (android_data['verify_kyc (Unique users)'].sum() / android_data['Installs'].sum()) * 100
    else:
        android_kyc = 0
    
    metrics.append('Install → KYC')
    ios_values.append(ios_kyc)
    android_values.append(android_kyc)
    
    # KYC → Account
    if ios_data['verify_kyc (Unique users)'].sum() > 0:
        ios_account = (ios_data['account number (Unique users)'].sum() / ios_data['verify_kyc (Unique users)'].sum()) * 100
    else:
        ios_account = 0
    
    if android_data['verify_kyc (Unique users)'].sum() > 0:
        android_account = (android_data['account number (Unique users)'].sum() / android_data['verify_kyc (Unique users)'].sum()) * 100
    else:
        android_account = 0
    
    metrics.append('KYC → Account')
    ios_values.append(ios_account)
    android_values.append(android_account)
    
    # Account → Trade
    if ios_data['account number (Unique users)'].sum() > 0:
        ios_trade = (ios_data['any_trade_s2s (Unique users)'].sum() / ios_data['account number (Unique users)'].sum()) * 100
    else:
        ios_trade = 0
    
    if android_data['account number (Unique users)'].sum() > 0:
        android_trade = (android_data['any_trade_s2s (Unique users)'].sum() / android_data['account number (Unique users)'].sum()) * 100
    else:
        android_trade = 0
    
    metrics.append('Account → Trade')
    ios_values.append(ios_trade)
    android_values.append(android_trade)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='iOS',
        x=metrics,
        y=ios_values,
        text=[f'{v:.2f}%' for v in ios_values],
        textposition='outside',
        marker_color='#3b82f6'
    ))
    
    fig.add_trace(go.Bar(
        name='Android',
        x=metrics,
        y=android_values,
        text=[f'{v:.2f}%' for v in android_values],
        textposition='outside',
        marker_color='#10b981'
    ))
    
    fig.update_layout(
        title='Conversion Rates by Platform',
        xaxis_title='Conversion Stage',
        yaxis_title='Conversion Rate (%)',
        barmode='group',
        height=450,
        yaxis=dict(range=[0, max(max(ios_values), max(android_values)) * 1.2])
    )
    
    return fig

def plot_funnel_chart(platform_data, platform_name):
    """Create funnel chart for a specific platform"""
    total_installs = platform_data['Installs'].sum()
    total_kyc = platform_data['verify_kyc (Unique users)'].sum()
    total_accounts = platform_data['account number (Unique users)'].sum()
    total_trades = platform_data['any_trade_s2s (Unique users)'].sum()
    
    stages = ['Installs', 'KYC Verified', 'Accounts Created', 'Traders']
    values = [total_installs, total_kyc, total_accounts, total_trades]
    
    fig = go.Figure(go.Funnel(
        y=stages,
        x=values,
        textposition="inside",
        textinfo="value+percent initial",
        marker=dict(
            color=['#93c5fd', '#60a5fa', '#3b82f6', '#2563eb']
        ),
        connector=dict(line=dict(color='royalblue', width=2))
    ))
    
    fig.update_layout(
        title=f'{platform_name} User Journey Funnel',
        height=400
    )
    
    return fig

def plot_time_series_with_baseline(df, metric_col, metric_label, baseline_start, baseline_end, campaign_start, campaign_end):
    """Plot time series with baseline and campaign periods highlighted"""
    fig = go.Figure()
    
    # Full data
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df[metric_col],
        mode='lines+markers',
        name=metric_label,
        line=dict(color='#3b82f6', width=2),
        marker=dict(size=6)
    ))
    
    # Add baseline period shading
    fig.add_vrect(
        x0=baseline_start, x1=baseline_end,
        fillcolor="gray", opacity=0.2,
        layer="below", line_width=0,
        annotation_text="Baseline", annotation_position="top left"
    )
    
    # Add campaign period shading
    fig.add_vrect(
        x0=campaign_start, x1=campaign_end,
        fillcolor="green", opacity=0.1,
        layer="below", line_width=0,
        annotation_text="Campaign", annotation_position="top left"
    )
    
    # Calculate averages for both periods
    baseline_df = df[(df['Date'] >= baseline_start) & (df['Date'] <= baseline_end)]
    campaign_df = df[(df['Date'] >= campaign_start) & (df['Date'] <= campaign_end)]
    
    baseline_avg = baseline_df[metric_col].mean()
    campaign_avg = campaign_df[metric_col].mean()
    
    # Add average lines
    fig.add_hline(
        y=baseline_avg,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Baseline Avg: {baseline_avg:.2f}",
        annotation_position="right"
    )
    
    fig.add_hline(
        y=campaign_avg,
        line_dash="dash",
        line_color="green",
        annotation_text=f"Campaign Avg: {campaign_avg:.2f}",
        annotation_position="right"
    )
    
    fig.update_layout(
        title=f"{metric_label} - Baseline vs Campaign Period",
        xaxis_title="Date",
        yaxis_title=metric_label,
        height=450,
        hovermode='x unified'
    )
    
    return fig, baseline_avg, campaign_avg




# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    st.title("📊 AppsFlyer Campaign Analytics Pro")
    st.markdown("### Advanced Correlation Analysis, Paid vs Organic, and Campaign Performance")
    
    # ============================================================================
    # SIDEBAR - FILE UPLOADS
    # ============================================================================
    with st.sidebar:
        st.markdown("## 📁 Data Upload")
        
        st.markdown("#### Platform Data")
        ios_file = st.file_uploader("Upload iOS Data (CSV)", type=['csv'], key='ios')
        android_file = st.file_uploader("Upload Android Data (CSV)", type=['csv'], key='android')
        
        st.markdown("#### Campaign Spends Data")
        st.info("📌 Upload CSV with 'Date' and 'Spends' columns")
        spends_file = st.file_uploader("Upload Daily Spends (CSV)", type=['csv'], key='spends')
        
        st.markdown("---")
        
        st.markdown("## ⚙️ Analysis Settings")
        
        # Date range filter
        use_date_filter = st.checkbox("Filter by Date Range", value=False)
        if use_date_filter:
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date")
            with col2:
                end_date = st.date_input("End Date")
        
        # Lag analysis settings
        st.markdown("#### Lag Analysis")
        max_lag_days = st.slider("Maximum Lag Days", 1, 14, 7)
        
        st.markdown("---")
        st.markdown("#### 📖 Instructions")
        st.markdown("""
        1. Upload iOS and/or Android CSV files
        2. Upload daily spends data (optional)
        3. Explore correlation & lift analysis
        4. Compare Paid vs Organic performance
        """)
    
    # ============================================================================
    # LOAD AND PROCESS DATA
    # ============================================================================
    if ios_file is None and android_file is None:
        st.info("👈 Please upload at least one platform data file to begin analysis")
        return
    
    # Load platform data
    combined_df = combine_platform_data(ios_file, android_file)
    
    if combined_df is None or len(combined_df) == 0:
        st.error("No data loaded. Please check your files.")
        return
    
    # Apply date filter if enabled
    if use_date_filter:
        combined_df = combined_df[(combined_df['Date'] >= pd.Timestamp(start_date)) & 
                                   (combined_df['Date'] <= pd.Timestamp(end_date))]
    
    # Aggregate data
    daily_agg = aggregate_daily_data(combined_df)
    source_agg = aggregate_by_source_type(combined_df)
    
    # Load and merge spends data
    has_spends = False
    if spends_file is not None:
        try:
            spends_df = load_spends_data(spends_file)
            daily_agg = merge_with_spends(daily_agg, spends_df)
            has_spends = True
        except Exception as e:
            st.warning(f"Could not load spends data: {e}")
    
    # ============================================================================
    # OVERVIEW METRICS
    # ============================================================================
    st.markdown("## 📈 Campaign Overview")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_impressions = int(combined_df['Impressions'].sum())
        st.metric("Total Impressions", f"{total_impressions:,}")
    
    with col2:
        total_clicks = int(combined_df['Clicks'].sum())
        avg_ctr = combined_df['CTR'].mean()
        st.metric("Total Clicks", f"{total_clicks:,}", f"CTR: {avg_ctr:.2f}%")
    
    with col3:
        total_installs = int(combined_df['Installs'].sum())
        st.metric("Total Installs", f"{total_installs:,}")
    
    with col4:
        total_kyc = int(combined_df['verify_kyc (Unique users)'].sum())
        kyc_rate = (total_kyc / total_installs * 100) if total_installs > 0 else 0
        st.metric("KYC Verified", f"{total_kyc:,}", f"{kyc_rate:.1f}%")
    
    with col5:
        total_accounts = int(combined_df['account number (Unique users)'].sum())
        account_rate = (total_accounts / total_installs * 100) if total_installs > 0 else 0
        st.metric("Accounts Created", f"{total_accounts:,}", f"{account_rate:.1f}%")
    
    if has_spends:
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_spends = daily_agg['Spends'].sum()
            st.metric("Total Spends", f"₹{total_spends:,.0f}")
        
        with col2:
            avg_cpi = daily_agg['CPI'].mean()
            st.metric("Avg CPI", f"₹{avg_cpi:.2f}")
        
        with col3:
            avg_cpc = daily_agg['CPC'].mean()
            st.metric("Avg CPC", f"₹{avg_cpc:.2f}")
        
        with col4:
            avg_cpm = daily_agg['CPM'].mean()
            st.metric("Avg CPM", f"₹{avg_cpm:.2f}")
    
    st.markdown("---")
    
    # ============================================================================
    # TABS FOR DIFFERENT ANALYSES
    # ============================================================================
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Funnel Analysis",
    "📱 iOS vs Android",  # NEW TAB
    "🔗 Correlation Analysis",
    "⏱️ Lag Analysis",
    "💰 Paid vs Organic",
    "📉 Lift Analysis",
    "📋 Data Export"
    ])
    # ============================================================================
    # TAB 1: FUNNEL ANALYSIS
    # ============================================================================
    with tab1:
        st.markdown("## Campaign Funnel Performance")
        
        # Funnel metrics over time
        st.plotly_chart(plot_funnel_metrics(daily_agg, show_spends=has_spends), use_container_width=True)
        
        st.markdown("---")
        
        # Conversion rates over time
        st.plotly_chart(plot_conversion_rates_over_time(daily_agg), use_container_width=True)
        
        st.markdown("---")
        
        # Summary statistics
        st.markdown("### 📊 Summary Statistics")
        
        summary_data = {
            'Metric': [
                'Impressions → Clicks (CTR)',
                'Clicks → Installs',
                'Installs → OTP',
                'Installs → KYC',
                'Installs → Account',
                'KYC → Account'
            ],
            'Mean %': [
                f"{daily_agg['CTR'].mean():.2f}%",
                f"{daily_agg['Click_to_Install_Rate'].mean():.2f}%",
                f"{daily_agg['Install_to_OTP_Rate'].mean():.2f}%",
                f"{daily_agg['Install_to_KYC_Rate'].mean():.2f}%",
                f"{daily_agg['Install_to_Account_Rate'].mean():.2f}%",
                f"{combined_df['KYC_to_Account_Rate'].mean():.2f}%"
            ],
            'Median %': [
                f"{daily_agg['CTR'].median():.2f}%",
                f"{daily_agg['Click_to_Install_Rate'].median():.2f}%",
                f"{daily_agg['Install_to_OTP_Rate'].median():.2f}%",
                f"{daily_agg['Install_to_KYC_Rate'].median():.2f}%",
                f"{daily_agg['Install_to_Account_Rate'].median():.2f}%",
                f"{combined_df['KYC_to_Account_Rate'].median():.2f}%"
            ],
            'Std Dev': [
                f"{daily_agg['CTR'].std():.2f}",
                f"{daily_agg['Click_to_Install_Rate'].std():.2f}",
                f"{daily_agg['Install_to_OTP_Rate'].std():.2f}",
                f"{daily_agg['Install_to_KYC_Rate'].std():.2f}",
                f"{daily_agg['Install_to_Account_Rate'].std():.2f}",
                f"{combined_df['KYC_to_Account_Rate'].std():.2f}"
            ]
        }
        
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
    
    # ============================================================================
    # TAB 2: iosvs android ANALYSIS
    # ============================================================================
    with tab2:
        st.markdown("## 📱 Platform Comparison: iOS vs Android")
        
        # Check if we have both platforms
        platforms_available = combined_df['Platform'].unique()
        
        if len(platforms_available) < 2:
            st.warning(f"⚠️ Only {platforms_available[0]} data available. Upload both iOS and Android data for comparison.")
            
            # Show single platform analysis
            platform_name = platforms_available[0]
            platform_data = combined_df[combined_df['Platform'] == platform_name]
            
            st.markdown(f"### {platform_name} Analysis")
            
            # Single platform funnel
            st.plotly_chart(
                plot_funnel_chart(platform_data, platform_name),
                use_container_width=True
            )
            
            # Summary stats
            summary = get_platform_summary(combined_df, platform_name)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Installs", f"{summary['Installs']:,}")
            with col2:
                st.metric("KYC Verified", f"{summary['KYC Verified']:,}")
            with col3:
                st.metric("Accounts Created", f"{summary['Accounts']:,}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Install → KYC", f"{summary['Install→KYC %']:.2f}%")
            with col2:
                st.metric("KYC → Account", f"{summary['KYC→Account %']:.2f}%")
            with col3:
                st.metric("Account → Trade", f"{summary['Account→Trade %']:.2f}%")
        
        else:
            # Both platforms available - full comparison
            st.markdown("### 📊 Platform-Wise Conversion Rates")
            
            # Get platform aggregated data
            platform_agg = aggregate_by_platform(combined_df)
            
            # Apply date filter if enabled
            if use_date_filter:
                platform_agg = platform_agg[
                    (platform_agg['Date'] >= pd.Timestamp(start_date)) & 
                    (platform_agg['Date'] <= pd.Timestamp(end_date))
                ]
            
            # Platform selector in sidebar (already in main filters)
            st.sidebar.markdown("#### 🎯 Platform Filter")
            selected_platforms = st.sidebar.multiselect(
                "Select Platforms",
                options=['iOS', 'Android'],
                default=['iOS', 'Android'],
                key='platform_filter'
            )
            
            # Filter data by selected platforms
            if selected_platforms:
                display_platform_agg = platform_agg[platform_agg['Platform'].isin(selected_platforms)]
                display_combined = combined_df[combined_df['Platform'].isin(selected_platforms)]
            else:
                display_platform_agg = platform_agg
                display_combined = combined_df
            
            # iOS Summary
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📱 iOS")
                ios_summary = get_platform_summary(display_combined, 'iOS')
                
                st.metric("INSTALL → KYC", 
                         f"{ios_summary['Install→KYC %']:.2f}%",
                         f"{ios_summary['KYC Verified']:,} users")
                
                st.metric("KYC → ACCOUNT", 
                         f"{ios_summary['KYC→Account %']:.2f}%",
                         f"{ios_summary['Accounts']:,} users")
                
                st.metric("ACCOUNT → TRADE", 
                         f"{ios_summary['Account→Trade %']:.2f}%",
                         f"{ios_summary['Traders']:,} users")
                
                st.markdown("**End-to-End Metrics:**")
                st.metric("INSTALL → ACCOUNT", f"{ios_summary['Install→Account %']:.2f}%")
                st.metric("INSTALL → TRADE", f"{ios_summary['Install→Trade %']:.2f}%")
            
            with col2:
                st.markdown("### 📱 Android")
                android_summary = get_platform_summary(display_combined, 'Android')
                
                st.metric("INSTALL → KYC", 
                         f"{android_summary['Install→KYC %']:.2f}%",
                         f"{android_summary['KYC Verified']:,} users")
                
                st.metric("KYC → ACCOUNT", 
                         f"{android_summary['KYC→Account %']:.2f}%",
                         f"{android_summary['Accounts']:,} users")
                
                st.metric("ACCOUNT → TRADE", 
                         f"{android_summary['Account→Trade %']:.2f}%",
                         f"{android_summary['Traders']:,} users")
                
                st.markdown("**End-to-End Metrics:**")
                st.metric("INSTALL → ACCOUNT", f"{android_summary['Install→Account %']:.2f}%")
                st.metric("INSTALL → TRADE", f"{android_summary['Install→Trade %']:.2f}%")
            
            st.markdown("---")
            
            # Conversion Rate Comparison Chart
            st.markdown("### 📊 Conversion Rate Comparison")
            st.plotly_chart(
                plot_platform_comparison_bars(display_platform_agg),
                use_container_width=True
            )
            
            st.markdown("---")
            
            # Funnel Charts Side by Side
            st.markdown("### 🔄 Funnel Journey")
            
            col1, col2 = st.columns(2)
            
            with col1:
                ios_data = display_combined[display_combined['Platform'] == 'iOS']
                st.plotly_chart(
                    plot_funnel_chart(ios_data, 'iOS'),
                    use_container_width=True
                )
            
            with col2:
                android_data = display_combined[display_combined['Platform'] == 'Android']
                st.plotly_chart(
                    plot_funnel_chart(android_data, 'Android'),
                    use_container_width=True
                )
            
            st.markdown("---")
            
            # Time Series Comparison
            st.markdown("### 📈 Platform Performance Over Time")
            
            comparison_metric = st.selectbox(
                "Select Metric to Compare",
                ['Installs', 'Install_to_KYC_Rate', 'KYC_to_Account_Rate', 
                 'Account_to_Trade_Rate', 'Install_to_Account_Rate'],
                format_func=lambda x: x.replace('_', ' → ').replace('Rate', '%').title()
            )
            
            fig = go.Figure()
            
            for platform in selected_platforms:
                platform_data = display_platform_agg[display_platform_agg['Platform'] == platform]
                
                fig.add_trace(go.Scatter(
                    x=platform_data['Date'],
                    y=platform_data[comparison_metric],
                    mode='lines+markers',
                    name=platform,
                    line=dict(width=3),
                    marker=dict(size=8)
                ))
            
            fig.update_layout(
                title=f"{comparison_metric.replace('_', ' → ').replace('Rate', '%').title()} by Platform",
                xaxis_title="Date",
                yaxis_title=comparison_metric.replace('_', ' → ').replace('Rate', '%').title(),
                height=450,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)

    # ============================================================================
    # TAB 2: CORRELATION ANALYSIS
    # ============================================================================
    with tab3:
        st.markdown("## Correlation Analysis")
        
        st.info("📊 Select 2-4 metrics to analyze their relationships. Asterisks indicate significance: *** p<0.001, ** p<0.01, * p<0.05")
        
        # Metric selection
        available_metrics = {
            'Impressions': 'Impressions',
            'Clicks': 'Clicks',
            'Installs': 'Installs',
            'Sessions': 'Sessions',
            'KYC Verified': 'verify_kyc (Unique users)',
            'Accounts Created': 'account number (Unique users)',
            'OTP Confirmed': 'confirm_mobileotp (Unique users)',
            'CTR %': 'CTR',
            'Click→Install %': 'Click_to_Install_Rate',
            'Install→OTP %': 'Install_to_OTP_Rate',
            'Install→KYC %': 'Install_to_KYC_Rate',
            'Install→Account %': 'Install_to_Account_Rate'
        }
        
        if has_spends:
            available_metrics.update({
                'Spends': 'Spends',
                'CPI': 'CPI',
                'CPC': 'CPC',
                'CPM': 'CPM'
            })
        
        # Use session state to prevent unnecessary recalculations
        if 'selected_corr_metrics' not in st.session_state:
            st.session_state.selected_corr_metrics = []
        
        selected_metrics = st.multiselect(
            "Select Metrics for Correlation Matrix",
            options=list(available_metrics.keys()),
            default=st.session_state.selected_corr_metrics,
            key='correlation_metrics_selector',
            help="💡 Start with 2-4 metrics. Add more gradually for complex analysis."
        )
        
        # Update session state only when changed
        if selected_metrics != st.session_state.selected_corr_metrics:
            st.session_state.selected_corr_metrics = selected_metrics
        
        if len(selected_metrics) >= 2:
            # Show progress for calculations
            with st.spinner('Calculating correlations...'):
                metric_cols = [available_metrics[m] for m in selected_metrics]
                
                # Calculate correlation matrix
                corr_matrix, p_values = calculate_correlation_matrix(daily_agg, metric_cols)
                corr_matrix.columns = selected_metrics
                corr_matrix.index = selected_metrics
                p_values.columns = selected_metrics
                p_values.index = selected_metrics
            
            # Plot heatmap
            st.plotly_chart(
                plot_correlation_heatmap(corr_matrix, p_values, "Correlation Matrix with Significance"),
                use_container_width=True,
                key='correlation_heatmap'
            )
            
            st.markdown("---")
            
            # Scatter plots
            st.markdown("### Scatter Plot Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                x_metric = st.selectbox(
                    "X-axis Metric", 
                    selected_metrics, 
                    key='scatter_x',
                    index=0
                )
            
            with col2:
                y_options = [m for m in selected_metrics if m != x_metric]
                y_metric = st.selectbox(
                    "Y-axis Metric", 
                    y_options,
                    key='scatter_y',
                    index=0 if y_options else 0
                )
            
            if x_metric and y_metric and x_metric != y_metric:
                x_col = available_metrics[x_metric]
                y_col = available_metrics[y_metric]
                
                with st.spinner('Creating scatter plot...'):
                    st.plotly_chart(
                        plot_scatter_with_regression(daily_agg, x_col, y_col, x_metric, y_metric),
                        use_container_width=True,
                        key='scatter_plot'
                    )
                
                # Display correlation stats
                if len(daily_agg[[x_col, y_col]].dropna()) > 2:
                    corr, p_val = pearsonr(daily_agg[x_col].dropna(), daily_agg[y_col].dropna())
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Pearson Correlation", f"{corr:.3f}")
                    with col2:
                        st.metric("P-value", f"{p_val:.4f}")
                    with col3:
                        sig_level = "Highly Significant" if p_val < 0.001 else "Significant" if p_val < 0.05 else "Not Significant"
                        st.metric("Significance", sig_level)
        elif len(selected_metrics) == 1:
            st.info("👆 Please select at least one more metric to see correlations")
        else:
            st.info("👆 Please select at least 2 metrics from the dropdown above to begin correlation analysis")
    
    # ============================================================================
    # TAB 3: LAG ANALYSIS
    # ============================================================================
    with tab4:
        st.markdown("## Lagged Correlation Analysis")
        
        st.info(f"📊 Analyzing correlations with {max_lag_days}-day lag to understand delayed effects of marketing efforts")
        
        if has_spends:
            # Lag analysis for spends vs key metrics
            st.markdown("### Spends Impact on Key Metrics (with Lag)")
            
            target_metrics = {
                'Installs': 'Installs',
                'KYC Verified': 'verify_kyc (Unique users)',
                'Accounts Created': 'account number (Unique users)',
                'Install→KYC %': 'Install_to_KYC_Rate',
                'Install→Account %': 'Install_to_Account_Rate'
            }
            
            selected_target = st.selectbox("Select Target Metric", list(target_metrics.keys()))
            
            if selected_target:
                target_col = target_metrics[selected_target]
                
                # Calculate lag correlation
                lag_results = calculate_correlation_with_lag(
                    daily_agg,
                    'Spends',
                    target_col,
                    max_lag=max_lag_days
                )
                
                # Plot lag correlation
                st.plotly_chart(
                    plot_lag_correlation(lag_results, f"Spends → {selected_target}"),
                    use_container_width=True
                )
                
                st.markdown("---")
                
                # Display lag results table
                st.markdown("### Detailed Lag Results")
                
                # Format the dataframe
                display_df = lag_results.copy()
                display_df['Pearson Correlation'] = display_df['Pearson Correlation'].apply(lambda x: f"{x:.3f}")
                display_df['Pearson p-value'] = display_df['Pearson p-value'].apply(lambda x: f"{x:.4f}")
                display_df['Spearman Correlation'] = display_df['Spearman Correlation'].apply(lambda x: f"{x:.3f}")
                display_df['Spearman p-value'] = display_df['Spearman p-value'].apply(lambda x: f"{x:.4f}")
                
                st.dataframe(display_df, use_container_width=True)
                
                # Find optimal lag
                best_lag = lag_results.loc[lag_results['Pearson Correlation'].abs().idxmax()]
                
                st.success(f"""
                **Optimal Lag: {int(best_lag['Lag (days)'])} days**
                - Pearson Correlation: {best_lag['Pearson Correlation']:.3f}
                - P-value: {best_lag['Pearson p-value']:.4f}
                - Significant: {best_lag['Significant']}
                """)
        else:
            st.warning("Upload spends data to enable lag analysis")
            
            # Alternative: lag between other metrics
            st.markdown("### Custom Lag Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                lead_metric = st.selectbox("Leading Metric", list(available_metrics.keys()), key='lag_lead')
            
            with col2:
                lag_metric = st.selectbox("Lagging Metric", 
                                         [m for m in available_metrics.keys() if m != lead_metric],
                                         key='lag_lag')
            
            if lead_metric and lag_metric:
                lead_col = available_metrics[lead_metric]
                lag_col = available_metrics[lag_metric]
                
                lag_results = calculate_correlation_with_lag(
                    daily_agg,
                    lead_col,
                    lag_col,
                    max_lag=max_lag_days
                )
                
                st.plotly_chart(
                    plot_lag_correlation(lag_results, f"{lead_metric} → {lag_metric}"),
                    use_container_width=True
                )
                
                st.dataframe(lag_results, use_container_width=True)
    
    # ============================================================================
    # TAB 4: PAID VS ORGANIC
    # ============================================================================
    with tab5:
        st.markdown("## Paid vs Organic Analysis")
        
        if source_agg is not None and len(source_agg) > 0:
            # Check if we have both paid and organic data
            source_types = source_agg['Source_Type'].unique()
            
            if len(source_types) > 1:
                # Comparison metrics
                st.markdown("### Performance Comparison")
                
                comparison_metrics = {
                    'Installs': 'Installs',
                    'Install→OTP %': 'Install_to_OTP_Rate',
                    'Install→KYC %': 'Install_to_KYC_Rate',
                    'Install→Account %': 'Install_to_Account_Rate'
                }
                
                selected_comparison = st.selectbox(
                    "Select Metric to Compare",
                    list(comparison_metrics.keys())
                )
                
                comparison_col = comparison_metrics[selected_comparison]
                
                # Plot comparison
                st.plotly_chart(
                    plot_paid_vs_organic_comparison(source_agg, comparison_col, selected_comparison),
                    use_container_width=True
                )
                
                st.markdown("---")
                
                # Summary statistics
                st.markdown("### Summary Statistics")
                
                paid_df = source_agg[source_agg['Source_Type'] == 'Paid']
                organic_df = source_agg[source_agg['Source_Type'] == 'Organic']
                
                stats_data = []
                
                for metric_name, metric_col in comparison_metrics.items():
                    paid_mean = paid_df[metric_col].mean()
                    paid_median = paid_df[metric_col].median()
                    organic_mean = organic_df[metric_col].mean()
                    organic_median = organic_df[metric_col].median()
                    
                    diff_mean = paid_mean - organic_mean
                    diff_pct = ((paid_mean - organic_mean) / organic_mean * 100) if organic_mean > 0 else 0
                    
                    stats_data.append({
                        'Metric': metric_name,
                        'Paid Mean': f"{paid_mean:.2f}",
                        'Organic Mean': f"{organic_mean:.2f}",
                        'Difference': f"{diff_mean:+.2f}",
                        'Difference %': f"{diff_pct:+.1f}%"
                    })
                
                st.dataframe(pd.DataFrame(stats_data), use_container_width=True)
            else:
                st.warning("Both Paid and Organic data required for comparison")
        else:
            st.warning("Source Type information not available in the data")
    
    # ============================================================================
    # TAB 6: LIFT ANALYSIS
    # ============================================================================
    with tab6:
        st.markdown("## 📉 Lift Analysis: Baseline vs Campaign Performance")
        
        st.info("📊 Compare baseline period (non-campaign/organic) vs campaign period to measure true lift")
        
        # Period Selection
        st.markdown("### 📅 Define Analysis Periods")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Baseline Period (Pre-Campaign/Organic)")
            baseline_start = st.date_input(
                "Baseline Start Date",
                value=combined_df['Date'].min(),
                key='baseline_start'
            )
            baseline_end = st.date_input(
                "Baseline End Date",
                value=combined_df['Date'].min() + timedelta(days=14),
                key='baseline_end'
            )
        
        with col2:
            st.markdown("#### Campaign Period")
            campaign_start = st.date_input(
                "Campaign Start Date",
                value=combined_df['Date'].max() - timedelta(days=14),
                key='campaign_start'
            )
            campaign_end = st.date_input(
                "Campaign End Date",
                value=combined_df['Date'].max(),
                key='campaign_end'
            )
        
        # Validate periods
        if baseline_start >= baseline_end:
            st.error("❌ Baseline start date must be before end date")
            return
        
        if campaign_start >= campaign_end:
            st.error("❌ Campaign start date must be before end date")
            return
        
        if baseline_end >= campaign_start:
            st.warning("⚠️ Warning: Baseline and campaign periods overlap")
        
        st.markdown("---")
        
        # Filter data for both periods
        baseline_df = daily_agg[
            (daily_agg['Date'] >= pd.Timestamp(baseline_start)) & 
            (daily_agg['Date'] <= pd.Timestamp(baseline_end))
        ]
        
        campaign_df = daily_agg[
            (daily_agg['Date'] >= pd.Timestamp(campaign_start)) & 
            (daily_agg['Date'] <= pd.Timestamp(campaign_end))
        ]
        
        if len(baseline_df) == 0 or len(campaign_df) == 0:
            st.error("❌ No data available for selected periods. Please adjust date ranges.")
            return
        
        # Metric selection for time series
        st.markdown("### 📊 Time Series Analysis")
        
        analysis_metrics = {
            'Installs': 'Installs',
            'Install → KYC %': 'Install_to_KYC_Rate',
            'Install → Account %': 'Install_to_Account_Rate',
            'KYC → Account %': 'KYC_to_Account_Rate'
        }
        
        selected_ts_metric = st.selectbox(
            "Select Metric for Time Series Visualization",
            list(analysis_metrics.keys())
        )
        
        metric_col = analysis_metrics[selected_ts_metric]
        
        # Plot time series with baseline
        fig, baseline_avg, campaign_avg = plot_time_series_with_baseline(
            daily_agg,
            metric_col,
            selected_ts_metric,
            pd.Timestamp(baseline_start),
            pd.Timestamp(baseline_end),
            pd.Timestamp(campaign_start),
            pd.Timestamp(campaign_end)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate lift
        if baseline_avg > 0:
            lift_pct = ((campaign_avg - baseline_avg) / baseline_avg) * 100
        else:
            lift_pct = 0
        
        # Display lift metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Baseline Average", f"{baseline_avg:.2f}")
        
        with col2:
            st.metric("Campaign Average", f"{campaign_avg:.2f}")
        
        with col3:
            st.metric("Absolute Lift", f"{campaign_avg - baseline_avg:+.2f}")
        
        with col4:
            st.metric("Lift %", f"{lift_pct:+.1f}%", 
                     delta_color="normal" if lift_pct > 0 else "inverse")
        
        st.markdown("---")
        
        # Comprehensive lift comparison for all metrics
        st.markdown("### 📈 Comprehensive Lift Analysis")
        
        lift_analysis = []
        
        for metric_name, metric_col in analysis_metrics.items():
            baseline_avg = baseline_df[metric_col].mean()
            campaign_avg = campaign_df[metric_col].mean()
            
            if baseline_avg > 0:
                lift = ((campaign_avg - baseline_avg) / baseline_avg) * 100
            else:
                lift = 0
            
            # Statistical test
            if len(baseline_df) > 1 and len(campaign_df) > 1:
                t_stat, p_val = stats.ttest_ind(
                    baseline_df[metric_col].dropna(),
                    campaign_df[metric_col].dropna()
                )
            else:
                t_stat, p_val = 0, 1
            
            lift_analysis.append({
                'Metric': metric_name,
                'Baseline Avg': f"{baseline_avg:.2f}",
                'Campaign Avg': f"{campaign_avg:.2f}",
                'Absolute Lift': f"{campaign_avg - baseline_avg:+.2f}",
                'Lift %': f"{lift:+.1f}%",
                'P-value': f"{p_val:.4f}",
                'Significant': 'Yes ✓' if p_val < 0.05 else 'No ✗'
            })
        
        lift_df = pd.DataFrame(lift_analysis)
        
        # Style the dataframe
        def highlight_lift(row):
            lift_val = float(row['Lift %'].strip('%+'))
            if lift_val > 10 and row['Significant'] == 'Yes ✓':
                return ['background-color: #d1fae5'] * len(row)  # Green
            elif lift_val < -10 and row['Significant'] == 'Yes ✓':
                return ['background-color: #fee2e2'] * len(row)  # Red
            else:
                return [''] * len(row)
        
        st.dataframe(
            lift_df.style.apply(highlight_lift, axis=1),
            use_container_width=True
        )
        
        st.markdown("---")
        
        # Lift visualization
        st.markdown("### 📊 Lift Comparison Visualization")
        
        lift_values = [float(x.strip('%+')) for x in lift_df['Lift %']]
        
        fig = go.Figure()
        
        colors = ['#10b981' if l > 0 else '#ef4444' for l in lift_values]
        
        fig.add_trace(go.Bar(
            x=lift_df['Metric'],
            y=lift_values,
            marker_color=colors,
            text=[f"{l:+.1f}%" for l in lift_values],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Lift: %{y:.1f}%<extra></extra>'
        ))
        
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        
        fig.update_layout(
            title="Lift % (Campaign vs Baseline)",
            xaxis_title="Metric",
            yaxis_title="Lift %",
            height=450,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Key insights
        st.markdown("### 💡 Key Insights")
        
        # Find best and worst
        max_lift_idx = lift_values.index(max(lift_values))
        min_lift_idx = lift_values.index(min(lift_values))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success(f"""
            **🎯 Best Improvement**
            - **Metric**: {lift_df.iloc[max_lift_idx]['Metric']}
            - **Baseline**: {lift_df.iloc[max_lift_idx]['Baseline Avg']}
            - **Campaign**: {lift_df.iloc[max_lift_idx]['Campaign Avg']}
            - **Lift**: {lift_df.iloc[max_lift_idx]['Lift %']}
            - **Significant**: {lift_df.iloc[max_lift_idx]['Significant']}
            """)
        
        with col2:
            if lift_values[min_lift_idx] < 0:
                st.error(f"""
                **⚠️ Needs Attention**
                - **Metric**: {lift_df.iloc[min_lift_idx]['Metric']}
                - **Baseline**: {lift_df.iloc[min_lift_idx]['Baseline Avg']}
                - **Campaign**: {lift_df.iloc[min_lift_idx]['Campaign Avg']}
                - **Lift**: {lift_df.iloc[min_lift_idx]['Lift %']}
                - **Significant**: {lift_df.iloc[min_lift_idx]['Significant']}
                """)
            else:
                st.info(f"""
                **📊 Lowest Lift**
                - **Metric**: {lift_df.iloc[min_lift_idx]['Metric']}
                - **Baseline**: {lift_df.iloc[min_lift_idx]['Baseline Avg']}
                - **Campaign**: {lift_df.iloc[min_lift_idx]['Campaign Avg']}
                - **Lift**: {lift_df.iloc[min_lift_idx]['Lift %']}
                - **Significant**: {lift_df.iloc[min_lift_idx]['Significant']}
                """)
        
        # Summary interpretation
        st.markdown("---")
        st.markdown("### 📋 Summary")
        
        significant_lifts = [l for l, s in zip(lift_values, lift_df['Significant']) if s == 'Yes ✓']
        
        if len(significant_lifts) > 0:
            avg_significant_lift = sum(significant_lifts) / len(significant_lifts)
            
            if avg_significant_lift > 10:
                st.success(f"✅ Campaign is performing well with an average lift of **{avg_significant_lift:.1f}%** across significant metrics.")
            elif avg_significant_lift > 0:
                st.info(f"ℹ️ Campaign shows moderate improvement with an average lift of **{avg_significant_lift:.1f}%** across significant metrics.")
            else:
                st.warning(f"⚠️ Campaign performance is below baseline with an average lift of **{avg_significant_lift:.1f}%** across significant metrics.")
        else:
            st.warning("⚠️ No statistically significant differences detected between baseline and campaign periods.")
    
    # ============================================================================
    # TAB 6: DATA EXPORT
    # ============================================================================
    with tab7:
        st.markdown("## 📥 Data Export")
        
        # Export options
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Combined Data")
            csv_combined = combined_df.to_csv(index=False)
            st.download_button(
                label="Download Combined Platform Data",
                data=csv_combined,
                file_name=f"combined_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col2:
            st.markdown("### Daily Aggregated Data")
            csv_daily = daily_agg.to_csv(index=False)
            st.download_button(
                label="Download Daily Aggregated Data",
                data=csv_daily,
                file_name=f"daily_aggregated_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        if source_agg is not None:
            st.markdown("### Source Type Data")
            csv_source = source_agg.to_csv(index=False)
            st.download_button(
                label="Download Paid vs Organic Data",
                data=csv_source,
                file_name=f"paid_vs_organic_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        st.markdown("---")
        
        # Create comprehensive Excel report
        st.markdown("### 📊 Comprehensive Excel Report")
        
        excel_buffer = io.BytesIO()
        
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            # Sheet 1: Summary
            summary_data = pd.DataFrame({
                'Metric': ['Total Impressions', 'Total Clicks', 'Total Installs', 
                          'Total KYC', 'Total Accounts', 'Date Range'],
                'Value': [
                    f"{int(combined_df['Impressions'].sum()):,}",
                    f"{int(combined_df['Clicks'].sum()):,}",
                    f"{int(combined_df['Installs'].sum()):,}",
                    f"{int(combined_df['verify_kyc (Unique users)'].sum()):,}",
                    f"{int(combined_df['account number (Unique users)'].sum()):,}",
                    f"{combined_df['Date'].min().strftime('%Y-%m-%d')} to {combined_df['Date'].max().strftime('%Y-%m-%d')}"
                ]
            })
            summary_data.to_excel(writer, sheet_name='Summary', index=False)
            
            # Sheet 2: Combined Data
            combined_df.to_excel(writer, sheet_name='Combined Data', index=False)
            
            # Sheet 3: Daily Aggregated
            daily_agg.to_excel(writer, sheet_name='Daily Aggregated', index=False)
            
            # Sheet 4: Source Type Comparison (if available)
            if source_agg is not None:
                source_agg.to_excel(writer, sheet_name='Paid vs Organic', index=False)
        
        excel_buffer.seek(0)
        
        st.download_button(
            label="📥 Download Complete Excel Report",
            data=excel_buffer,
            file_name=f"campaign_analytics_report_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    # ============================================================================
    # FOOTER
    # ============================================================================
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #6b7280; padding: 20px;'>
        <p><b>AppsFlyer Campaign Analytics Pro</b><br>
        Advanced Correlation Analysis | Paid vs Organic | Lift Analysis</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
