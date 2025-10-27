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

def get_platform_summary(df, platform_name):
    """Get summary statistics for a specific platform"""
    platform_df = df[df['Platform'] == platform_name]
    
    total_installs = platform_df['Installs'].sum()
    total_kyc =
