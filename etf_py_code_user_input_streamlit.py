import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns 
import streamlit as st           
from pathlib import Path

def get_etf_data(etf_tickers, period="ytd"):
    """
    Pulls top holdings, weights, and historical returns for a list of ETFs 
    and their underlying components.
    """
    etf_summary = {}
    all_underlying_tickers = set()
    
    print("Fetching ETF structural data and holdings...")
    for ticker in etf_tickers:
        fund = yf.Ticker(ticker)
        
        # Extract fund holdings data safely
        try:
            holdings_df = fund.funds_data.top_holdings
            if holdings_df is None or holdings_df.empty:
                print(f"Warning: Could not fetch top holdings for {ticker} via yfinance API.")
                continue
            
            # THE FIX: Move the ticker symbol out of the hidden Index and into a normal column
            holdings_df = holdings_df.reset_index()
            
        except Exception as e:
            print(f"Error accessing holdings for {ticker}: {e}")
            continue
            
        # Clean up data frame columns (removing spaces and standardizing casing)
        holdings_df.columns = [str(col).lower().replace(' ', '') for col in holdings_df.columns]
        
        # The first column is now guaranteed to be the ticker, and the last is the weight
        sym_col = holdings_df.columns[0]
        pct_col = holdings_df.columns[-1]
        
        # Store structured holdings data
        holdings = []
        for _, row in holdings_df.iterrows():
            sym = row[sym_col]
            weight = row[pct_col]
            
            # NEW: Extract the company name (Yahoo usually labels this column 'name')
            company_name = row.get('name', 'Unknown')

            # yfinance sometimes returns weights as fractions (0.07) or percentages (7.0)
            if weight > 1.0:
                weight = weight / 100.0
            
            holdings.append({'symbol': sym, 'name': company_name, 'weight': weight})
            all_underlying_tickers.add(sym)
            
        etf_summary[ticker] = holdings

    # Collect all tickers to fetch historical data in one batch call
    all_tickers_to_fetch = list(all_underlying_tickers) + etf_tickers
    print(f"Fetching historical price data for {len(all_tickers_to_fetch)} assets...")
    
    # Fetch historical data
    hist_data = yf.download(all_tickers_to_fetch, period=period, progress=False)['Close']
    
    # Calculate total return over the period selected
    returns = {}
    for col in hist_data.columns:
        # Check if the column has valid data to prevent errors on truly delisted assets
        if hist_data[col].dropna().empty:
            returns[col] = 0
            continue
            
        first_price = hist_data[col].dropna().iloc[0]
        last_price = hist_data[col].dropna().iloc[-1]
        total_return = (last_price - first_price) / first_price
        returns[col] = total_return

    # Build the final comprehensive analysis matrix
    analysis_records = []
    for etf, holdings in etf_summary.items():
        etf_return = returns.get(etf, 0)
        
        for holding in holdings:
            stock_sym = holding['symbol']
            weight = holding['weight']
            stock_return = returns.get(stock_sym, 0)
            
            # Weighted Contribution = How much this specific stock drove the ETF's return
            weighted_contribution = stock_return * weight
            
            analysis_records.append({
                'ETF': etf,
                'ETF Return': etf_return,
                'Asset Symbol': stock_sym,
                'Company Name': holding['name'],
                'Weight in ETF': weight,
                'Asset Return': stock_return,
                'Weighted Contribution': weighted_contribution
            })
            
    return pd.DataFrame(analysis_records)

def generate_concentration_plot(df):
    """
    Aggregates total exposure and returns a Matplotlib Figure for Streamlit.
    """
    total_exposure = df.groupby(['Asset Symbol', 'Company Name'])['Weight in ETF'].sum().reset_index()
    top_holdings = total_exposure.sort_values(by='Weight in ETF', ascending=False).head(15)
    
    # Create the figure and axis objects
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.set_theme(style="whitegrid")
    
    # Draw the plot onto our specific axis (ax)
    sns.barplot(
        x='Weight in ETF', 
        y='Company Name', 
        data=top_holdings, 
        palette="viridis",
        hue='Company Name',
        legend=False,
        ax=ax
    )
    
    # Format the x-axis
    import matplotlib.ticker as mtick
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    
    ax.set_title('Top 15 Heaviest Exposures Across All Selected ETFs', fontsize=16, pad=15)
    ax.set_xlabel('Total Cumulative Weight', fontsize=12)
    ax.set_ylabel('')
    fig.tight_layout()
    
    # Return the figure object back to Streamlit
    return fig

def generate_heatmap_plot(pivot_df):
    """
    Generates a heatmap with dynamic sizing for Streamlit.
    """
    pivot_df['Total Overlap Weight'] = pivot_df.sum(axis=1)
    # We can increase the head() limit now that the chart can scroll
    top_overlaps = pivot_df.sort_values(by='Total Overlap Weight', ascending=False).head(50) 
    top_overlaps = top_overlaps.drop(columns=['Total Overlap Weight'])
    
    # --- NEW: Dynamic Sizing Math ---
    num_rows = len(top_overlaps)
    num_cols = len(top_overlaps.columns)
    
    # Base size of 10x10, but grows if there are more than 20 rows or 10 columns
    fig_width = max(10, num_cols * 0.8)
    fig_height = max(10, num_rows * 0.5)
    
    # Apply the dynamic size
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.set_theme(style="white")
    
    # Draw the heatmap
    sns.heatmap(
        top_overlaps, 
        annot=True,              
        fmt=".2%",               
        cmap="YlOrRd",           
        linewidths=.5, 
        cbar_kws={'label': 'Weight in ETF'},
        ax=ax
    )
    
    # Move the x-axis ticks and label to the top
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    
    ax.set_ylabel('Asset Symbol')
    ax.set_xlabel('ETF')
    fig.tight_layout()
    
    return fig
    
    # --- UI ADJUSTMENTS ---
    # Move the x-axis ticks and label to the top
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    
    # Set the labels (Title has been removed so Streamlit's subheader takes over)
    ax.set_ylabel('Asset Symbol')
    ax.set_xlabel('ETF')
    fig.tight_layout()
    
    # Return the figure object back to Streamlit
    return fig

def Overlap_analysis(df):
    """
    Identifies specific individual assets that appear in multiple ETFs
    and returns a pivot table for export.
    """
    print("\n--- OVERLAP ANALYSIS ---")
    duplicated_assets = df[df.duplicated(subset=['Asset Symbol'], keep=False)]
    
    if duplicated_assets.empty:
        print("No overlapping holdings found in the top tier of these ETFs.")
        return None
        
    # Create the matrix
    pivot = duplicated_assets.pivot_table(
        index='Asset Symbol', 
        columns='ETF', 
        values='Weight in ETF', 
        aggfunc='sum'
    ).fillna(0)
    
    # Format to percentage style strings JUST for the terminal printout
    pivot_formatted = pivot.map(lambda x: f"{x*100:.2f}%" if x > 0 else "-")
    print(pivot_formatted)
    
    # Return the raw numerical data for the CSV export
    return pivot

# --- EXECUTION BLOCK ---
import streamlit as st
from pathlib import Path
# [Keep your existing get_etf_data, plot_portfolio_concentration, etc. functions here]

# --- STREAMLIT WEB UI CONFIGURATION ---
st.set_page_config(page_title="ETF Analyzer", layout="wide")

st.title("📊 ETF Portfolio Concentration & Overlap Analyzer")
st.markdown("Analyze asset concentration and redundant exposures across multiple funds.")

# --- SIDEBAR CONTROLS (Replacing Terminal Inputs) ---
st.sidebar.header("Analysis Settings")

ticker_input = st.sidebar.text_input(
    "1. Enter ETF Tickers (comma-separated)", 
    value="VOO, VGT, SCHD"
)

period_options = {
    "1 Month": "1mo", "3 Months": "3mo", "6 Months": "6mo", 
    "Year-to-Date": "ytd", "1 Year": "1y", "3 Years": "3y", "5 Years": "5y"
}
selected_label = st.sidebar.selectbox("2. Select Timeframe", list(period_options.keys()), index=4)
selected_period = period_options[selected_label]

# --- RUN BUTTON ---
if st.sidebar.button("Run Analysis", type="primary"):
    target_etfs = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]
    
    if target_etfs:
        with st.spinner("Fetching data from Yahoo Finance..."):
            # Execute your existing pipeline
            analysis_df = get_etf_data(target_etfs, period=selected_period)
            
        if not analysis_df.empty:
            # Create web layout columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Holdings Data Analysis")
                # Render an interactive spreadsheet on the web page
                st.dataframe(analysis_df, use_container_width=True)
                
                # Provide a native browser download button for the CSV
                csv_data = analysis_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Holdings CSV", csv_data, "etf_holdings.csv", "text/csv")

            with col2:
                st.subheader("Portfolio Concentration")
                # Generate and display the plot directly in the UI layout
                fig1 = generate_concentration_plot(analysis_df) # Modified to return fig
                st.pyplot(fig1)
                
            # Render overlap analysis below
            overlap_matrix = Overlap_analysis(analysis_df)
            if overlap_matrix is not None:
                st.markdown("---")
                st.subheader("ETF Overlap Heatmap")
                fig2 = generate_heatmap_plot(overlap_matrix) # Modified to return fig
                # --- NEW: Add the use_container_width flag ---
                st.pyplot(fig2, use_container_width=False)
                
    else:
        st.error("Please enter at least one valid ETF ticker.")