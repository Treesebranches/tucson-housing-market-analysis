import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tucson New Home Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=DM+Serif+Display&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #1a237e 0%, #283593 60%, #3949ab 100%);
        padding: 2.5rem 2rem 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
    }
    .main-header h1 {
        font-family: 'DM Serif Display', serif;
        font-size: 2.4rem;
        margin: 0 0 0.4rem 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        font-size: 1rem;
        opacity: 0.85;
        margin: 0;
    }

    /* Metric cards */
    .metric-card {
        background: white;
        border: 1px solid #e8eaf6;
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(26,35,126,0.07);
    }
    .metric-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: #5c6bc0;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 0.3rem;
    }
    .metric-value {
        font-size: 1.9rem;
        font-weight: 700;
        color: #1a237e;
        line-height: 1.1;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #78909c;
        margin-top: 0.2rem;
    }

    /* Prediction result */
    .prediction-box {
        background: linear-gradient(135deg, #e8eaf6 0%, #c5cae9 100%);
        border: 2px solid #3949ab;
        border-radius: 12px;
        padding: 1.8rem;
        text-align: center;
        margin: 1rem 0;
    }
    .prediction-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #3949ab;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    .prediction-price {
        font-family: 'DM Serif Display', serif;
        font-size: 3rem;
        font-weight: 700;
        color: #1a237e;
        line-height: 1;
    }
    .prediction-delta {
        font-size: 1rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    .delta-positive { color: #2e7d32; }
    .delta-negative { color: #c62828; }

    /* Section headers */
    .section-header {
        font-size: 1rem;
        font-weight: 700;
        color: #1a237e;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        border-bottom: 2px solid #e8eaf6;
        padding-bottom: 0.4rem;
        margin: 1.2rem 0 0.8rem 0;
    }

    /* Insight box */
    .insight-box {
        background: #f3f4fd;
        border-left: 4px solid #3949ab;
        border-radius: 0 8px 8px 0;
        padding: 0.9rem 1.1rem;
        margin: 0.6rem 0;
        font-size: 0.88rem;
        color: #37474f;
        line-height: 1.5;
    }

    /* Submarket badge */
    .badge-green { 
        background:#e8f5e9; color:#2e7d32; 
        padding:3px 10px; border-radius:20px; 
        font-size:0.78rem; font-weight:600;
    }
    .badge-orange { 
        background:#fff3e0; color:#e65100; 
        padding:3px 10px; border-radius:20px; 
        font-size:0.78rem; font-weight:600;
    }
    .badge-red { 
        background:#ffebee; color:#b71c1c; 
        padding:3px 10px; border-radius:20px; 
        font-size:0.78rem; font-weight:600;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
</style>
""", unsafe_allow_html=True)


# ── Load models ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    """Load saved models and feature configs."""
    try:
        ridge     = joblib.load('models/new_home_ridge.pkl')
        scaler    = joblib.load('models/new_home_scaler.pkl')
        feat_cfg  = joblib.load('models/new_home_feature_config.pkl')
        return ridge, scaler, feat_cfg, None
    except FileNotFoundError as e:
        return None, None, None, str(e)

ridge, scaler, feat_cfg, load_error = load_models()


# ── Constants ───────────────────────────────────────────────────────────────────
NUMERIC_FEATURES = [
    'approx_sqft', 'year_built', 'bedrooms', 'baths_total', 'stories',
    'approx_lot_sqft', 'geo_lat', 'geo_lon', 'hoa_monthly', 'seller_concessions'
]
CATEGORICAL_FEATURES = [
    'builder_name', 'subdivision',
    'kitchen_featurescountertops',
    'association__feeshoa_payment_frequency'
]
BOOLEAN_FEATURES = [
    'has_pool', 'has_spa', 'has_casita', 'has_solar', 'mountain_view',
    'golf_course', 'has_den', 'updated_kitchen', 'gated', 'new_hvac'
]

# Submarket data from analysis
SUBMARKET_DATA = {
    'Rocking K South Neighborhood 3 Parcel DD': {
        'score': 73.9, 'tier': 'Prioritize', 'dom': 6,
        'median_price': 420000, 'price_sqft': 220.94,
        'lat': 32.0915, 'lon': -110.6942,
        'hoa': 84.5, 'concessions': 7000,
        'note': 'Fastest absorption (6-day DOM). Recommended submarket.'
    },
    'Rocking K South Neighborhood 4 Parcel L': {
        'score': 60.2, 'tier': 'Monitor', 'dom': 118,
        'median_price': 435830, 'price_sqft': 214.55,
        'lat': 32.117036, 'lon': -110.717452,
        'hoa': 84.5, 'concessions': 10788,
        'note': 'High volume but long DOM (118 days) and high concessions.'
    },
    'Rocking K South Neighborhood 2': {
        'score': 48.2, 'tier': 'Monitor', 'dom': 162,
        'median_price': 441790, 'price_sqft': 230.87,
        'lat': 32.104734, 'lon': -110.701507,
        'hoa': 305.0, 'concessions': 0,
        'note': 'Best pricing power but very slow absorption (162-day DOM).'
    },
    'Rocking K South Neighborhood 1 Parcel A-1': {
        'score': 32.3, 'tier': 'Avoid', 'dom': 58,
        'median_price': 448940, 'price_sqft': 202.06,
        'lat': 32.099764, 'lon': -110.718420,
        'hoa': 84.5, 'concessions': 7000,
        'note': 'Demand not yet established in this parcel.'
    },
    'Rocking K South Neighborhood 3 Parcel BB': {
        'score': 29.7, 'tier': 'Avoid', 'dom': 81,
        'median_price': 373538, 'price_sqft': 199.11,
        'lat': 32.096254, 'lon': -110.693939,
        'hoa': 84.5, 'concessions': 7437,
        'note': 'Low pricing power and slow absorption.'
    },
    'Rocking K South Neighborhood 1 Parcel B-1': {
        'score': 25.6, 'tier': 'Avoid', 'dom': 63,
        'median_price': 557495, 'price_sqft': 218.11,
        'lat': 32.107783, 'lon': -110.716015,
        'hoa': 55.0, 'concessions': 10000,
        'note': 'Lowest composite score — high price but very thin volume.'
    },
}

BUILDER_OPTIONS = [
    'Lennar', 'Meritage Homes', 'Mattamy',
    'Del Webb/Pulte Homes', 'Richmond American Homes', 'Other'
]
COUNTERTOP_OPTIONS = ['Quartz', 'Quartz/Granite', 'Granite', 'Silestone', 'Unknown']
HOA_FREQ_OPTIONS  = ['Monthly', 'Quarterly', 'Semi-Annually', 'Unknown']
TIER_BADGE = {
    'Prioritize': 'badge-green',
    'Monitor':    'badge-orange',
    'Avoid':      'badge-red',
}


# ── Prediction function ─────────────────────────────────────────────────────────
def predict_price(spec: dict) -> float | None:
    if ridge is None:
        return None
    encoded_cols = feat_cfg['encoded_columns']

    row_num  = pd.DataFrame([{k: spec[k] for k in NUMERIC_FEATURES}])
    row_bool = pd.DataFrame([{k: int(spec[k]) for k in BOOLEAN_FEATURES}])
    row_cat  = pd.DataFrame([{k: spec[k] for k in CATEGORICAL_FEATURES}])

    row_encoded = pd.get_dummies(row_cat)
    cat_cols    = [c for c in encoded_cols
                   if any(c.startswith(cat) for cat in CATEGORICAL_FEATURES)]
    row_encoded = row_encoded.reindex(columns=cat_cols, fill_value=0)

    X = pd.concat([
        row_num.reset_index(drop=True),
        row_bool.reset_index(drop=True),
        row_encoded.reset_index(drop=True)
    ], axis=1).reindex(columns=encoded_cols, fill_value=0)

    X_scaled = X.copy()
    X_scaled[NUMERIC_FEATURES] = scaler.transform(X[NUMERIC_FEATURES])

    return float(np.exp(ridge.predict(X_scaled)[0]))


# ── Waterfall chart ─────────────────────────────────────────────────────────────
def make_waterfall(baseline: float, predicted: float, spec: dict, sub_data: dict):
    sqft_diff  = spec['approx_sqft'] - 1815
    bath_diff  = spec['baths_total'] - 2.0
    bed_diff   = spec['bedrooms'] - 4
    story_diff = spec['stories'] - 1

    sqft_impact  =  (np.exp(0.097835 * sqft_diff / 253.7) - 1) * predicted
    bath_impact  =  (np.exp(0.020082 * bath_diff) - 1) * predicted if bath_diff != 0 else 0
    bed_impact   =  (np.exp(-0.017196 * bed_diff) - 1) * predicted if bed_diff != 0 else 0
    story_impact =  (np.exp(-0.021648 * story_diff) - 1) * predicted if story_diff != 0 else 0
    quartz_impact = (np.exp(0.014491) - 1) * predicted \
                    if 'Quartz' in spec.get('kitchen_featurescountertops', '') else 0

    spec_total  = sqft_impact + bath_impact + bed_impact + story_impact + quartz_impact
    other       = predicted - baseline - spec_total

    items = [('Current\nMedian', baseline, False, '#90CAF9')]
    if sqft_diff  != 0: items.append((f'Sqft\n({sqft_diff:+.0f})',  sqft_impact,  True, '#4CAF50' if sqft_impact  >= 0 else '#EF9A9A'))
    if bath_diff  != 0: items.append((f'Baths\n({bath_diff:+.1f})', bath_impact,  True, '#4CAF50' if bath_impact  >= 0 else '#EF9A9A'))
    if bed_diff   != 0: items.append((f'Beds\n({bed_diff:+.0f})',   bed_impact,   True, '#4CAF50' if bed_impact   >= 0 else '#FF9800'))
    if story_diff != 0: items.append((f'Stories\n({story_diff:+.0f})', story_impact, True, '#4CAF50' if story_impact >= 0 else '#FF9800'))
    if quartz_impact != 0: items.append(('Quartz/\nGranite', quartz_impact, True, '#4CAF50'))
    if abs(other) > 500: items.append(('Location\n& Other', other, True, '#9C27B0'))
    items.append(('Predicted\nPrice', predicted, False, '#1565C0'))

    fig, ax = plt.subplots(figsize=(max(8, len(items) * 1.3), 4.5))
    fig.patch.set_facecolor('#f8f9ff')
    ax.set_facecolor('#f8f9ff')

    running = baseline
    for i, (label, val, is_change, color) in enumerate(items):
        if not is_change:
            ax.bar(i, val, color=color, alpha=0.85, width=0.6,
                   edgecolor='white', linewidth=1.5)
            ax.text(i, val + (predicted - baseline) * 0.02,
                    f'${val:,.0f}', ha='center', va='bottom',
                    fontsize=8.5, fontweight='bold', color='#1a237e')
        else:
            bottom = min(running, running + val)
            ax.bar(i, abs(val), bottom=bottom, color=color,
                   alpha=0.85, width=0.6, edgecolor='white', linewidth=1.5)
            sign = '+' if val >= 0 else ''
            ax.text(i, max(running, running + val) + (predicted - baseline) * 0.02,
                    f'{sign}${val:,.0f}', ha='center', va='bottom',
                    fontsize=8, color='#37474f')
            running += val

    ax.set_xticks(range(len(items)))
    ax.set_xticklabels([it[0] for it in items], fontsize=8.5)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    price_range = predicted - baseline
    ax.set_ylim(min(baseline, predicted) - abs(price_range) * 0.15,
                max(baseline, predicted) + abs(price_range) * 0.35)
    ax.set_title('Price Build-Up: Market Median → Your Spec',
                 fontsize=11, fontweight='bold', color='#1a237e', pad=10)
    ax.grid(True, alpha=0.25, axis='y', color='#c5cae9')
    ax.spines[['top', 'right', 'left']].set_visible(False)

    legend_els = [
        Patch(facecolor='#4CAF50', alpha=0.85, label='Positive impact'),
        Patch(facecolor='#FF9800', alpha=0.85, label='Negative avoided'),
        Patch(facecolor='#9C27B0', alpha=0.85, label='Location/other'),
        Patch(facecolor='#1565C0', alpha=0.85, label='Final price'),
    ]
    ax.legend(handles=legend_els, fontsize=7.5, loc='upper left',
              framealpha=0.9, edgecolor='#e8eaf6')
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div class="main-header">
    <h1>🏠 Tucson New Home Price Predictor</h1>
    <p>Ridge regression model trained on Rocking K South closed sales (Dec 2025 – May 2026) &nbsp;·&nbsp; 
       R² = 0.793 &nbsp;·&nbsp; RMSE $27,253</p>
</div>
""", unsafe_allow_html=True)

# Model load error
if load_error:
    st.error(f"""
    **Model files not found.** Make sure the `models/` folder is in the same 
    directory as `app.py` and contains:
    - `new_home_ridge.pkl`
    - `new_home_scaler.pkl`  
    - `new_home_feature_config.pkl`
    
    *Error: {load_error}*
    """)
    st.stop()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Configure Your Spec")
    st.markdown("Adjust inputs below to predict sale price.")

    st.markdown('<div class="section-header">Location</div>', unsafe_allow_html=True)
    subdivision = st.selectbox(
        "Submarket",
        options=list(SUBMARKET_DATA.keys()),
        format_func=lambda x: x.replace('Rocking K South Neighborhood ', 'RKS Nbhd ')
                               .replace('Parcel ', 'Pcl '),
        index=0
    )
    sub_data = SUBMARKET_DATA[subdivision]

    st.markdown('<div class="section-header">Home Spec</div>', unsafe_allow_html=True)
    approx_sqft   = st.slider("Square Footage", 1200, 3500, 2100, 50)
    bedrooms      = st.slider("Bedrooms", 2, 5, 3)
    baths_total   = st.select_slider("Bathrooms", options=[2.0, 2.5, 3.0, 3.5, 4.0], value=2.5)
    stories       = st.radio("Stories", [1, 2], horizontal=True)
    approx_lot_sqft = st.slider("Lot Size (sqft)", 3000, 12000, 6000, 250)

    st.markdown('<div class="section-header">Financials</div>', unsafe_allow_html=True)
    hoa_monthly        = st.number_input("HOA (monthly $)", 0, 500,
                                          int(sub_data['hoa']), 5)
    seller_concessions = st.number_input("Seller Concessions ($)", 0, 40000,
                                          int(sub_data['concessions']), 500)

    st.markdown('<div class="section-header">Builder & Finishes</div>', unsafe_allow_html=True)
    builder_name    = st.selectbox("Builder", BUILDER_OPTIONS, index=0)
    countertops     = st.selectbox("Countertops", COUNTERTOP_OPTIONS, index=1)
    hoa_freq        = st.selectbox("HOA Payment Frequency", HOA_FREQ_OPTIONS, index=0)

    st.markdown('<div class="section-header">Amenities</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        has_pool    = st.checkbox("Pool")
        has_casita  = st.checkbox("Casita")
        has_solar   = st.checkbox("Solar")
        has_den     = st.checkbox("Den/Office")
        gated       = st.checkbox("Gated")
    with col_b:
        has_spa         = st.checkbox("Spa")
        mountain_view   = st.checkbox("Mountain View")
        golf_course     = st.checkbox("Golf Course")
        updated_kitchen = st.checkbox("Updated Kitchen")
        new_hvac        = st.checkbox("New HVAC")


# ── Build spec dict & predict ────────────────────────────────────────────────
spec = {
    'approx_sqft':        approx_sqft,
    'year_built':         2026,
    'bedrooms':           bedrooms,
    'baths_total':        baths_total,
    'stories':            stories,
    'approx_lot_sqft':    approx_lot_sqft,
    'geo_lat':            sub_data['lat'],
    'geo_lon':            sub_data['lon'],
    'hoa_monthly':        hoa_monthly,
    'seller_concessions': seller_concessions,
    'builder_name':       builder_name,
    'subdivision':        subdivision,
    'kitchen_featurescountertops':              countertops,
    'association__feeshoa_payment_frequency':   hoa_freq,
    'has_pool':    has_pool,  'has_spa':      has_spa,
    'has_casita':  has_casita, 'has_solar':   has_solar,
    'mountain_view': mountain_view, 'golf_course': golf_course,
    'has_den':     has_den,   'updated_kitchen': updated_kitchen,
    'gated':       gated,     'new_hvac':     new_hvac,
}

predicted = predict_price(spec)
baseline  = sub_data['median_price']
delta     = predicted - baseline
delta_pct = delta / baseline * 100


# ── Main content ─────────────────────────────────────────────────────────────
col1, col2 = st.columns([1.1, 1.9], gap="large")

with col1:
    # Predicted price
    delta_class = 'delta-positive' if delta >= 0 else 'delta-negative'
    delta_sign  = '+' if delta >= 0 else ''
    st.markdown(f"""
    <div class="prediction-box">
        <div class="prediction-label">Predicted Sale Price</div>
        <div class="prediction-price">${predicted:,.0f}</div>
        <div class="prediction-delta {delta_class}">
            {delta_sign}${delta:,.0f} ({delta_sign}{delta_pct:.1f}%) vs. submarket median
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Submarket scorecard
    tier       = sub_data['tier']
    badge_cls  = TIER_BADGE[tier]
    short_name = subdivision.replace('Rocking K South Neighborhood ', 'RKS Nbhd ')\
                             .replace('Parcel ', 'Pcl ')

    st.markdown('<div class="section-header">Submarket</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <b>{short_name}</b> &nbsp;
    <span class="{badge_cls}">{tier}</span>
    """, unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Score</div>
            <div class="metric-value">{sub_data['score']}</div>
            <div class="metric-sub">out of 100</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Median DOM</div>
            <div class="metric-value">{sub_data['dom']}</div>
            <div class="metric-sub">days on market</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">$/SqFt</div>
            <div class="metric-value">${sub_data['price_sqft']:.0f}</div>
            <div class="metric-sub">market median</div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f'<div class="insight-box">💡 {sub_data["note"]}</div>',
                unsafe_allow_html=True)

    # Key spec insights
    st.markdown('<div class="section-header">Spec Insights</div>',
                unsafe_allow_html=True)

    insights = []
    if approx_sqft > 1815:
        lift = (np.exp(0.097835 * (approx_sqft - 1815) / 253.7) - 1) * predicted
        insights.append(f"📐 +{approx_sqft - 1815} sqft above median → ~<b>${lift:,.0f}</b> price lift")
    if baths_total > 2.0:
        lift = (np.exp(0.020082 * (baths_total - 2.0)) - 1) * predicted
        insights.append(f"🚿 +{baths_total - 2.0:.1f} bath above median → ~<b>${lift:,.0f}</b> lift")
    if bedrooms < 4:
        drag = (1 - np.exp(-0.017196 * (4 - bedrooms))) * predicted
        insights.append(f"🛏️ {bedrooms} bed (vs. 4 typical) → avoids ~<b>${drag:,.0f}</b> price drag")
    if stories == 1:
        drag = (1 - np.exp(-0.021648)) * predicted
        insights.append(f"🏠 Single-story → avoids ~<b>${drag:,.0f}</b> price drag")
    if 'Quartz' in countertops:
        lift = (np.exp(0.014491) - 1) * predicted
        insights.append(f"✨ {countertops} countertops → ~<b>${lift:,.0f}</b> lift")
    if not insights:
        insights.append("Adjust the spec to see model-driven insights here.")

    for ins in insights:
        st.markdown(f'<div class="insight-box">{ins}</div>',
                    unsafe_allow_html=True)

with col2:
    # Waterfall chart
    fig = make_waterfall(baseline, predicted, spec, sub_data)
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # Submarket comparison table
    st.markdown('<div class="section-header">All Submarket Scores</div>',
                unsafe_allow_html=True)

    rows = []
    for name, d in SUBMARKET_DATA.items():
        short = name.replace('Rocking K South Neighborhood ', 'RKS Nbhd ')\
                    .replace('Parcel ', 'Pcl ')
        rows.append({
            'Submarket':      short,
            'Score':          d['score'],
            'Tier':           d['tier'],
            'Median Price':   f"${d['median_price']:,.0f}",
            '$/SqFt':         f"${d['price_sqft']:.0f}",
            'Median DOM':     d['dom'],
        })

    df_table = pd.DataFrame(rows).sort_values('Score', ascending=False)\
                                  .reset_index(drop=True)

    def color_tier(val):
        colors = {
            'Prioritize': 'background-color:#e8f5e9; color:#2e7d32; font-weight:600',
            'Monitor':    'background-color:#fff3e0; color:#e65100; font-weight:600',
            'Avoid':      'background-color:#ffebee; color:#b71c1c; font-weight:600',
        }
        return colors.get(val, '')

    styled = df_table.style.applymap(color_tier, subset=['Tier'])\
                           .highlight_between(subset=['Score'],
                                              left=60, right=100,
                                              color='#e8f5e9')\
                           .format({'Score': '{:.1f}'})
    st.dataframe(styled, use_container_width=True, hide_index=True)

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#90a4ae; font-size:0.8rem; padding:0.5rem'>
    Ridge Regression · Trained on Rocking K South closed sales Dec 2025 – May 2026 · 
    R² = 0.793 · RMSE $27,253 · For analytical purposes only
</div>
""", unsafe_allow_html=True)
