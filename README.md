# ROI-Lens: Marketing Attribution & Budget Optimization

A data-driven analytics pipeline that uncovers the true ROI of marketing channels using multi-touch attribution models, replacing the flawed last-click baseline with probabilistic and game-theoretic approaches.

Built for **Nexus Consumer Brands** — analyzing 10 brands across 5 marketing channels to optimally reallocate a Rs.100 Crore budget.

## 🎯 Problem

Most marketing teams rely on **last-click attribution**, which gives 100% credit to the final channel before a purchase. This systematically:

- **Over-credits** "closer" channels (e.g., Marketplace, Google Search) that capture demand
- **Under-credits** "primer" channels (e.g., YouTube, Influencer Blog) that create demand
- Leads to **misallocated budgets** and missed conversion opportunities

ROI-Lens fixes this by implementing advanced attribution models and budget optimization.

## 📊 Data

The pipeline processes three datasets covering Q1 2026 activity:

| Dataset | Rows | Description |
|---------|------|-------------|
| `touchpoints.csv` | ~566K | User journey events — impressions, clicks, add-to-carts, purchases |
| `user_profiles.csv` | ~100K | User persona intelligence — segment, trend affinity, geography |
| `campaign_spend.csv` | ~50 | Campaign financial layer — channel, pricing model (CPC/CPM), cost rates |

**Scope**: 10 brands (B01–B10) · 5 channels (Google Search, Instagram, YouTube, Influencer Blog, Marketplace) · 4-stage funnel (Impression → Click → Add-to-Cart → Purchase)

## ⚙️ Pipeline Architecture

The analysis runs through 6 sequential phases:

```
Phase 1          Phase 2              Phase 3                Phase 4           Phase 5            Phase 6
Data Loading  →  Funnel Analysis   →  Multi-Touch         →  Financial      →  Budget          →  Visualization
& Cleaning       & Last-Click         Attribution            Layer             Optimization       & Reporting
                 Baseline             (Markov + Shapley)     (True CPA)        (SLSQP)
```

### Phase 1 — Data Loading & Cleaning
**`src/data_loader.py`** · **`src/data_cleaner.py`**

- Loads, parses, and merges all three CSV datasets into a unified DataFrame
- **Bot detection engine** using 4 signals:
  - Inhuman frequency (>20 events/hour)
  - Impression-only bots (50+ impressions, zero clicks)
  - Timestamp clustering (<2s gaps between events)
  - 24/7 activity (active across 20+ hours/day)
- Removes exact duplicates, validates timestamp ranges, flags orphan purchase events

### Phase 2 — Funnel Analysis & Last-Click Attribution
**`src/funnel_analysis.py`**

- Builds conversion funnels per Brand × Channel (Impressions → Clicks → Add-to-Cart → Purchases)
- Computes **last-click attribution** — the industry-standard (but flawed) baseline
- Calculates CPA (Cost Per Acquisition) under the last-click model

### Phase 3 — Multi-Touch Attribution
**`src/attribution.py`**

- **Markov Chain Attribution**: Builds transition probability matrices from user journey paths, computes channel "removal effects" using absorption probability theory
- **Shapley Value Attribution**: Game-theoretic fair-credit allocation evaluating marginal contributions across all 2^5 = 32 channel coalitions per brand
- **Channel Role Classification**: Labels each channel as Primer/Introducer, Influencer/Assist, or Closer/Converter based on positional frequency
- Side-by-side comparison of all three models with delta analysis

### Phase 4 — Financial Layer
**`src/financials.py`**

- Verifies actual campaign costs against allocated budgets (CPC vs CPM)
- Computes **true CPA** using Markov and Shapley attribution weights
- Fits **ad fatigue / saturation curves** (log response functions) to detect diminishing returns per Brand × Channel

### Phase 5 — Budget Optimization
**`src/optimizer.py`**

- **Constrained optimization** using scipy SLSQP to maximize total conversions
- Budget: Rs.10 Crore per brand (Rs.100 Crore total)
- Constraints: 5% minimum (viability floor) and 50% maximum (diversification cap) per channel
- **Sensitivity analysis** across ±20% and ±40% budget scenarios
- Outputs reallocation recommendations with expected conversion lift

### Phase 6 — Visualization
**`src/visualizations.py`**

Generates 10 publication-quality charts:

| # | Chart | What It Shows |
|---|-------|---------------|
| 1 | Bot Detection | Bot traffic impact per channel |
| 2 | Conversion Funnel | Impression-to-purchase drop-off |
| 3 | Brand Conversions | Purchases by brand (horizontal bar) |
| 4 | Attribution Comparison | Last-Click vs Markov vs Shapley (grouped bar) |
| 5 | Attribution Heatmap | Markov vs Last-Click delta per Brand × Channel |
| 6 | Channel Roles | Primer / Influencer / Closer classification grid |
| 7 | CPA Comparison | Last-Click CPA vs True (Markov) CPA |
| 8 | Budget Reallocation | Recommended spend shifts by channel |
| 9 | Conversion Lift | Expected lift per brand after optimization |
| 10 | Sensitivity Analysis | Budget vs conversions with diminishing returns |

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/your-username/ROI-Lens.git
cd ROI-Lens
pip install -r requirements.txt
```

### Running the Pipeline

**Option 1 — Jupyter Notebook** (recommended for exploration):

```bash
jupyter notebook notebooks/roi_lens_analysis.ipynb
```

**Option 2 — Run individual phases from the command line**:

```bash
# Phase 1: Load and clean data
python -m src.data_cleaner

# Phase 2: Funnel analysis & last-click baseline
python -m src.funnel_analysis

# Phase 3: Multi-touch attribution (Markov + Shapley)
python -m src.attribution

# Phase 4: Financial layer (true CPA + saturation curves)
python -m src.financials

# Phase 5: Budget optimization
python -m src.optimizer

# Phase 6: Generate all charts
python -m src.visualizations
```

Each phase depends on the previous ones. Running a later phase standalone will automatically execute all prerequisite phases.

## 📁 Project Structure

```
ROI-Lens/
├── data/
│   ├── touchpoints.csv          # User journey events (~566K rows)
│   ├── user_profiles.csv        # User personas (~100K rows)
│   └── campaign_spend.csv       # Campaign budgets (~50 rows)
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # Phase 1A: Data loading & parsing
│   ├── data_cleaner.py          # Phase 1B: Bot detection & data quality
│   ├── funnel_analysis.py       # Phase 2: Funnels & last-click attribution
│   ├── attribution.py           # Phase 3: Markov & Shapley attribution
│   ├── financials.py            # Phase 4: True CPA & ad fatigue curves
│   ├── optimizer.py             # Phase 5: Constrained budget optimization
│   └── visualizations.py        # Phase 6: Chart generation
├── notebooks/
│   ├── roi_lens_analysis.ipynb          # Main analysis notebook
│   └── executed_roi_lens.ipynb          # Pre-executed notebook with outputs
├── outputs/
│   ├── figures/                 # Generated charts (PNG)
│   └── results/                 # CSV outputs from each phase
├── requirements.txt
└── README.md
```

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| Data Processing | Pandas, NumPy |
| Statistical Modeling | SciPy (curve fitting, SLSQP optimization) |
| Visualization | Matplotlib, Seaborn, Plotly |
| Notebooks | Jupyter |
| Export | python-pptx (slides), openpyxl (Excel) |

## 📈 Key Results

| Metric | Value |
|--------|-------|
| Bot users detected | ~1,371 users (~24% of all events) |
| Attribution shift (avg) | ±5–15% credit reallocation per channel |
| Expected conversion lift | **+17.4%** with optimized budget allocation |
| Total budget optimized | Rs.100 Crore across 10 brands |

## 💡 Key Insight

Last-click attribution systematically over-credits "closer" channels and under-credits "primer" channels that initiate customer journeys. By switching to Markov/Shapley attribution and reallocating budgets accordingly, the same Rs.100 Crore spend is projected to deliver **~17% more conversions**.


**Built for smarter marketing spend decisions 📊**
