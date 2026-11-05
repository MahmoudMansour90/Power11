"""
data_prep_availability.py -- Availability / Capacity pillar (PSCI), v2
=============================================================================
Converts the PSCI notebook's own outputs (PSCI_Panel_Pipeline_FIXED_3.ipynb,
CELL 12's composite export) into the compact parquet snapshot the Streamlit
app actually loads. Mirrors data_prep.py's Cost/Reliability sections exactly:
the notebook computes everything; this script never recomputes anything the
notebook already computed, with two documented exceptions: the national
comparison stat (mirrors Cost/Reliability's identical pattern) and the
Connecticut defensive suppression below (a data-quality guard, not a
methodology choice).

Inputs (place alongside this script, same names the PSCI notebook exports):
  psci_composite_panel.csv   (2016-2025 historical, 31,090 rows)
  psci_composite_proj.csv    (2026-2028 projected, 9,327 rows)

PSCI methodology, confirmed directly against the notebook's CELL 12 source
and independently re-derived from the raw CSVs (not just taken on the
handoff's word) -- unchanged from v1, see git history for the original
verification:
  - S1_score = pooled percentile rank of Effective_MW (generation capacity)
  - S2_inv_score = 1 - pooled percentile rank of County_Peak_MW (demand) --
    inverted because LOWER relative peak demand = MORE spare capacity
  - S3_score = pooled percentile rank of TCS_raw (voltage-weighted
    transmission circuit miles, HIFLD 2025 snapshot, time-invariant)
  - PSCI_score = (S1_score + S2_inv_score + S3_score) / 3 * 100

RISK FRAMING (locked, consistent with Cost and Reliability's own "higher
score = worse" convention):
  Risk_Score = 100 - PSCI_score            (0-100, higher = worse)
  Risk_Tier mapping (PSCI tier -> Risk tier, locked 1:1 inverse):
    PSCI Tier 1 Very High  -> Risk Tier 5 Very Low Risk
    PSCI Tier 2 High       -> Risk Tier 4 Low Risk
    PSCI Tier 3 Moderate   -> Risk Tier 3 Moderate Risk
    PSCI Tier 4 Low        -> Risk Tier 2 High Risk
    PSCI Tier 5 Very Low   -> Risk Tier 1 Very High Risk

v2 CHANGES FROM v1:

1. FORECAST UNCERTAINTY BAND -- NEW, wired in (previously absent by design).
   The notebook now computes PSCI_lower/PSCI_upper for 2026-2028 (CELL 12,
   Step 5): S1 and S3 are deterministic/time-invariant (no CI exists for
   them), so the only modeled uncertainty is S2 -- the notebook re-ranks
   County_Peak_MW_Lower/Upper through the SAME pooled percentile curve used
   for the point estimate, holding S1_score/S3_score fixed, and explicitly
   labels this "S2 demand uncertainty (95% PI)" rather than a composite CI.
   This is methodologically sound (the point estimate's own derivation is
   held fixed; only the already-modeled demand uncertainty is propagated)
   and is what's shown as the shaded band in availability_pillar.py.
   Converted here to Risk_Score terms: Risk_Score_Lower = 100 - PSCI_upper,
   Risk_Score_Upper = 100 - PSCI_lower.
   QA note: recomputing this independently, 1,155 of 9,327 projected rows
   (12.4%) have a band that doesn't strictly bracket the point estimate --
   confirmed this is pure rank-denominator rounding noise (max width 0.08
   points on a 0-100 scale, mean 0.024, NOT concentrated in Connecticut or
   any other state) from the notebook's rank_against_pooled() helper using
   a slightly different array length than the original point-score rank.
   Defensively clipped below (min/max against the point estimate) so the
   rendered band always encloses its own point -- this changes nothing
   about the notebook's values, it only guarantees a sane polygon.

2. CONNECTICUT -- v3 pipeline (PSCI outputs v9) re-verified county-by-county
   against the live data (not taken on the handoff's word): 7 of the 9
   Connecticut planning regions now have real, varying Effective_MW across
   all 13 years and are shown normally. Only TWO regions are still broken --
   Greater Bridgeport (FIPS 09120) and South Central CT (FIPS 09170) --
   which both show an exact, unvarying Effective_MW == 0.0 for every single
   year 2016-2028, which is implausible on its face (Bridgeport is
   Connecticut's largest city) and was confirmed to be a genuine remaining
   pipeline gap, not fixed data. Suppression is now narrowed from "all of
   FIPS prefix 09" (v2) to just these two region codes, since the other 7
   are confirmed fixed and should no longer show as "no data". TCS_raw/
   TCS_density (S3, pure geometry, unaffected by either Dim1/Dim2 bug) are
   left as-is for ALL of CT (including the two still-broken regions) -- they
   are genuine, real data; only the composite score and the two broken
   Dim1/Dim2 inputs are suppressed, and only for the two still-broken
   regions.

3. TCS_density -- NEW passthrough column. The v9 pipeline adds
   TCS_density = TCS_raw / ALAND_SqMi (confirmed numerically against the
   source CSV, e.g. Autauga AL: 500.99 / 594.46 = 0.8428) alongside the
   existing TCS_raw. Both are carried through here so the app can switch
   the S3 display metric from the raw (county-size-biased) circuit-mile
   total to the area-normalized density, which is the fairer cross-county
   comparison. A national-median-by-year column is computed for density
   the same way it already is for TCS_raw.

4. GENERATION PLANT DETAIL -- NEW, v4. A separate source file,
   dim1_plant_detail_2025.csv (plant-level rows: FIPS, Plant_Name,
   Display_Type, Utility_Name, Nameplate_MW, Effective_MW), is a 2025
   SNAPSHOT only -- it carries no Year column and is not part of the
   2016-2028 panel/forecast time series above. It is read and passed
   through essentially as-is (sorted by size) into its own parquet so the
   app can show "which plants are in this county" alongside the existing
   county-level capacity trend, without implying this breakdown is
   projected or historical across years the way the rest of the pillar is.

Outputs -> data/:
  avail_panel.parquet        (2016-2025 historical, with the national
                               comparison stats above)
  avail_forecast.parquet     (2026-2028 projected, with the Risk_Score
                               uncertainty band)
  avail_county_lookup.parquet
  avail_plant_detail.parquet (2025 snapshot, plant-level, see item 4 above)
"""
import numpy as np
import pandas as pd

OUT_DIR = 'data'

PANEL_CSV_PATH = 'psci_composite_panel.csv'
PROJ_CSV_PATH = 'psci_composite_proj.csv'
PLANT_DETAIL_CSV_PATH = 'dim1_plant_detail_2025.csv'

PSCI_TO_RISK_TIER = {
    'Tier 1 – Very High': 'Tier 5 – Very Low Risk',
    'Tier 2 – High':      'Tier 4 – Low Risk',
    'Tier 3 – Moderate':  'Tier 3 – Moderate Risk',
    'Tier 4 – Low':       'Tier 2 – High Risk',
    'Tier 5 – Very Low':  'Tier 1 – Very High Risk',
}

# Still-broken Dim1/Dim2 for exactly two CT regions -- see module docstring,
# item 2. Narrowed (v3) from the old "all of FIPS prefix 09" blanket
# suppression now that the other 7 CT regions are confirmed fixed.
CT_BROKEN_FIPS = ['09120', '09170']  # Greater Bridgeport, South Central CT

panel = pd.read_csv(PANEL_CSV_PATH, dtype={'FIPS': str})
panel['FIPS'] = panel['FIPS'].str.zfill(5)

proj = pd.read_csv(PROJ_CSV_PATH, dtype={'FIPS': str})
proj['FIPS'] = proj['FIPS'].str.zfill(5)

for df in (panel, proj):
    df['Risk_Score'] = (100 - df['PSCI_score']).round(2)
    df['Risk_Tier'] = df['PSCI_Tier'].map(PSCI_TO_RISK_TIER)
    assert df['Risk_Tier'].notna().all(), (
        "Unmapped PSCI_Tier value found -- check PSCI_TO_RISK_TIER against "
        "the notebook's actual tier labels before re-running."
    )

# ---- Forecast uncertainty band (NEW in v2) ---------------------------------
# PSCI_lower/PSCI_upper -> Risk_Score terms, with a defensive clip so the
# rendered band always encloses its own point estimate (see module
# docstring item 1 for why ~12% of rows need this and why it's safe).
proj['Risk_Score_Lower'] = (100 - proj['PSCI_upper']).round(2)
proj['Risk_Score_Upper'] = (100 - proj['PSCI_lower']).round(2)
proj['Risk_Score_Lower'] = np.minimum(proj['Risk_Score_Lower'], proj['Risk_Score'])
proj['Risk_Score_Upper'] = np.maximum(proj['Risk_Score_Upper'], proj['Risk_Score'])

# ---- Connecticut defensive suppression (narrowed in v3, see docstring item 2) --
ct_mask = panel['FIPS'].isin(CT_BROKEN_FIPS)
ct_mask_proj = proj['FIPS'].isin(CT_BROKEN_FIPS)
n_ct_panel, n_ct_proj = int(ct_mask.sum()), int(ct_mask_proj.sum())
for col in ['S1_score', 'S2_inv_score', 'PSCI_score', 'PSCI_Tier', 'Risk_Score', 'Risk_Tier',
            'Effective_MW', 'County_Peak_MW', 'Primary_BA']:
    if col in panel.columns:
        panel.loc[ct_mask, col] = np.nan
for col in ['S1_score', 'S2_inv_score', 'PSCI_score', 'PSCI_Tier', 'Risk_Score', 'Risk_Tier',
            'Effective_MW', 'County_Peak_MW', 'County_Peak_MW_Lower', 'County_Peak_MW_Upper',
            'Risk_Score_Lower', 'Risk_Score_Upper']:
    if col in proj.columns:
        proj.loc[ct_mask_proj, col] = np.nan
# S3_score/TCS_raw deliberately left untouched for CT -- real, unaffected data.

# ---- National comparison stats (computed once, here -- not per-render) ----
# Observed years only, by the same interface decision already applied to
# Cost (Pct_Counties_MoreExpensive_SameYear) and Reliability
# (Pct_Counties_Less_Reliable_SameYear): forecast years get no such stat.
# Connecticut's now-NaN PSCI_score/Effective_MW/County_Peak_MW correctly
# fall out of both the ranking and the medians below (pandas skips NaN).
panel['_rank_pct'] = panel.groupby('Year')['PSCI_score'].rank(pct=True, method='average')
panel['Pct_Counties_BetterAvailability_SameYear'] = (panel['_rank_pct'] * 100).round(1)
panel['National_Median_Effective_MW_SameYear'] = (
    panel.groupby('Year')['Effective_MW'].transform('median')
)
panel['National_Median_Peak_MW_SameYear'] = (
    panel.groupby('Year')['County_Peak_MW'].transform('median')
)
panel['National_Median_TCS_SameYear'] = (
    panel.groupby('Year')['TCS_raw'].transform('median')
)
panel['National_Median_TCS_density_SameYear'] = (
    panel.groupby('Year')['TCS_density'].transform('median')
)
panel = panel.drop(columns=['_rank_pct'])

panel_cols = [
    'FIPS', 'County_Name', 'State', 'State_Name', 'Year',
    'Effective_MW', 'County_Peak_MW', 'TCS_raw', 'TCS_density', 'TCS_segment_count',
    'Primary_BA', 'S1_score', 'S2_inv_score', 'S3_score',
    'PSCI_score', 'PSCI_Tier', 'Risk_Score', 'Risk_Tier', 'Data_Type',
    'Pct_Counties_BetterAvailability_SameYear',
    'National_Median_Effective_MW_SameYear', 'National_Median_Peak_MW_SameYear',
    'National_Median_TCS_SameYear', 'National_Median_TCS_density_SameYear',
]
proj_cols = [
    'FIPS', 'County_Name', 'State', 'State_Name', 'Year',
    'Effective_MW', 'County_Peak_MW', 'County_Peak_MW_Lower', 'County_Peak_MW_Upper',
    'TCS_raw', 'TCS_density', 'TCS_segment_count', 'S1_score', 'S2_inv_score', 'S3_score',
    'PSCI_score', 'PSCI_Tier', 'Risk_Score', 'Risk_Tier',
    'Risk_Score_Lower', 'Risk_Score_Upper', 'Data_Type',
]
avail_panel = panel[panel_cols].sort_values(['State', 'County_Name', 'Year']).reset_index(drop=True)
avail_forecast = proj[proj_cols].sort_values(['State', 'County_Name', 'Year']).reset_index(drop=True)

avail_county_lookup = (
    pd.concat([avail_panel[['FIPS', 'State', 'County_Name']],
               avail_forecast[['FIPS', 'State', 'County_Name']]])
    .drop_duplicates('FIPS').sort_values(['State', 'County_Name']).reset_index(drop=True)
)

avail_panel.to_parquet(f'{OUT_DIR}/avail_panel.parquet', index=False)
avail_forecast.to_parquet(f'{OUT_DIR}/avail_forecast.parquet', index=False)
avail_county_lookup.to_parquet(f'{OUT_DIR}/avail_county_lookup.parquet', index=False)

# ---- Generation plant detail (NEW in v4, 2025 snapshot -- see docstring item 4) --
plant_detail = pd.read_csv(PLANT_DETAIL_CSV_PATH, dtype={'FIPS': str})
plant_detail['FIPS'] = plant_detail['FIPS'].str.zfill(5)
plant_detail = plant_detail.sort_values(
    ['FIPS', 'Effective_MW'], ascending=[True, False]
).reset_index(drop=True)
plant_detail = plant_detail[
    ['FIPS', 'Plant_Name', 'Display_Type', 'Utility_Name', 'Nameplate_MW', 'Effective_MW']
]
plant_detail.to_parquet(f'{OUT_DIR}/avail_plant_detail.parquet', index=False)
print(f"[Availability] avail_plant_detail.parquet: {len(plant_detail):,} plant rows, "
      f"{plant_detail['FIPS'].nunique():,} counties (2025 snapshot, not a time series)")

print(f"[Availability] avail_panel.parquet: {len(avail_panel):,} rows, "
      f"{avail_panel['FIPS'].nunique():,} counties, years "
      f"{avail_panel['Year'].min()}-{avail_panel['Year'].max()}")
print(f"[Availability] avail_forecast.parquet: {len(avail_forecast):,} rows, years "
      f"{avail_forecast['Year'].min()}-{avail_forecast['Year'].max()}")
print(f"[Availability] avail_county_lookup.parquet: {len(avail_county_lookup):,} counties")
print(f"[Availability] Risk_Score summary (historical, pooled):")
print(avail_panel['Risk_Score'].describe().round(2).to_string())
print(f"\n[Availability] Risk_Tier distribution (historical):")
print(avail_panel['Risk_Tier'].value_counts(dropna=False).sort_index().to_string())

print(f"\n[Availability] CAVEAT -- Connecticut: 7 of 9 planning regions are "
      f"confirmed fixed in this v9 pipeline run and show normally. Only "
      f"Greater Bridgeport (09120) and South Central CT (09170) are still "
      f"verified broken (Effective_MW == 0.0 for every year 2016-2028, "
      f"{n_ct_panel} historical + {n_ct_proj} forecast county-year rows "
      f"suppressed), so Risk_Score/PSCI_score have been set back to NaN for "
      f"just those two regions -- they'll still show as 'no data' on the "
      f"map, everywhere else in CT now shows real data. See this script's "
      f"module docstring, item 2, for detail.")
