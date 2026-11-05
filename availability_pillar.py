"""Availability / Capacity pillar page: Power Supply Capacity Index (PSCI),
2016-2025 history + 2026-2028 projection. Real data only -- produced by
PSCI_Panel_Pipeline_FIXED_2.ipynb and converted to the app's parquet
snapshot by data_prep_availability.py (see that file's module docstring
for the full methodology verification and data-quality caveats). The app
never recomputes anything the notebook already computed.

Mirrors reliability_pillar.py's structure and interaction model exactly
(unified click/search detail panel, static per-rerun choropleth, the same
year-slider/ruler, the same peer-window comparison table) -- same
rationale as that file's own docstring, not repeated here.

RISK FRAMING (consistent with Cost's cents/kWh and Reliability's CRI --
"higher score = worse" everywhere in this app):
  Risk_Score = 100 - PSCI_score, 0-100, higher = worse (lower availability).
  PSCI_score itself (the notebook's own native unit) is no longer shown in
  the UI -- Risk_Score alone, per explicit request.

Three sub-dimensions behind PSCI_score (all pooled-percentile-rank based,
see data_prep_availability.py):
  S1 -- generation capacity (Effective_MW)
  S2 (inverted) -- spare peak-demand headroom (County_Peak_MW, inverted:
       lower relative peak demand = more spare capacity = better)
  S3 -- transmission connectivity (TCS_raw, HIFLD 2025 snapshot,
       time-invariant across all 13 years)

KNOWN GAP carried through from data_prep_availability.py (not hidden --
surfaced in the info banner and the no-data legend, same as every other
gap this app already shows this way):
  - 7 of Connecticut's 9 planning regions are confirmed fixed in the v9
    pipeline and show real data normally. Only Greater Bridgeport (09120)
    and South Central CT (09170) are still verified broken (Effective_MW
    == 0.0 every year), so Risk_Score/PSCI_score are suppressed back to
    NaN for just those two regions at prep time -- see
    data_prep_availability.py's module docstring for detail.

FORECAST UNCERTAINTY BAND (v2, new): the Risk Score trend chart now shows
a real shaded band for 2026-2028, built from the notebook's own
PSCI_lower/PSCI_upper (re-ranking the S2 demand forecast's confidence
bounds through the same pooled percentile curve as the point estimate,
S1/S3 held fixed -- see data_prep_availability.py). Labeled "S2 demand
uncertainty" in the legend, matching the notebook author's own chosen
label, since it is not a true composite-score confidence interval.
"""
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from map_utils import (
    build_static_choropleth, render_year_slider, get_clicked_fips,
    render_gradient_legend, render_selected_year_caption, NO_DOWNLOAD_CONFIG,
    FORECAST_LINE_COLOR, FORECAST_CI_COLOR,
)

KEY = 'avail'


@st.cache_data
def load_data():
    panel = pd.read_parquet('data/avail_panel.parquet')
    forecast = pd.read_parquet('data/avail_forecast.parquet')
    county_lookup = pd.read_parquet('data/avail_county_lookup.parquet')
    return panel, forecast, county_lookup


@st.cache_data
def load_plant_detail():
    """2025 generation-plant snapshot (dim1_plant_detail_2025.csv via
    data_prep_availability.py) -- plant-level, no Year dimension, separate
    from the 2016-2028 panel/forecast time series above."""
    return pd.read_parquet('data/avail_plant_detail.parquet')


def _all_fips_base(geo, county_lookup):
    all_fips = [str(f.get('id', '')).zfill(5) for f in geo['features']]
    base = pd.DataFrame({'FIPS': all_fips}).drop_duplicates('FIPS')
    return base.merge(county_lookup, on='FIPS', how='left')


def _year_df(base, panel, forecast, sel_year, last_observed_year):
    """One row per contiguous county for sel_year, including counties
    with no PSCI value that year (left as NaN -> shown as gray 'no
    data') -- same convention as every other pillar in this app."""
    cols = ['FIPS', 'Risk_Score', 'Risk_Tier']
    if sel_year <= last_observed_year:
        slice_ = panel[panel['Year'] == sel_year][cols]
    else:
        slice_ = forecast[forecast['Year'] == sel_year][cols]
    slice_ = slice_.rename(columns={'Risk_Score': 'Risk_Score_value', 'Risk_Tier': 'Risk_Tier_value'})
    return base.merge(slice_, on='FIPS', how='left')


def _hover_text(df, is_forecast):
    out = []
    for _, r in df.iterrows():
        name = str(r['County_Name']).title() if pd.notna(r.get('County_Name')) else ''
        if pd.isna(r.get('Risk_Score_value')):
            out.append(f"{name}, {r['State']}<br>No data reported")
        else:
            tier = r.get('Risk_Tier_value', '')
            tag = ' (forecast)' if is_forecast else ''
            out.append(f"{name}, {r['State']}<br>Risk Score: {r['Risk_Score_value']:.1f} ({tier}){tag}")
    return out


def render(geo, state_borders):
    panel, forecast, county_lookup = load_data()
    last_observed_year = int(panel['Year'].max())
    forecast_horizon_year = last_observed_year + 3
    base = _all_fips_base(geo, county_lookup)

    # Click/search sync queued by a prior run -- identical mechanism to
    # cost_pillar.py / reliability_pillar.py, see those files for why this
    # has to happen before the search widgets below are instantiated.
    pending_state = st.session_state.get(f'{KEY}_pending_state')
    if pending_state is not None:
        st.session_state[f'{KEY}_search_state'] = pending_state
        st.session_state[f'{KEY}_search_county_{pending_state}'] = st.session_state.get(f'{KEY}_pending_label')
        del st.session_state[f'{KEY}_pending_state']
        del st.session_state[f'{KEY}_pending_label']

    st.info(
        'Drag the slider to select a year from 2016 to 2028. Years 2016–'
        f'{last_observed_year} show actual values, while {last_observed_year + 1}–'
        f'{forecast_horizon_year} show projected values. Click any county on '
        'the map or search for a county name below to view its detailed information '
        'and trends.'
    )

    # -------------------------------------------------- search (two dropdowns)
    s1, s2 = st.columns(2)
    state_options = sorted(county_lookup['State'].dropna().unique())
    sel_state = s1.selectbox(
        'State', state_options, index=None, placeholder='Select a state...', key=f'{KEY}_search_state',
    )
    if sel_state:
        county_df = county_lookup[county_lookup['State'] == sel_state].sort_values('County_Name')
        county_labels = [f"{r.County_Name.title()} ({r.FIPS})" for r in county_df.itertuples()]
    else:
        county_labels = []
    sel_county_label = s2.selectbox(
        'County', county_labels, index=None, placeholder='Select a county...',
        key=f'{KEY}_search_county_{sel_state}',
    )

    # ---------------------------------------------------------------- map
    st.subheader("Power availability risk index")
    st.caption(
        'A combined index of power supply/generation capacity, peak demand, and '
        'transmission network connectivity across contiguous U.S. counties '
        '(0-100 scale, higher = worse availability).'
    )
    vmin, vmax = 0, 100  # fixed, same as Reliability's CRI

    years = list(range(2016, forecast_horizon_year + 1))
    sel_year = render_year_slider(f'{KEY}_year', years, last_observed_year, last_observed_year)
    is_forecast = sel_year > last_observed_year

    year_df = _year_df(base, panel, forecast, sel_year, last_observed_year)
    fig_map = build_static_choropleth(
        year_df, geo, state_borders, 'Risk_Score_value', 'OrRd', vmin, vmax,
        hover_text=_hover_text(year_df, is_forecast),
    )
    event = st.plotly_chart(
        fig_map, width='stretch', key=f'{KEY}_map',
        on_select='rerun', selection_mode='points', config=NO_DOWNLOAD_CONFIG,
    )
    render_selected_year_caption(sel_year)
    render_gradient_legend(
        'OrRd', 'Lower Risk / Higher Availability', 'Higher Risk / Lower Availability',
    )

    # ---------------------------------- resolve selection, keeping map click
    # ---------------------------------- and the search dropdowns in sync
    click_fips = get_clicked_fips(event)
    prev_click = st.session_state.get(f'{KEY}_prev_click')

    if click_fips is not None and click_fips != prev_click:
        st.session_state[f'{KEY}_prev_click'] = click_fips
        match = base[base['FIPS'] == click_fips]
        if not match.empty:
            c_state = match.iloc[0]['State']
            c_label = f"{str(match.iloc[0]['County_Name']).title()} ({click_fips})"
            st.session_state[f'{KEY}_pending_state'] = c_state
            st.session_state[f'{KEY}_pending_label'] = c_label
            st.session_state[f'{KEY}_prev_search_val'] = (c_state, c_label)
            st.session_state[f'{KEY}_sel_fips'] = click_fips
            st.rerun()

    sel_fips = st.session_state.get(f'{KEY}_sel_fips')
    prev_search = st.session_state.get(f'{KEY}_prev_search_val')
    search_key = (sel_state, sel_county_label)
    if sel_county_label is not None and search_key != prev_search:
        sel_fips = sel_county_label.split('(')[-1].rstrip(')')
        st.session_state[f'{KEY}_prev_search_val'] = search_key

    st.session_state[f'{KEY}_sel_fips'] = sel_fips

    # ============================================================ county detail
    st.divider()
    if sel_fips is None:
        st.caption('Click a county on the map, or search for one above, to see its detail.')
    else:
        _render_detail(panel, forecast, sel_fips, last_observed_year)

    # ======================================================== comparison table
    st.divider()
    _render_comparison(panel, forecast, last_observed_year)

    # ======================================================== data sources
    st.divider()
    _render_data_sources()


def _national_rank(panel, sel_fips, year):
    """Rank among all U.S. counties with a reported Risk_Score in `year`
    (1 = best availability / lowest risk). None if not covered that year."""
    year_slice = panel[panel['Year'] == year].dropna(subset=['Risk_Score'])
    if year_slice.empty:
        return None
    year_slice = year_slice.sort_values('Risk_Score').reset_index(drop=True)
    row = year_slice[year_slice['FIPS'] == sel_fips]
    if row.empty:
        return None
    return int(row.index[0]) + 1, len(year_slice)


def _render_metrics_block(panel, sel_fips, year):
    row = panel[(panel['FIPS'] == sel_fips) & (panel['Year'] == year)]
    if row.empty or pd.isna(row.iloc[0]['Risk_Score']):
        st.metric(f'Risk Score ({year})', f'No {year} data')
        st.caption('No data reported for this county in this year.')
        return
    r = row.iloc[0]

    m1, m2 = st.columns(2)
    m1.metric(f'Risk Score ({year})', f"{r['Risk_Score']:.1f}", r['Risk_Tier'])
    better_pct = r.get('Pct_Counties_BetterAvailability_SameYear')
    if pd.notna(better_pct):
        m2.metric('National standing', f"Top {100 - better_pct:.0f}%")
        m2.caption(f"better availability than {better_pct:.0f}% of counties")
    else:
        m2.metric('National standing', 'N/A')

    n1, n2, n3 = st.columns(3)
    n1.metric(f'Generation capacity (S1)', f"{r['Effective_MW']:,.0f} MW")
    n1.caption(f"National median: {r['National_Median_Effective_MW_SameYear']:,.0f} MW")
    n2.metric(f'Peak demand (S2)', f"{r['County_Peak_MW']:,.0f} MW")
    n2.caption(f"National median: {r['National_Median_Peak_MW_SameYear']:,.0f} MW")
    n3.metric(f'Transmission Network connectivity density (S3)', f"{r['TCS_density']:,.2f} kV-mi/sq mi")
    n3.caption(f"National median: {r['National_Median_TCS_density_SameYear']:,.2f} kV-mi/sq mi")

    rank_info = _national_rank(panel, sel_fips, year)
    if rank_info:
        rank, n_total = rank_info
        st.caption(f"National rank ({year}): **#{rank} best availability** of {n_total:,} U.S. counties")


def _render_detail(panel, forecast, sel_fips, last_observed_year):
    """Always the full experience, always the latest observed year --
    identical whether sel_fips came from a map click or a search."""
    p_sel = panel[panel['FIPS'] == sel_fips].sort_values('Year')
    f_sel = forecast[forecast['FIPS'] == sel_fips].sort_values('Year')
    if p_sel.empty and f_sel.empty:
        st.warning('No data for this county.')
        return
    name_row = (p_sel.iloc[-1] if not p_sel.empty else f_sel.iloc[-1])
    st.subheader(f"{name_row['County_Name'].title()}, {name_row['State']}")

    _render_metrics_block(panel, sel_fips, last_observed_year)

    # ------------------------------------------------- Risk Score trend chart
    p_sel_valid = p_sel.dropna(subset=['Risk_Score'])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=p_sel['Year'], y=p_sel['Risk_Score'], mode='lines+markers',
                              name='Observed', line=dict(color='#1f77b4', width=2),
                              connectgaps=False))
    if not f_sel.empty and not p_sel_valid.empty:
        bridge_year = pd.concat([p_sel_valid['Year'].tail(1), f_sel['Year']])
        bridge_risk = pd.concat([p_sel_valid['Risk_Score'].tail(1), f_sel['Risk_Score']])
        has_band = f_sel['Risk_Score_Lower'].notna().all() and f_sel['Risk_Score_Upper'].notna().all()
        if has_band:
            upper = pd.concat([p_sel_valid['Risk_Score'].tail(1), f_sel['Risk_Score_Upper']])
            lower = pd.concat([p_sel_valid['Risk_Score'].tail(1), f_sel['Risk_Score_Lower']])
            fig.add_trace(go.Scatter(
                x=pd.concat([bridge_year, bridge_year[::-1]]),
                y=pd.concat([upper, lower[::-1]]),
                fill='toself', fillcolor='rgba(255,204,128,0.45)',  # FORECAST_CI_COLOR as rgba
                line=dict(color='rgba(255,255,255,0)'), hoverinfo='skip',
                showlegend=False,
            ))
        fig.add_trace(go.Scatter(x=bridge_year, y=bridge_risk, mode='lines+markers',
                                  name='Projected', line=dict(color=FORECAST_LINE_COLOR, width=2, dash='dash'),
                                  hovertemplate='%{x}: %{y:.1f} (forecast)<extra></extra>'))
    fig.update_layout(
        xaxis_title='Year', yaxis_title='Risk Score (0-100, higher = worse)',
        # tickfont bumped up per explicit request (was Plotly's default ~ size 11)
        xaxis=dict(tickformat='d', dtick=1, tickfont=dict(size=14)),
        height=380, margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation='h', y=1.12), hovermode='x unified',
        yaxis_range=[0, 100],
    )
    st.plotly_chart(fig, width='stretch', key=f'{KEY}_trend_{sel_fips}', config=NO_DOWNLOAD_CONFIG)

    # ------------------------------------------------- supply vs demand
    with st.expander('**Supply vs. demand trend (generation vs. peak demand)**'):
        hist = p_sel[['Year', 'Effective_MW', 'County_Peak_MW']]
        fc = f_sel[['Year', 'Effective_MW', 'County_Peak_MW']]
        combo = pd.concat([hist, fc]).sort_values('Year')
        if combo.empty:
            st.caption('No generation/demand history available for this county.')
        else:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=combo['Year'], y=combo['Effective_MW'],
                                       name='Generation capacity (MW)', mode='lines+markers',
                                       line=dict(color='#1f77b4'), yaxis='y1'))
            fig2.add_trace(go.Scatter(x=combo['Year'], y=combo['County_Peak_MW'],
                                       name='Peak demand (MW)', mode='lines+markers',
                                       line=dict(color='#e35b1e'), yaxis='y2'))
            fig2.update_layout(
                height=320, margin=dict(l=10, r=10, t=20, b=10),
                xaxis=dict(title='Year', tickformat='d', dtick=1),
                yaxis=dict(title='Generation capacity (MW)'),
                yaxis2=dict(title='Peak demand (MW)', overlaying='y', side='right'),
                legend=dict(orientation='h', y=1.15),
            )
            st.plotly_chart(fig2, width='stretch', key=f'{KEY}_supply_demand_{sel_fips}', config=NO_DOWNLOAD_CONFIG)

    # ------------------------------------------------- generation plants (2025)
    with st.expander('**Generation plants (2025)**'):
        plants = load_plant_detail()
        p_sel_plants = plants[plants['FIPS'] == sel_fips].sort_values('Effective_MW', ascending=False)
        if p_sel_plants.empty:
            st.caption('No generation plants on record for this county.')
        else:
            st.dataframe(
                p_sel_plants[
                    ['Plant_Name', 'Display_Type', 'Utility_Name', 'Nameplate_MW', 'Effective_MW']
                ].rename(columns={
                    'Plant_Name': 'Plant', 'Display_Type': 'Technology', 'Utility_Name': 'Utility',
                    'Nameplate_MW': 'Nameplate (MW)', 'Effective_MW': 'Effective (MW)',
                }).round(1),
                width='stretch', hide_index=True,
            )


_AVAIL_COL_RENAME = {
    'County_Name': 'County', 'Last_Year': 'Year', 'Risk_Score': 'Risk', 'Risk_Tier': 'Tier',
    'Pct_Counties_BetterAvailability_SameYear': 'Avail %', 'Projected_Risk_Score': 'Fcst Risk',
    'Projected_Risk_Tier': 'Fcst Tier', 'Change_to_forecast_pct': 'Chg (%)',
    'Effective_MW': 'Capacity', 'County_Peak_MW': 'Peak', 'TCS_density': 'Trans. density',
}
_AVAIL_COL_WIDTH = {
    'FIPS': 'small', 'State': 'small', 'County': 'medium', 'Year': 'small',
    'Risk': 'small', 'Tier': 'small', 'Avail %': 'small', 'Fcst Risk': 'small',
    'Fcst Tier': 'small', 'Chg (%)': 'small', 'Capacity': 'small', 'Peak': 'small',
    'Trans. density': 'small',
}


def _display_availability_comparison(table, forecast_horizon_year):
    """Shared dataframe rendering for the shortlist comparison, with
    generation capacity / peak demand / transmission connectivity density
    included per explicit request. Column headers are kept short (so they
    fit on screen without horizontal scrolling -- st.dataframe doesn't
    support wrapping a long header onto two lines, and ignores an explicit
    narrow width whenever the header text itself needs more room, confirmed
    directly) with the full description moved into each column's hover
    tooltip instead. Returns the rendered (renamed/rounded) dataframe so
    the download button can hand off the exact same table."""
    cols = [
        'FIPS', 'State', 'County_Name', 'Last_Year', 'Risk_Score', 'Risk_Tier',
        'Pct_Counties_BetterAvailability_SameYear', 'Projected_Risk_Score',
        'Projected_Risk_Tier', 'Change_to_forecast_pct',
        'Effective_MW', 'County_Peak_MW', 'TCS_density',
    ]
    display_df = table[cols].rename(columns=_AVAIL_COL_RENAME).round(1)
    help_text = {
        'Year': 'Last observed year', 'Risk': 'Last observed Risk Score (0-100, higher = worse)',
        'Avail %': 'Better availability than % of counties',
        'Fcst Risk': f'{forecast_horizon_year} projected Risk Score',
        'Fcst Tier': f'{forecast_horizon_year} projected tier',
        'Chg (%)': f'Change from last observed Risk Score to the {forecast_horizon_year} forecast, percent',
        'Capacity': 'Generation capacity, MW', 'Peak': 'Peak demand, MW',
        'Trans. density': 'Transmission connectivity density, kV-mi/sq mi',
    }
    st.dataframe(
        display_df, width='stretch', hide_index=True,
        column_config={
            c: st.column_config.Column(width=_AVAIL_COL_WIDTH.get(c, 'small'), help=help_text.get(c))
            for c in display_df.columns
        },
    )
    return display_df


def _render_comparison(panel, forecast, last_observed_year):
    """v3.9: mirrors reliability_pillar.py's _render_comparison exactly --
    this used to also have a rank-based 'peer window' and top/bottom
    toggle with Scope/State controls; per explicit request those are gone,
    leaving just the user's own up-to-20 shortlist, built by name via
    multiselect, shown side-by-side with generation capacity / peak demand
    / transmission connectivity density and a CSV download scoped to just
    this one table -- see that file for the full disableDataExport
    rationale, not repeated here."""
    st.subheader('Compare counties')
    latest_panel = panel.sort_values('Year').drop_duplicates('FIPS', keep='last').set_index('FIPS')
    forecast_horizon_year = last_observed_year + 3
    fc_last = forecast[forecast['Year'] == forecast_horizon_year][
        ['FIPS', 'Risk_Score', 'Risk_Tier']
    ].rename(columns={'Risk_Score': 'Projected_Risk_Score', 'Risk_Tier': 'Projected_Risk_Tier'})

    table = latest_panel.reset_index().rename(columns={'Year': 'Last_Year'})
    table = table.merge(fc_last, on='FIPS', how='left')
    table['County_Name'] = table['County_Name'].str.title()
    table['Change_to_forecast_pct'] = (
        (table['Projected_Risk_Score'] - table['Risk_Score']) / table['Risk_Score'] * 100
    ).where(table['Risk_Score'] > 0)

    label_map = {
        f"{r.County_Name}, {r.State} ({r.FIPS})": r.FIPS
        for r in table.sort_values(['State', 'County_Name']).itertuples()
    }
    sel_labels = st.multiselect(
        'Counties to compare (up to 20)', sorted(label_map.keys()),
        max_selections=20, key=f'{KEY}_shortlist',
    )
    if not sel_labels:
        st.caption('Choose counties above to build a side-by-side comparison.')
        return
    chosen_fips = [label_map[lbl] for lbl in sel_labels]
    shortlist = table[table['FIPS'].isin(chosen_fips)].copy()
    shortlist['_order'] = shortlist['FIPS'].map({f: i for i, f in enumerate(chosen_fips)})
    shortlist = shortlist.sort_values('_order').drop(columns='_order')
    display_df = _display_availability_comparison(shortlist, forecast_horizon_year)
    st.download_button(
        'Download this comparison as CSV', display_df.to_csv(index=False).encode('utf-8'),
        file_name='availability_comparison_shortlist.csv', mime='text/csv',
        key=f'{KEY}_shortlist_download',
    )


def _render_data_sources():
    st.subheader('Data sources')
    st.markdown(
        '- Energy Information Administration (EIA Form 860). 2016-2025\n'
        '- Energy Information Administration (EIA Form 861) (retail sales, '
        'balancing-authority). 2016-2025\n'
        '- Energy Information Administration (EIA Form 930) (hourly '
        'balancing-authority demand). 2016-2025\n'
        '- U.S. Census Bureau: (county-level population). 2016-2025\n'
        '- HIFLD Electric Power Transmission Lines. 2025\n'
        '- U.S. Census Bureau TIGER/Line county boundaries. 2025'
    )
