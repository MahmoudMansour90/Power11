"""Reliability pillar page: County Reliability Index (CRI), 2013-2025
history (now including final 2025 EIA-861 data) + 2026-2028 projection.
Real data only -- see data_prep.py and the MOCCI_Reliability_Export v3
handoff note for how these files were produced and validated. The app
never recomputes anything the notebook already computed.

v3.3 -- third round of fixes from real post-deployment use (see
cost_pillar.py's module docstring for the full rationale -- this pillar
gets the same two structural changes, mirrored for CRI):
  - Click and search are now ONE unified experience. Clicking a county or
    searching for one both open the same full detail panel, always for
    the latest observed year: CRI, national standing, national median
    SAIDI, reporting utilities, national rank, the full trend chart, and
    the SAIDI/SAIFI expander. The separate "compact click card" from v3.2
    is gone.
  - Clicking a county on the map now also sets the State/County search
    dropdowns above the map to that same county, so map and search stay
    visually in sync (see cost_pillar.py for the session-state + rerun
    mechanism -- identical here).
  - Legend caption shortened -- dropped the "(not binned into natural-break
    classes)" aside per explicit request; CRI's 0-100 scale is unaffected
    by this pillar (always fixed, no percentile transform needed here,
    unlike Cost).
  - Removed "Years of CRI history" entirely, per the prior round's
    explicit request -- national rank is in the shared metrics block.
  - Metric fonts rebalanced: a long explanatory phrase ("more reliable
    than 19% of counties") is a caption under a short metric value, not
    the metric value itself.

Carried over unchanged from v3.1/v3.2:
  - Map is a single static choropleth per rerun, not a Plotly animation,
    built as two traces (real data + no data) with no dimming on click.
  - Alaska/Hawaii structurally impossible to show.
  - Year control is st.select_slider with an HTML every-2-years ruler.
  - Search is two dropdowns: State, then County (+ FIPS).
  - No "About this data and methodology" section anywhere (backtest
    scorecard, disclosures, the 100-ceiling explanation are all removed).
  - No confidence grade, no reporting-utility COUNT, no trend label, no
    Projected_CRI_Raw anywhere -- per the handoff's own "do not show" list.
  - No 68%/likely-range/probability-band numbers shown anywhere -- forecast
    years are tagged "(forecast)" only. The shaded band stays on the trend
    chart as a visual only, with no numeric label.
"""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from map_utils import (
    build_static_choropleth, render_year_slider, get_clicked_fips,
    render_gradient_legend, render_selected_year_caption, NO_DOWNLOAD_CONFIG,
    FORECAST_LINE_COLOR, FORECAST_CI_COLOR,
)

KEY = 'rel'


@st.cache_data
def load_data():
    panel = pd.read_parquet('data/reliability_panel.parquet')
    forecast = pd.read_parquet('data/reliability_forecast.parquet')
    not_projected = pd.read_parquet('data/reliability_not_projected.parquet')
    latest_gaps = pd.read_parquet('data/reliability_latest_year_gaps.parquet')
    county_lookup = pd.read_parquet('data/reliability_county_lookup.parquet')
    return panel, forecast, not_projected, latest_gaps, county_lookup


def _all_fips_base(geo, county_lookup):
    all_fips = [str(f.get('id', '')).zfill(5) for f in geo['features']]
    base = pd.DataFrame({'FIPS': all_fips}).drop_duplicates('FIPS')
    return base.merge(county_lookup, on='FIPS', how='left')


def _year_df(base, panel, forecast, sel_year, last_observed_year):
    """One row per contiguous county for sel_year, including counties
    with no CRI value that year (left as NaN -> shown as gray 'no data')."""
    if sel_year <= last_observed_year:
        slice_ = panel[panel['Year'] == sel_year][['FIPS', 'CRI', 'CRI_Tier']].rename(
            columns={'CRI': 'CRI_value'})
    else:
        slice_ = forecast[forecast['Year'] == sel_year][
            ['FIPS', 'Projected_CRI', 'CRI_Display', 'Projected_Tier']
        ].rename(columns={'Projected_CRI': 'CRI_value'})
    return base.merge(slice_, on='FIPS', how='left')


def _hover_text(df, is_forecast):
    out = []
    for _, r in df.iterrows():
        name = str(r['County_Name']).title() if pd.notna(r.get('County_Name')) else ''
        if pd.isna(r.get('CRI_value')):
            out.append(f"{name}, {r['State']}<br>No data reported")
        elif is_forecast:
            tier = r.get('Projected_Tier', '')
            out.append(f"{name}, {r['State']}<br>CRI: {r.get('CRI_Display', r['CRI_value'])} ({tier}) (forecast)")
        else:
            tier = r.get('CRI_Tier', '')
            out.append(f"{name}, {r['State']}<br>CRI: {r['CRI_value']:.1f} ({tier})")
    return out


def render(geo, state_borders):
    panel, forecast, not_projected, latest_gaps, county_lookup = load_data()
    last_observed_year = int(panel['Year'].max())
    forecast_horizon_year = last_observed_year + 3
    not_projected_fips = set(not_projected['FIPS'])
    gaps_by_fips = latest_gaps.set_index('FIPS')['Reason'].to_dict()
    base = _all_fips_base(geo, county_lookup)

    # Apply a click-driven dropdown sync queued by a PRIOR run, BEFORE the
    # search widgets below are instantiated this run -- see cost_pillar.py
    # for why this can't happen down near the map where the click is
    # actually detected (Streamlit refuses to write a widget's own
    # session-state key once that widget has already rendered earlier in
    # the current script pass, confirmed directly).
    pending_state = st.session_state.get(f'{KEY}_pending_state')
    if pending_state is not None:
        st.session_state[f'{KEY}_search_state'] = pending_state
        st.session_state[f'{KEY}_search_county_{pending_state}'] = st.session_state.get(f'{KEY}_pending_label')
        del st.session_state[f'{KEY}_pending_state']
        del st.session_state[f'{KEY}_pending_label']

    st.info(
        'Drag the slider to select a year from 2013 to 2028. Years 2013–'
        f'{last_observed_year} show actual values, while {last_observed_year + 1}–'
        f'{forecast_horizon_year} show forecast values. Click any county on '
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
    st.subheader("Power reliability risk index")
    st.caption(
        'A combined index of electricity outage frequency and duration across '
        'contiguous U.S. counties (0-100 scale, higher = worse reliability).'
    )
    vmin, vmax = 0, 100  # fixed per the handoff -- same scale every year

    years = list(range(2013, forecast_horizon_year + 1))
    sel_year = render_year_slider(f'{KEY}_year', years, last_observed_year, last_observed_year)
    is_forecast = sel_year > last_observed_year

    year_df = _year_df(base, panel, forecast, sel_year, last_observed_year)
    fig_map = build_static_choropleth(
        year_df, geo, state_borders, 'CRI_value', 'OrRd', vmin, vmax,
        hover_text=_hover_text(year_df, is_forecast),
    )
    event = st.plotly_chart(
        fig_map, width='stretch', key=f'{KEY}_map',
        on_select='rerun', selection_mode='points', config=NO_DOWNLOAD_CONFIG,
    )
    render_selected_year_caption(sel_year)
    render_gradient_legend(
        'OrRd', 'Lower Risk / Higher Reliability', 'Higher Risk / Lower Reliability',
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
            # Queue the sync for the top of the NEXT run (see above) --
            # the search widgets below have already rendered this run, so
            # their keys can't be written here even right before a rerun.
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
        _render_detail(panel, forecast, not_projected_fips, gaps_by_fips, sel_fips, last_observed_year)

    # ======================================================== comparison table
    st.divider()
    _render_comparison(panel, forecast, last_observed_year)

    # ======================================================== data sources
    st.divider()
    _render_data_sources()


def _national_rank(panel, sel_fips, year):
    """Rank among all U.S. counties with a reported CRI in `year`
    (1 = most reliable). Returns None if this county has no CRI that year."""
    year_slice = panel[panel['Year'] == year].dropna(subset=['CRI'])
    if year_slice.empty:
        return None
    year_slice = year_slice.sort_values('CRI').reset_index(drop=True)  # lowest CRI = most reliable = rank 1
    row = year_slice[year_slice['FIPS'] == sel_fips]
    if row.empty:
        return None
    return int(row.index[0]) + 1, len(year_slice)


def _render_metrics_block(panel, sel_fips, year, gaps_by_fips, last_observed_year):
    """The core set of metrics for one county/year: CRI, national
    standing, national median SAIDI, reporting utilities, national rank.
    Metric fonts are kept balanced -- a short numeric value in st.metric,
    with the explanatory sentence as a caption underneath rather than
    crammed into the metric itself."""
    row = panel[(panel['FIPS'] == sel_fips) & (panel['Year'] == year)]
    if row.empty or pd.isna(row.iloc[0]['CRI']):
        st.metric(f'CRI ({year})', f'No {year} data')
        reason = gaps_by_fips.get(sel_fips) if year == last_observed_year else None
        st.caption(reason or 'No data reported for this county in this year.')
        return
    r = row.iloc[0]
    m1, m2, m3 = st.columns(3)
    m1.metric(f'CRI ({year})', f"{r['CRI']:.1f}", r['CRI_Tier'])
    m2.metric('National standing', f"{r['Pct_Counties_Less_Reliable_SameYear']:.0f}%")
    m2.caption(f"more reliable than {r['Pct_Counties_Less_Reliable_SameYear']:.0f}% of counties")
    m3.metric('National median SAIDI', f"{r['National_Median_SAIDI_SameYear']:.1f} min/cust")

    saidi_val = r.get('SAIDI_min_per_customer')
    saifi_val = r.get('SAIFI_interruptions_per_customer')
    median_saifi = r.get('National_Median_SAIFI_SameYear')
    n1, n2, n3 = st.columns(3)
    n1.metric(f'County SAIDI ({year})',
              f"{saidi_val:.1f} min/cust" if pd.notna(saidi_val) else 'No data')
    n2.metric(f'County SAIFI ({year})',
              f"{saifi_val:.2f} interruptions/cust" if pd.notna(saifi_val) else 'No data')
    n3.metric('National median SAIFI',
              f"{median_saifi:.2f} interruptions/cust" if pd.notna(median_saifi) else 'No data')

    reporting = r.get('Reporting_Utilities')
    if isinstance(reporting, str) and reporting:
        st.caption(f"Reporting utilities: {reporting}")
    rank_info = _national_rank(panel, sel_fips, year)
    if rank_info:
        rank, n_total = rank_info
        st.caption(f"National rank ({year}): **#{rank} most reliable** of {n_total:,} U.S. counties")


def _render_detail(panel, forecast, not_projected_fips, gaps_by_fips, sel_fips, last_observed_year):
    """Always the full experience, always the latest observed year --
    identical whether sel_fips came from a map click or a search (see
    module docstring: click and search are now unified)."""
    p_sel = panel[panel['FIPS'] == sel_fips].sort_values('Year')
    f_sel = forecast[forecast['FIPS'] == sel_fips].sort_values('Year')
    if p_sel.empty and f_sel.empty:
        st.warning('No data for this county.')
        return
    name_row = (p_sel.iloc[-1] if not p_sel.empty else f_sel.iloc[-1])
    st.subheader(f"{name_row['County_Name'].title()}, {name_row['State']}")

    _render_metrics_block(panel, sel_fips, last_observed_year, gaps_by_fips, last_observed_year)
    if sel_fips in not_projected_fips:
        st.caption('Note: this county has too little history for a 3-year projection.')

    p_sel_valid = p_sel.dropna(subset=['CRI'])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=p_sel['Year'], y=p_sel['CRI'], mode='lines+markers',
                              name='Observed', line=dict(color='#1f77b4', width=2),
                              connectgaps=False))
    if not f_sel.empty and not p_sel_valid.empty:
        bridge_year = pd.concat([p_sel_valid['Year'].tail(1), f_sel['Year']])
        bridge_cri = pd.concat([p_sel_valid['CRI'].tail(1), f_sel['Projected_CRI']])
        has_ci = f_sel['CRI_Low_68'].notna().all() and f_sel['CRI_High_68'].notna().all()
        if has_ci:
            upper = pd.concat([p_sel_valid['CRI'].tail(1), f_sel['CRI_High_68']])
            lower = pd.concat([p_sel_valid['CRI'].tail(1), f_sel['CRI_Low_68']])
            fig.add_trace(go.Scatter(
                x=pd.concat([bridge_year, bridge_year[::-1]]),
                y=pd.concat([upper, lower[::-1]]),
                fill='toself', fillcolor='rgba(255,204,128,0.45)',  # FORECAST_CI_COLOR as rgba
                line=dict(color='rgba(255,255,255,0)'), hoverinfo='skip',
                name='Forecast range', showlegend=False,
            ))
        fig.add_trace(go.Scatter(x=bridge_year, y=bridge_cri, mode='lines+markers',
                                  name='Projected', line=dict(color=FORECAST_LINE_COLOR, width=2, dash='dash'),
                                  hovertemplate='%{x}: %{y:.1f} (forecast)<extra></extra>'))
    fig.update_layout(
        xaxis_title='Year', yaxis_title='CRI (0-100, higher = worse)',
        # a narrow year range otherwise gets fractional/comma'd ticks; tickfont
        # bumped up per explicit request (was Plotly's default ~ size 11)
        xaxis=dict(tickformat='d', dtick=1, tickfont=dict(size=14)),
        height=380, margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation='h', y=1.12), hovermode='x unified',
        yaxis_range=[0, 100],
    )
    st.plotly_chart(fig, width='stretch', key=f'{KEY}_trend_{sel_fips}', config=NO_DOWNLOAD_CONFIG)

    with st.expander('**SAIDI / SAIFI history**'):
        saidi_saifi = p_sel[['Year', 'SAIDI_min_per_customer', 'SAIFI_interruptions_per_customer']].dropna(
            how='all', subset=['SAIDI_min_per_customer', 'SAIFI_interruptions_per_customer'])
        if saidi_saifi.empty:
            st.caption('No SAIDI/SAIFI history available for this county.')
        else:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=saidi_saifi['Year'], y=saidi_saifi['SAIDI_min_per_customer'],
                                       name='SAIDI (min/customer)', mode='lines+markers', yaxis='y1'))
            fig2.add_trace(go.Scatter(x=saidi_saifi['Year'], y=saidi_saifi['SAIFI_interruptions_per_customer'],
                                       name='SAIFI (interruptions/customer)', mode='lines+markers', yaxis='y2'))
            fig2.update_layout(
                height=300, margin=dict(l=10, r=10, t=20, b=10),
                xaxis=dict(title='Year', tickformat='d', dtick=1),
                yaxis=dict(title='SAIDI (min/customer)'),
                yaxis2=dict(title='SAIFI', overlaying='y', side='right'),
                legend=dict(orientation='h', y=1.15),
            )
            st.plotly_chart(fig2, width='stretch', key=f'{KEY}_saidi_saifi_{sel_fips}', config=NO_DOWNLOAD_CONFIG)


_RELIABILITY_COL_RENAME = {
    'County_Name': 'County', 'Last_Year': 'Year', 'CRI': 'CRI', 'CRI_Tier': 'Tier',
    'Pct_Counties_Less_Reliable_SameYear': 'Reliable %', 'Projected_CRI': 'Fcst CRI',
    'Projected_Tier': 'Fcst Tier', 'Change_to_forecast_pct': 'Chg (%)',
    'SAIDI_min_per_customer': 'SAIDI', 'SAIFI_interruptions_per_customer': 'SAIFI',
}
_RELIABILITY_COL_WIDTH = {
    'FIPS': 'small', 'State': 'small', 'County': 'medium', 'Year': 'small',
    'CRI': 'small', 'Tier': 'small', 'Reliable %': 'small', 'Fcst CRI': 'small',
    'Fcst Tier': 'small', 'Chg (%)': 'small', 'SAIDI': 'small', 'SAIFI': 'small',
}


def _display_reliability_comparison(table, forecast_horizon_year):
    """Shared dataframe rendering for the shortlist comparison, with SAIDI/
    SAIFI included per explicit request. Column headers are kept short (so
    they fit on screen without horizontal scrolling -- st.dataframe doesn't
    support wrapping a long header onto two lines, and ignores an explicit
    narrow width whenever the header text itself needs more room, confirmed
    directly) with the full description moved into each column's hover
    tooltip instead. Returns the rendered (renamed/rounded) dataframe so
    the download button can hand off the exact same table."""
    cols = [
        'FIPS', 'State', 'County_Name', 'Last_Year', 'CRI', 'CRI_Tier',
        'Pct_Counties_Less_Reliable_SameYear', 'Projected_CRI', 'Projected_Tier',
        'Change_to_forecast_pct', 'SAIDI_min_per_customer', 'SAIFI_interruptions_per_customer',
    ]
    display_df = table[cols].rename(columns=_RELIABILITY_COL_RENAME).round(1)
    help_text = {
        'Year': 'Last observed year', 'CRI': 'Last observed CRI (0-100, higher = worse)',
        'Reliable %': 'More reliable than % of counties',
        'Fcst CRI': f'{forecast_horizon_year} projected CRI', 'Fcst Tier': f'{forecast_horizon_year} projected tier',
        'Chg (%)': f'Change from last observed CRI to the {forecast_horizon_year} forecast, percent',
        'SAIDI': 'SAIDI, minutes per customer per year', 'SAIFI': 'SAIFI, interruptions per customer per year',
    }
    st.dataframe(
        display_df, width='stretch', hide_index=True,
        column_config={
            c: st.column_config.Column(width=_RELIABILITY_COL_WIDTH.get(c, 'small'), help=help_text.get(c))
            for c in display_df.columns
        },
    )
    return display_df


def _render_comparison(panel, forecast, last_observed_year):
    """v3.5/v3.9: this section used to be a single context-free top/bottom-N
    leaderboard, then grew a rank-based 'peer window' and a rank-based
    top/bottom toggle. Per explicit request, both of those (and their
    Scope/State controls) are now gone -- this is just the user's own
    up-to-20 shortlist, built by name via multiselect, shown side-by-side
    with SAIDI/SAIFI and a CSV download scoped to just this one table (the
    app's global disableDataExport setting hides the implicit per-dataframe
    download icon everywhere else; this is one deliberate, scoped
    exception, not a reversal of that policy). The 'Change to
    {forecast_horizon_year} (%)' column is still computed from CRI and
    Projected_CRI (negative = improving, since lower CRI = more reliable)
    -- a trajectory the map's single-year snapshot can't show."""
    st.subheader('Compare counties')
    latest_panel = panel.sort_values('Year').drop_duplicates('FIPS', keep='last').set_index('FIPS')
    forecast_horizon_year = last_observed_year + 3
    fc_last = forecast[forecast['Year'] == forecast_horizon_year][
        ['FIPS', 'Projected_CRI', 'Projected_Tier']
    ]

    table = latest_panel.reset_index().rename(columns={'Year': 'Last_Year'})
    table = table.merge(fc_last, on='FIPS', how='left')
    table['County_Name'] = table['County_Name'].str.title()
    table['Change_to_forecast_pct'] = np.where(
        table['CRI'] > 0,
        (table['Projected_CRI'] - table['CRI']) / table['CRI'] * 100,
        np.nan,
    )

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
    display_df = _display_reliability_comparison(shortlist, forecast_horizon_year)
    st.download_button(
        'Download this comparison as CSV', display_df.to_csv(index=False).encode('utf-8'),
        file_name='reliability_comparison_shortlist.csv', mime='text/csv',
        key=f'{KEY}_shortlist_download',
    )


def _render_data_sources():
    st.subheader('Data sources')
    st.markdown(
        '- Energy Information Administration (EIA Form 861): (Reliability, '
        'Service Territory). 2013-2025\n'
        '- U.S. Census Bureau: (county-level population). 2013-2025\n'
        '- U.S. Census Bureau TIGER/Line county boundaries. 2025'
    )
