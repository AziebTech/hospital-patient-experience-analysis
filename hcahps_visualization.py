"""Generate visual summaries of the HCAHPS sentiment analysis findings.

Reads the CSV outputs produced by hcahps_sentiment_analysis.py and saves PNG charts
to the visuals/ folder. Run hcahps_sentiment_analysis.py first to (re)generate those CSVs.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

OUTPUT_DIR = 'visuals'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Sequential, colorblind-safe blue ramp (light -> dark = low -> high magnitude).
# A single-hue sequential ramp is the correct encoding for ranked/ordered data
# like these negative-rate rankings; a red/green diverging map implies a
# meaningful "good vs. bad" midpoint that isn't there, and red/green is the
# one pairing that collapses for the most common form of color blindness.
SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list(
    'sequential_blue', ['#cde2fb', '#6da7ec', '#2a78d6', '#104281']
)

# Validated categorical hues, used in fixed order across the report so the same
# color always means the same thing (never reused for "just another series").
BLUE = '#2a78d6'
ORANGE = '#eb6834'
AQUA = '#1baf7a'
TREND_LINE = '#52514e'  # muted ink, not a data color -- reads as annotation, not a series

# Human-readable labels for each HCAHPS measure_group code
MEASURE_GROUP_LABELS = {
    'H_CLEAN_HSP': 'Room Cleanliness',
    'H_COMP_1': 'Nurse Communication',
    'H_COMP_2': 'Doctor Communication',
    'H_COMP_5': 'Medicine Communication (Before Giving)',
    'H_COMP_6': 'Discharge Information',
    'H_DISCH_HELP': 'Discussed Help Needed After Discharge',
    'H_DOCTOR_EXPLAIN': 'Doctors Explained Clearly',
    'H_DOCTOR_LISTEN': 'Doctors Listened Carefully',
    'H_DOCTOR_RESPECT': 'Doctors\' Courtesy & Respect',
    'H_HSP_RATING': 'Overall Hospital Rating (0-10)',
    'H_MED_FOR': 'New Medication Purpose Explained',
    'H_NURSE_EXPLAIN': 'Nurses Explained Clearly',
    'H_NURSE_LISTEN': 'Nurses Listened Carefully',
    'H_NURSE_RESPECT': 'Nurses\' Courtesy & Respect',
    'H_QUIET_HSP': 'Quietness at Night',
    'H_RECMND': 'Would Recommend Hospital',
    'H_SIDE_EFFECTS': 'Medication Side Effects Explained',
    'H_SYMPTOMS': 'Symptoms Info Given at Discharge',
}


def _millions_formatter(value, _pos):
    return f'{value / 1_000_000:.0f}M' if value else '0'


def _add_trend_line(ax, x, y):
    """Overlay a linear fit so the correlation stated in the title is visible, not just asserted."""
    coeffs = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, np.polyval(coeffs, x_line), color=TREND_LINE, linewidth=2, linestyle='--', zorder=1)


def plot_state_negative_rate(summary):
    data = summary.sort_values('recommend_negative_rate_pct', ascending=True)
    fig_height = max(6, 0.22 * len(data))
    fig, ax = plt.subplots(figsize=(9, fig_height))

    rate = data['recommend_negative_rate_pct']
    colors = SEQUENTIAL_BLUE((rate - rate.min()) / (rate.max() - rate.min()))
    ax.barh(data['State'], rate, color=colors)
    ax.set_xlabel('% of patients who would NOT recommend the hospital')
    ax.set_title('"Would Not Recommend" Rate by State/Territory\n(single comparable HCAHPS measure, no cross-question blending)')
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'state_negative_rate.png'), dpi=150)
    plt.close(fig)


def plot_national_rate_by_measure(by_measure):
    national = (
        by_measure.groupby('measure_group')
        .agg(total=('total_estimated_responses', 'sum'), negative=('negative_estimated_responses', 'sum'))
        .reset_index()
    )
    national['negative_rate_pct'] = (national['negative'] / national['total'] * 100).round(2)
    national['label'] = national['measure_group'].map(MEASURE_GROUP_LABELS).fillna(national['measure_group'])
    national = national.sort_values('negative_rate_pct', ascending=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    rate = national['negative_rate_pct']
    colors = SEQUENTIAL_BLUE((rate - rate.min()) / (rate.max() - rate.min()))
    ax.barh(national['label'], rate, color=colors)
    ax.set_xlabel('National negative-answer rate (%)')
    ax.set_title('National Negative Rate by HCAHPS Question\n(each question aggregated separately -- not blended together)')
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'national_negative_rate_by_measure.png'), dpi=150)
    plt.close(fig)


def plot_population_bias_check(summary):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    x = summary['state_population']
    raw_y = summary['recommend_negative_estimated_responses']
    raw_corr = x.corr(raw_y)
    axes[0].scatter(x, raw_y, alpha=0.7, color=BLUE)
    _add_trend_line(axes[0], x, raw_y)
    axes[0].set_xlabel('State population')
    axes[0].set_ylabel('Raw negative response count')
    axes[0].set_title(f'Raw count vs. population\n(r = {raw_corr:.2f} -- mostly just measures state size)')
    axes[0].grid(linestyle='--', alpha=0.4)
    axes[0].xaxis.set_major_formatter(FuncFormatter(_millions_formatter))

    per_capita_y = summary['negative_per_million']
    per_capita_corr = x.corr(per_capita_y)
    axes[1].scatter(x, per_capita_y, alpha=0.7, color=ORANGE)
    _add_trend_line(axes[1], x, per_capita_y)
    axes[1].set_xlabel('State population')
    axes[1].set_ylabel('Negative responses per million residents')
    axes[1].set_title(f'Per-capita rate vs. population\n(r = {per_capita_corr:.2f} -- bias removed)')
    axes[1].grid(linestyle='--', alpha=0.4)
    axes[1].xaxis.set_major_formatter(FuncFormatter(_millions_formatter))

    fig.suptitle('Why Per-Capita Normalization Matters')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'population_bias_check.png'), dpi=150)
    plt.close(fig)


def plot_validation_against_cms_star_rating(summary):
    valid = summary.dropna(subset=['avg_cms_summary_star_rating'])
    x = valid['recommend_negative_rate_pct']
    y = valid['avg_cms_summary_star_rating']
    corr = x.corr(y)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(x, y, alpha=0.75, color=AQUA)
    _add_trend_line(ax, x, y)
    ax.set_xlabel('"Would not recommend" rate (%)')
    ax.set_ylabel("CMS official Summary Star Rating (avg. by state)")
    ax.set_title(f'Validation vs. CMS\'s Own Official Rating\n(r = {corr:.2f} -- strongly negative, as expected)')
    ax.grid(linestyle='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'validation_vs_cms_star_rating.png'), dpi=150)
    plt.close(fig)


def main():
    summary = pd.read_csv('HCAHPS-Hospital-state-negative-summary.csv')
    by_measure = pd.read_csv('HCAHPS-Hospital-state-negative-by-measure.csv')

    plot_state_negative_rate(summary)
    plot_national_rate_by_measure(by_measure)
    plot_population_bias_check(summary)
    plot_validation_against_cms_star_rating(summary)

    print(f'Saved 4 charts to {OUTPUT_DIR}/:')
    print('  - state_negative_rate.png')
    print('  - national_negative_rate_by_measure.png')
    print('  - population_bias_check.png')
    print('  - validation_vs_cms_star_rating.png')


if __name__ == '__main__':
    main()
