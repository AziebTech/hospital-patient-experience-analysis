"""Generate visual summaries of the HCAHPS sentiment analysis findings.

Reads the CSV outputs produced by hcahps_sentiment_analysis.py and saves PNG charts
to the visuals/ folder. Run hcahps_sentiment_analysis.py first to (re)generate those CSVs.
"""
import os

import matplotlib.pyplot as plt
import pandas as pd

OUTPUT_DIR = 'visuals'
os.makedirs(OUTPUT_DIR, exist_ok=True)

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


def plot_state_negative_rate(summary):
    data = summary.sort_values('recommend_negative_rate_pct', ascending=True)
    fig_height = max(6, 0.22 * len(data))
    fig, ax = plt.subplots(figsize=(9, fig_height))

    colors = plt.cm.RdYlGn_r((data['recommend_negative_rate_pct'] - data['recommend_negative_rate_pct'].min())
                              / (data['recommend_negative_rate_pct'].max() - data['recommend_negative_rate_pct'].min()))
    ax.barh(data['State'], data['recommend_negative_rate_pct'], color=colors)
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
    colors = plt.cm.RdYlGn_r((national['negative_rate_pct'] - national['negative_rate_pct'].min())
                              / (national['negative_rate_pct'].max() - national['negative_rate_pct'].min()))
    ax.barh(national['label'], national['negative_rate_pct'], color=colors)
    ax.set_xlabel('National negative-answer rate (%)')
    ax.set_title('National Negative Rate by HCAHPS Question\n(each question aggregated separately -- not blended together)')
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'national_negative_rate_by_measure.png'), dpi=150)
    plt.close(fig)


def plot_population_bias_check(summary):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    raw_corr = summary['state_population'].corr(summary['recommend_negative_estimated_responses'])
    axes[0].scatter(summary['state_population'], summary['recommend_negative_estimated_responses'], alpha=0.7)
    axes[0].set_xlabel('State population')
    axes[0].set_ylabel('Raw negative response count')
    axes[0].set_title(f'Raw count vs. population\n(r = {raw_corr:.2f} -- mostly just measures state size)')
    axes[0].grid(linestyle='--', alpha=0.4)

    per_capita_corr = summary['state_population'].corr(summary['negative_per_million'])
    axes[1].scatter(summary['state_population'], summary['negative_per_million'], alpha=0.7, color='seagreen')
    axes[1].set_xlabel('State population')
    axes[1].set_ylabel('Negative responses per million residents')
    axes[1].set_title(f'Per-capita rate vs. population\n(r = {per_capita_corr:.2f} -- bias removed)')
    axes[1].grid(linestyle='--', alpha=0.4)

    fig.suptitle('Why Per-Capita Normalization Matters')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'population_bias_check.png'), dpi=150)
    plt.close(fig)


def plot_validation_against_cms_star_rating(summary):
    valid = summary.dropna(subset=['avg_cms_summary_star_rating'])
    corr = valid['recommend_negative_rate_pct'].corr(valid['avg_cms_summary_star_rating'])

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(valid['recommend_negative_rate_pct'], valid['avg_cms_summary_star_rating'], alpha=0.75, color='darkorange')
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
