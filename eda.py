"""Felix Qu and Nathan Zhao: economic development and health EDA.

Run from the project folder. Read local OWID CSVs and save results there.
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.stats.weightstats as sm


def clean_data(data, indicator, name):
    """Return unique country/territory rows for 2000-2020.

    Keep three-letter codes. Fail on duplicate country-years rather than
    silently choosing one conflicting observation. Do not impute values.
    """
    country_code = data['Code'].str.len() == 3
    in_period = (data['Year'] >= 2000) & (data['Year'] <= 2020)
    result = data[country_code & in_period]
    result = result[['Entity', 'Code', 'Year', indicator]].copy()
    result.columns = ['Entity', 'Code', 'Year', name]
    counts = result.groupby(['Code', 'Year']).size()
    assert counts.max() <= 1
    return result


def join_data(left, right, name):
    """Inner-join a second indicator using country code and year."""
    return left.merge(right[['Code', 'Year', name]], on=['Code', 'Year'])


def changes(data):
    """Return complete 2000/2020 country pairs and endpoint differences."""
    start = data[data['Year'] == 2000]
    end = data[data['Year'] == 2020]
    result = start.merge(end, on='Code', suffixes=('_2000', '_2020'))
    result = result.dropna()
    result['gdp_change_pct'] = 100 * (
        result['gdp_pc_2020'] / result['gdp_pc_2000'] - 1)
    result['life_change'] = result['life_exp_2020'] - result['life_exp_2000']
    result['mortality_change'] = (
        result['child_mortality_2020'] - result['child_mortality_2000'])
    return result


def summarize(data, columns, name):
    """Save size, per-column missingness and seven-number summaries."""
    print(name, data.shape)
    print(data.isna().sum())
    data[columns].describe().to_csv(name + '_summary.csv')
    data.isna().sum().to_csv(name + '_missing.csv')


def scatter(data, x, y, xlabel, ylabel, title, filename):
    """Save a labeled scatterplot with one point per country."""
    sns.relplot(data=data, x=x, y=y, height=3.2, aspect=1.65)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=180)
    plt.close('all')


def mean_change_test(values):
    """Test a zero mean change under the independent-country model."""
    return sm.ztest(values, value=0)


def validate_changes(data):
    """Save conditional mean-change tests and diagnostic histograms.

    These tests do not test GDP-health associations or causation.
    Country independence cannot be established from these data.
    """
    results = []
    for column in ['life_change', 'mortality_change']:
        values = data[column]
        assert values.count() == len(values)
        assert values.std() > 0
        statistic, pvalue = mean_change_test(values)
        results.append([column, len(values), values.mean(),
                        statistic, pvalue, 0.05 / 2])
        sns.histplot(data=data, x=column)
        plt.xlabel(column + ' (2020 minus 2000)')
        plt.title('Distribution of within-country changes')
        plt.tight_layout()
        plt.savefig(column + '_diagnostic.png', dpi=180)
        plt.close('all')
    result = pd.DataFrame(results, columns=[
        'variable', 'n', 'mean_change', 'z', 'p', 'threshold'])
    result.to_csv('validity.csv', index=False)
    return result


def main():
    """Load, check, merge, summarize and graph all research datasets."""
    sources = ['gdp-per-capita-worldbank', 'life-expectancy',
               'child-mortality']
    names = ['gdp_pc', 'life_exp', 'child_mortality']
    cleaned = []
    for source, name in zip(sources, names):
        raw = pd.read_csv('data/' + source + '.csv')
        print(source, 'raw shape:', raw.shape)
        print('Raw missing values:', raw.isna().sum())
        data = clean_data(raw, raw.columns[3], name)
        summarize(data, [name], name)
        cleaned.append(data)
    gdp, life, mortality = cleaned
    gdp = gdp[gdp['gdp_pc'] > 0]
    gl = join_data(gdp, life, 'life_exp')
    gm = join_data(gdp, mortality, 'child_mortality')
    panel = join_data(gl, mortality, 'child_mortality')
    tables = [gl, gm, panel]
    table_names = ['gdp_life', 'gdp_mortality', 'all_three']
    for table, name in zip(tables, table_names):
        columns = list(table.columns[3:])
        summarize(table, columns, name + '_panel')
        table.to_csv(name + '_panel.csv', index=False)
    gl = gl[gl['Year'] == 2020].dropna()
    gm = gm[gm['Year'] == 2020].dropna()
    delta = changes(panel)
    summarize(gl, ['gdp_pc', 'life_exp'], 'gdp_life_2020')
    summarize(gm, ['gdp_pc', 'child_mortality'], 'gdp_mortality_2020')
    summarize(delta, ['gdp_change_pct', 'life_change', 'mortality_change'],
              'changes')
    delta.to_csv('changes.csv', index=False)
    validate_changes(delta)
    scatter(gl, 'gdp_pc', 'life_exp', 'GDP per capita (2021 int. $)',
            'Life expectancy (years)', 'GDP and life expectancy, 2020',
            'figure1.png')
    sns.histplot(data=gl, x='life_exp')
    plt.xlabel('Life expectancy (years)')
    plt.title('Life expectancy distribution, GDP-matched countries, 2020')
    plt.tight_layout()
    plt.savefig('figure2.png', dpi=180)
    plt.close('all')
    scatter(gm, 'gdp_pc', 'child_mortality', 'GDP per capita (2021 int. $)',
            'Deaths before age 5 per 100 live births',
            'GDP and child mortality, 2020', 'figure3.png')
    sns.histplot(data=gm, x='child_mortality')
    plt.xlabel('Deaths before age 5 per 100 live births')
    plt.title('Child mortality distribution, GDP-matched countries, 2020')
    plt.tight_layout()
    plt.savefig('figure4.png', dpi=180)
    plt.close('all')
    scatter(delta, 'gdp_change_pct', 'life_change',
            'GDP per capita change (%)',
            'Life expectancy change (years)', 'GDP and longevity changes',
            'figure5.png')
    scatter(delta, 'gdp_change_pct', 'mortality_change',
            'GDP per capita change (%)', 'Child mortality change (points)',
            'GDP and mortality changes, 2000-2020', 'figure6.png')


if __name__ == '__main__':
    main()
