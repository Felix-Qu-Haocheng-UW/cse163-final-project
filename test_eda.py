"""Test EDA calculations using small, hand-checkable data."""
import pandas as pd
from eda import clean_data, join_data, changes, mean_change_test


def test_clean():
    """Check years, aggregate removal, and missing values."""
    raw = pd.DataFrame({
        'Entity': ['A', 'A', 'World', 'Unknown', 'B', 'C'],
        'Code': ['AAA', 'AAA', 'OWID_WRL', None, 'BBB', 'CCC'],
        'Year': [1999, 2000, 2000, 2000, 2020, 2021],
        'value': [1, 2, 3, 4, None, 6]})
    result = clean_data(raw, 'value', 'gdp_pc')
    assert list(result['Code']) == ['AAA', 'BBB']
    assert result['gdp_pc'].isna().sum() == 1
    assert result.loc[1, 'gdp_pc'] == 2


def test_join_and_changes():
    """Check country-year matching, missing endpoints and change signs."""
    gdp = pd.DataFrame({
        'Entity': ['A', 'A', 'B', 'C'],
        'Code': ['AAA', 'AAA', 'BBB', 'CCC'],
        'Year': [2000, 2020, 2020, 2000],
        'gdp_pc': [100, 150, 200, 300]})
    health = pd.DataFrame({
        'Code': ['AAA', 'AAA', 'BBB', 'DDD'],
        'Year': [2000, 2020, 2020, 2000],
        'life_exp': [60, 65, 70, 80],
        'child_mortality': [10, 4, 2, 1]})
    joined = join_data(gdp, health, 'life_exp')
    joined = join_data(joined, health, 'child_mortality')
    assert len(joined) == 3
    result = changes(joined)
    assert len(result) == 1
    assert result.loc[0]['gdp_change_pct'] == 50
    assert result.loc[0]['life_change'] == 5
    assert result.loc[0]['mortality_change'] == -6
    joined.loc[joined['Year'] == 2000, 'life_exp'] = None
    assert len(changes(joined)) == 0


def test_summary():
    """Check the summary method against a known numerical example."""
    summary = pd.Series([1, 2, 3, 4, 5]).describe()
    assert summary['mean'] == 3
    assert summary['min'] == 1
    assert summary['25%'] == 2
    assert summary['50%'] == 3
    assert summary['75%'] == 4
    assert summary['max'] == 5
    assert abs(summary['std'] - 2.5 ** 0.5) < 0.000001


def test_mean_change_test():
    """Check zero and nonzero mean cases against a hand calculation."""
    values = pd.Series([-2, -1, 1, 2] * 10)
    statistic, pvalue = mean_change_test(values)
    assert abs(statistic) < 0.000001
    assert abs(pvalue - 1) < 0.000001
    shifted = values + 1
    statistic, pvalue = mean_change_test(shifted)
    expected = shifted.mean() / (shifted.std() / len(shifted) ** 0.5)
    assert abs(statistic - expected) < 0.000001
    assert 0 < pvalue < 0.05


def main():
    """Run all independent checks."""
    test_clean()
    test_join_and_changes()
    test_summary()
    test_mean_change_test()
    print('All tests passed.')


if __name__ == '__main__':
    main()
