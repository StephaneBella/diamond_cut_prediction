def test_load_raw_data():
    from src.data_loader import load_raw_data
    df = load_raw_data("data/raw/diamonds.csv")
    assert not df.empty