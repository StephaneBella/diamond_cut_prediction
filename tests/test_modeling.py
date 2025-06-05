from src.Modelling import initialize_models

def test_initialize_models():
    models = initialize_models()
    assert 'DecisionTree' in models
    assert 'ANN' in models
    for model in models.values():
        assert model is not None
