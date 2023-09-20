import pytest
from facerec.siamese_models import get_base_model, get_siamese_model


# Define a fixture for the input_shape
@pytest.fixture
def input_shape():
    return (150, 150, 3)


# Test get_base_model function
@pytest.mark.parametrize("base_architecture", ['ResNet50', 'VGG16', 'InceptionV3'])
def test_get_base_model_supported_architectures(input_shape, base_architecture):
    model = get_base_model(base_architecture, input_shape)
    assert model is not None


def test_get_base_model_unsupported_architecture(input_shape):
    with pytest.raises(ValueError):
        get_base_model('InvalidArchitecture', input_shape)


# Test get_siamese_model function
@pytest.mark.parametrize("base_architecture", ['ResNet50', 'VGG16', 'InceptionV3'])
def test_get_siamese_model_supported_architectures(input_shape, base_architecture):
    model = get_siamese_model(base_architecture, input_shape)
    assert model is not None


def test_get_siamese_model_unsupported_architecture(input_shape):
    with pytest.raises(ValueError):
        get_siamese_model('InvalidArchitecture', input_shape)


# Run the tests
if __name__ == "__main__":
    pytest.main()
