import pytest
from unittest.mock import MagicMock, patch
from reveriecore.enrichment import EnrichmentService


class TestDeviceConfig:
    """Tests for device configuration (cpu/cuda/mps/auto)."""
    
    def test_default_device_is_cpu(self):
        """Test that default device is 'cpu' when no config is provided."""
        with patch.dict('os.environ', {'REVERIE_SYNC_SERVICE': 'false'}):
            with patch('reveriecore.enrichment.load_reverie_config', return_value={}):
                service = EnrichmentService()
                assert service.device == "cpu"
    
    def test_device_from_config(self):
        """Test that device is read from config."""
        config = {
            "enrichment": {
                "classifier": {"device": "cuda"},
                "embedding": {"device": "cuda"},
                "summarization": {"device": "cuda"}
            }
        }
        with patch('reveriecore.enrichment.load_reverie_config', return_value=config):
            with patch('reveriecore.enrichment.AutoTokenizer'), \
                 patch('reveriecore.enrichment.AutoModelForSeq2SeqLM'), \
                 patch('reveriecore.enrichment.AutoModelForSequenceClassification'), \
                 patch('reveriecore.enrichment.SentenceTransformer'):
                service = EnrichmentService()
                assert service.device == "cuda"
    
    def test_device_from_kwargs_overrides_config(self):
        """Test that kwargs device overrides config device."""
        config = {
            "enrichment": {
                "classifier": {"device": "cuda"},
            }
        }
        with patch('reveriecore.enrichment.load_reverie_config', return_value=config):
            with patch('reveriecore.enrichment.AutoTokenizer'), \
                 patch('reveriecore.enrichment.AutoModelForSeq2SeqLM'), \
                 patch('reveriecore.enrichment.AutoModelForSequenceClassification'), \
                 patch('reveriecore.enrichment.SentenceTransformer'):
                service = EnrichmentService(device="mps")
                assert service.device == "mps"
    
    def test_invalid_device_falls_back_to_cpu(self):
        """Test that invalid device values fall back to cpu."""
        config = {
            "enrichment": {
                "classifier": {"device": "invalid_device"},
            }
        }
        with patch('reveriecore.enrichment.load_reverie_config', return_value=config):
            with patch('reveriecore.enrichment.AutoTokenizer'), \
                 patch('reveriecore.enrichment.AutoModelForSeq2SeqLM'), \
                 patch('reveriecore.enrichment.AutoModelForSequenceClassification'), \
                 patch('reveriecore.enrichment.SentenceTransformer'):
                service = EnrichmentService()
                assert service.device == "cpu"
    
    @patch('reveriecore.enrichment.torch.cuda.is_available', return_value=True)
    def test_auto_detects_cuda_when_available(self, mock_cuda):
        """Test that 'auto' detects CUDA when available."""
        config = {
            "enrichment": {
                "classifier": {"device": "auto"},
            }
        }
        with patch('reveriecore.enrichment.load_reverie_config', return_value=config):
            with patch('reveriecore.enrichment.AutoTokenizer'), \
                 patch('reveriecore.enrichment.AutoModelForSeq2SeqLM'), \
                 patch('reveriecore.enrichment.AutoModelForSequenceClassification'), \
                 patch('reveriecore.enrichment.SentenceTransformer'):
                service = EnrichmentService()
                assert service.device == "cuda"
    
    @patch('reveriecore.enrichment.torch.cuda.is_available', return_value=False)
    @patch('reveriecore.enrichment.torch.backends.mps.is_available', return_value=True)
    def test_auto_detects_mps_when_no_cuda(self, mock_mps, mock_cuda):
        """Test that 'auto' detects MPS when CUDA is not available."""
        config = {
            "enrichment": {
                "classifier": {"device": "auto"},
            }
        }
        with patch('reveriecore.enrichment.load_reverie_config', return_value=config):
            with patch('reveriecore.enrichment.AutoTokenizer'), \
                 patch('reveriecore.enrichment.AutoModelForSeq2SeqLM'), \
                 patch('reveriecore.enrichment.AutoModelForSequenceClassification'), \
                 patch('reveriecore.enrichment.SentenceTransformer'):
                service = EnrichmentService()
                assert service.device == "mps"
    
    @patch('reveriecore.enrichment.torch.cuda.is_available', return_value=False)
    @patch('reveriecore.enrichment.torch.backends.mps.is_available', return_value=False)
    def test_auto_falls_back_to_cpu_when_no_gpu(self, mock_mps, mock_cuda):
        """Test that 'auto' falls back to CPU when no GPU is available."""
        config = {
            "enrichment": {
                "classifier": {"device": "auto"},
            }
        }
        with patch('reveriecore.enrichment.load_reverie_config', return_value=config):
            with patch('reveriecore.enrichment.AutoTokenizer'), \
                 patch('reveriecore.enrichment.AutoModelForSeq2SeqLM'), \
                 patch('reveriecore.enrichment.AutoModelForSequenceClassification'), \
                 patch('reveriecore.enrichment.SentenceTransformer'):
                service = EnrichmentService()
                assert service.device == "cpu"
