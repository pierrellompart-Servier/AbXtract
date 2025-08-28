"""
Unit tests for AbXtract core functionality.

Run with: pytest tests/test_core_functionality.py -v
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

# Test data
TEST_HEAVY_SEQUENCE = (
    "QVQLVQSGAEVKKPGASVKVSCKASGGTFSSYAISWVRQAPGQGLEWMG"
    "GIIPIFGTANYAQKFQGRVTITADESTSTAYMELSSLRSEDTAVYYCAR"
    "SHYGLDYWGQGTLVTVSS"  # VH domain only for faster testing
)

TEST_LIGHT_SEQUENCE = (
    "DIQMTQSPSSLSASVGDRVTITCRASHSISSYLAWYQQKPGKAPKLLIY"
    "AASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPLTF"
    "GGGTKVEIK"  # VL domain only for faster testing
)


class TestConfiguration:
    """Test configuration system."""
    
    def test_default_config_creation(self):
        """Test creating default configuration."""
        from AbXtract import Config
        
        config = Config()
        assert config.numbering_scheme == 'imgt'
        assert config.pH == 7.0
        assert config.temperature == 25.0
        assert config.verbose == True
    
    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        from AbXtract import Config
        
        config_dict = {
            'pH': 7.4,
            'numbering_scheme': 'kabat',
            'verbose': False
        }
        config = Config.from_dict(config_dict)
        
        assert config.pH == 7.4
        assert config.numbering_scheme == 'kabat'
        assert config.verbose == False
        assert config.temperature == 25.0  # Should keep default
    
    def test_config_validation(self):
        """Test configuration validation."""
        from AbXtract import Config
        
        # Test invalid numbering scheme
        with pytest.raises(ValueError):
            Config(numbering_scheme='invalid_scheme')
        
        # Test invalid pH
        with pytest.raises(ValueError):
            Config(pH=15.0)
        
        # Test invalid n_jobs
        with pytest.raises(ValueError):
            Config(n_jobs=0)


class TestUtilities:
    """Test utility functions."""
    
    def test_sequence_validation(self):
        """Test sequence validation."""
        from AbXtract.utils import validate_sequence, is_valid_amino_acid_sequence
        
        # Valid sequence
        is_valid, msg = validate_sequence(TEST_HEAVY_SEQUENCE)
        assert is_valid == True
        assert msg == ""
        
        # Invalid sequence (contains numbers)
        is_valid, msg = validate_sequence("ACGT123")
        assert is_valid == False
        assert "Invalid characters" in msg
        
        # Test quick validation function
        assert is_valid_amino_acid_sequence(TEST_HEAVY_SEQUENCE) == True
        assert is_valid_amino_acid_sequence("ACGT123") == False
    
    def test_amino_acid_conversions(self):
        """Test amino acid code conversions."""
        from AbXtract.utils import three_to_one_letter, one_to_three_letter
        
        assert three_to_one_letter("ALA") == "A"
        assert three_to_one_letter("GLY") == "G"
        assert three_to_one_letter("INVALID") == "X"  # Unknown default
        
        assert one_to_three_letter("A") == "ALA"
        assert one_to_three_letter("G") == "GLY"
        assert one_to_three_letter("X") == "UNK"  # Unknown default
    
    def test_fasta_handling(self):
        """Test FASTA file reading and writing."""
        from AbXtract.utils import write_fasta, read_fasta
        
        test_sequences = {
            "seq1": "ACDEFGHIKLMNPQRSTVWY",
            "seq2": "QVQLVQSGAEVKKPGA"
        }
        
        with tempfile.NamedTemporaryFile(suffix='.fasta', delete=False) as tmp:
            tmp_path = Path(tmp.name)
        
        try:
            # Write FASTA
            write_fasta(test_sequences, tmp_path)
            assert tmp_path.exists()
            
            # Read FASTA
            loaded_sequences = read_fasta(tmp_path)
            assert len(loaded_sequences) == 2
            assert loaded_sequences["seq1"] == test_sequences["seq1"]
            assert loaded_sequences["seq2"] == test_sequences["seq2"]
        
        finally:
            tmp_path.unlink(missing_ok=True)


class TestSequenceAnalyzers:
    """Test sequence-based analyzers."""
    
    def test_liability_analyzer(self):
        """Test sequence liability analyzer."""
        from AbXtract.sequence import SequenceLiabilityAnalyzer
        
        analyzer = SequenceLiabilityAnalyzer()
        results = analyzer.analyze(
            heavy_sequence=TEST_HEAVY_SEQUENCE,
            light_sequence=TEST_LIGHT_SEQUENCE
        )
        
        assert isinstance(results, dict)
        assert 'count' in results
        assert 'liabilities' in results
        assert 'total_possible' in results
        assert isinstance(results['count'], int)
        assert results['count'] >= 0
    
    def test_bashour_calculator(self):
        """Test Bashour descriptor calculator."""
        from AbXtract.sequence import BashourDescriptorCalculator
        
        calc = BashourDescriptorCalculator()
        results = calc.calculate({
            'Heavy': TEST_HEAVY_SEQUENCE,
            'Light': TEST_LIGHT_SEQUENCE
        })
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 2  # Two sequences
        assert 'Molecular Weight' in results.columns
        assert 'pI' in results.columns
        assert 'Instability Index' in results.columns
        
        # Check that values are reasonable
        for _, row in results.iterrows():
            assert row['Molecular Weight'] > 0
            assert 0 < row['pI'] < 14
            assert row['Instability Index'] >= 0
    
    def test_peptide_calculator(self):
        """Test peptide descriptor calculator."""
        from AbXtract.sequence import PeptideDescriptorCalculator
        
        calc = PeptideDescriptorCalculator()
        results = calc.calculate_all(
            heavy_sequence=TEST_HEAVY_SEQUENCE,
            light_sequence=TEST_LIGHT_SEQUENCE
        )
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 2  # Two chains
        assert 'chain' in results.columns
        assert 'length' in results.columns
        assert 'molecular_weight' in results.columns
        
        # Check basic properties
        for _, row in results.iterrows():
            assert row['length'] > 0
            assert row['molecular_weight'] > 0
            assert row['chain'] in ['Heavy', 'Light']


class TestMainCalculator:
    """Test the main AntibodyDescriptorCalculator."""
    
    def test_calculator_initialization(self):
        """Test calculator initialization."""
        from AbXtract import AntibodyDescriptorCalculator, Config
        
        # Default initialization
        calc = AntibodyDescriptorCalculator()
        assert hasattr(calc, 'config')
        assert hasattr(calc, 'results')
        assert hasattr(calc, 'metadata')
        
        # Custom config initialization
        config = Config(pH=7.4, verbose=False)
        calc = AntibodyDescriptorCalculator(config=config)
        assert calc.config.pH == 7.4
        assert calc.config.verbose == False
    
    def test_sequence_descriptors(self):
        """Test sequence descriptor calculation."""
        from AbXtract import AntibodyDescriptorCalculator, Config
        
        # Use minimal config to avoid external tool dependencies
        config = Config.from_dict({
            'calculate_dssp': False,
            'calculate_propka': False,
            'calculate_arpeggio': False,
            'verbose': False
        })
        
        calc = AntibodyDescriptorCalculator(config=config)
        results = calc.calculate_sequence_descriptors(
            heavy_sequence=TEST_HEAVY_SEQUENCE,
            light_sequence=TEST_LIGHT_SEQUENCE,
            sequence_id="TestAb"
        )
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 1
        assert 'SeqID' in results.columns
        assert results['SeqID'].iloc[0] == "TestAb"
        
        # Check that we got some descriptors
        assert len(results.columns) > 10
    
    def test_input_validation(self):
        """Test input validation."""
        from AbXtract import AntibodyDescriptorCalculator, Config
        
        config = Config(verbose=False)
        calc = AntibodyDescriptorCalculator(config=config)
        
        # Test that no input raises error
        with pytest.raises(ValueError):
            calc.calculate_sequence_descriptors()
        
        # Test invalid sequence
        with pytest.raises(ValueError):
            calc.calculate_sequence_descriptors(heavy_sequence="123INVALID")
    
    def test_get_summary(self):
        """Test results summary generation."""
        from AbXtract import AntibodyDescriptorCalculator, Config
        
        config = Config(verbose=False)
        calc = AntibodyDescriptorCalculator(config=config)
        
        # Calculate some results first
        calc.calculate_sequence_descriptors(
            heavy_sequence=TEST_HEAVY_SEQUENCE,
            light_sequence=TEST_LIGHT_SEQUENCE,
            sequence_id="TestAb"
        )
        
        summary = calc.get_summary()
        assert isinstance(summary, dict)
        assert 'timestamp' in summary
        assert 'analyses_performed' in summary
        assert 'sequence' in summary['analyses_performed']


class TestIntegration:
    """Integration tests."""
    
    def test_full_sequence_pipeline(self):
        """Test complete sequence analysis pipeline."""
        from AbXtract import AntibodyDescriptorCalculator, Config
        
        # Use safe configuration
        config = Config.from_dict({
            'calculate_dssp': False,
            'calculate_propka': False,
            'calculate_arpeggio': False,
            'verbose': False
        })
        
        calc = AntibodyDescriptorCalculator(config=config)
        
        # Run complete sequence analysis
        results = calc.calculate_all(
            heavy_sequence=TEST_HEAVY_SEQUENCE,
            light_sequence=TEST_LIGHT_SEQUENCE,
            sample_id="IntegrationTest"
        )
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 1
        assert 'SampleID' in results.columns
        assert results['SampleID'].iloc[0] == "IntegrationTest"
        
        # Check for key descriptor categories
        result_row = results.iloc[0]
        
        # Should have basic sequence properties
        assert pd.notna(result_row.get('Heavy_Length')) or pd.notna(result_row.get('Light_Length'))
        
        # Should have some liability analysis
        assert 'Total_Liabilities' in results.columns or any('liability' in col.lower() for col in results.columns)
    
    def test_error_handling(self):
        """Test error handling in various scenarios."""
        from AbXtract import AntibodyDescriptorCalculator, Config
        
        config = Config(verbose=False)
        calc = AntibodyDescriptorCalculator(config=config)
        
        # Test empty sequence
        with pytest.raises(ValueError):
            calc.calculate_sequence_descriptors(heavy_sequence="")
        
        # Test very short sequence
        with pytest.raises(ValueError):
            calc.calculate_sequence_descriptors(heavy_sequence="ACDE")  # Too short
        
        # Test sequence with invalid characters
        with pytest.raises(ValueError):
            calc.calculate_sequence_descriptors(heavy_sequence="ACDEFGHIKLMNPQRSTVWY123XYZ!")


class TestFileOperations:
    """Test file I/O operations."""
    
    def test_results_export(self):
        """Test exporting results to different formats."""
        from AbXtract import AntibodyDescriptorCalculator, Config
        
        # Create some test results
        config = Config(verbose=False)
        calc = AntibodyDescriptorCalculator(config=config)
        results = calc.calculate_sequence_descriptors(
            heavy_sequence=TEST_HEAVY_SEQUENCE,
            light_sequence=TEST_LIGHT_SEQUENCE,
            sequence_id="ExportTest"
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Test CSV export
            csv_path = tmpdir / "test_results.csv"
            calc.save_results(str(csv_path), format='csv')
            assert csv_path.exists()
            
            # Verify we can read it back
            loaded_df = pd.read_csv(csv_path)
            assert len(loaded_df) == 1
            assert loaded_df['SeqID'].iloc[0] == "ExportTest"
            
            # Test JSON export
            json_path = tmpdir / "test_results.json"
            calc.save_results(str(json_path), format='json')
            assert json_path.exists()


# Test fixtures and utilities
@pytest.fixture
def sample_calculator():
    """Fixture providing a configured calculator."""
    from AbXtract import AntibodyDescriptorCalculator, Config
    
    config = Config.from_dict({
        'calculate_dssp': False,
        'calculate_propka': False,  
        'calculate_arpeggio': False,
        'verbose': False
    })
    return AntibodyDescriptorCalculator(config=config)


@pytest.fixture
def sample_sequences():
    """Fixture providing test sequences."""
    return {
        'heavy': TEST_HEAVY_SEQUENCE,
        'light': TEST_LIGHT_SEQUENCE
    }


def test_with_fixtures(sample_calculator, sample_sequences):
    """Test using fixtures."""
    results = sample_calculator.calculate_sequence_descriptors(
        heavy_sequence=sample_sequences['heavy'],
        light_sequence=sample_sequences['light'],
        sequence_id="FixtureTest"
    )
    
    assert not results.empty
    assert 'SeqID' in results.columns
    assert results['SeqID'].iloc[0] == "FixtureTest"


# Performance tests
@pytest.mark.slow
def test_performance_large_sequence():
    """Test performance with larger sequences."""
    from AbXtract import AntibodyDescriptorCalculator, Config
    
    # Create larger test sequence
    large_heavy = TEST_HEAVY_SEQUENCE * 10  # Much longer sequence
    
    config = Config(verbose=False)
    calc = AntibodyDescriptorCalculator(config=config)
    
    import time
    start_time = time.time()
    
    results = calc.calculate_sequence_descriptors(
        heavy_sequence=large_heavy,
        sequence_id="PerformanceTest"
    )
    
    end_time = time.time()
    calculation_time = end_time - start_time
    
    assert not results.empty
    # Should complete within reasonable time (adjust threshold as needed)
    assert calculation_time < 30.0  # 30 seconds max
    
    print(f"Performance test completed in {calculation_time:.2f} seconds")


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])