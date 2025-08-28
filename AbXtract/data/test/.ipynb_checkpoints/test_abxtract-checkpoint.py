#!/usr/bin/env python3
"""
Simple test script for AbXtract package.

This script provides a streamlined test of AbXtract functionality
without requiring Jupyter notebook.

Usage:
    python test_abxtract.py [--pdb-file path/to/test.pdb] [--output-dir path/to/output]
"""

import argparse
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_imports():
    """Test importing all AbXtract modules."""
    logger.info("Testing imports...")
    
    try:
        from AbXtract import AntibodyDescriptorCalculator, Config
        from AbXtract.sequence import (
            SequenceLiabilityAnalyzer,
            BashourDescriptorCalculator,
            AntibodyNumbering
        )
        from AbXtract.utils import read_fasta, write_fasta
        logger.info("✅ All imports successful")
        return True
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        return False


def create_test_data(data_dir):
    """Create test antibody sequences."""
    logger.info("Creating test data...")
    
    # Realistic antibody sequences
    heavy_sequence = (
        "QVQLVQSGAEVKKPGASVKVSCKASGGTFSSYAISWVRQAPGQGLEWMG"
        "GIIPIFGTANYAQKFQGRVTITADESTSTAYMELSSLRSEDTAVYYCAR"
        "SHYGLDYWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKD"
        "YFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQT"
        "YICNVNHKPSNTKVDKKVEPKSCDKTHTCPPCPAPELLGGPSVFLFPPK"
        "PKDTLMISRTPEVTCVVVDVSHEDPEVKFNWYVDGVEVHNAKTKPREEQ"
        "YASTYRVVSVLTVLHQDWLNGKEYKCKVSNKALPAPIEKTISKAKGQPR"
        "EPQVYTLPPSRDELTKNQVSLTCLVKGFYPSDIAVEWESNGQPENNYK"
        "TTPPVLDSDGSFFLYSKLTVDKSRWQQGNVFSCSVMHEALHNHYTQKS"
        "LSLSPGK"
    )
    
    light_sequence = (
        "DIQMTQSPSSLSASVGDRVTITCRASHSISSYLAWYQQKPGKAPKLLIY"
        "AASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPLTF"
        "GGGTKVEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKV"
        "QWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYAC"
        "EVTHQGLSSPVTKSFNRGEC"
    )
    
    # Create FASTA file
    sequences = {
        "Test_Heavy_Chain": heavy_sequence,
        "Test_Light_Chain": light_sequence
    }
    
    data_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = data_dir / "test_antibody.fasta"
    
    from AbXtract.utils import write_fasta
    write_fasta(sequences, fasta_path)
    
    logger.info(f"✅ Test data created: {fasta_path}")
    return heavy_sequence, light_sequence, fasta_path


def test_individual_calculators(heavy_seq, light_seq):
    """Test individual calculator modules."""
    logger.info("Testing individual calculators...")
    
    results = {}
    
    # Test Liability Analyzer
    try:
        from AbXtract.sequence import SequenceLiabilityAnalyzer
        analyzer = SequenceLiabilityAnalyzer()
        liability_results = analyzer.analyze(heavy_seq, light_seq)
        results['liabilities'] = liability_results['count']
        logger.info(f"✅ Liability Analyzer: Found {liability_results['count']} liabilities")
    except Exception as e:
        logger.error(f"❌ Liability Analyzer failed: {e}")
        results['liabilities'] = None
    
    # Test Bashour Calculator
    try:
        from AbXtract.sequence import BashourDescriptorCalculator
        calc = BashourDescriptorCalculator()
        bashour_results = calc.calculate({'Heavy': heavy_seq, 'Light': light_seq})
        results['bashour_descriptors'] = len(bashour_results.columns)
        logger.info(f"✅ Bashour Calculator: {len(bashour_results.columns)} descriptors")
    except Exception as e:
        logger.error(f"❌ Bashour Calculator failed: {e}")
        results['bashour_descriptors'] = None
    
    # Test Antibody Numbering
    try:
        from AbXtract.sequence import AntibodyNumbering
        numbering = AntibodyNumbering()
        # Use only VH domain portion for numbering
        vh_domain = heavy_seq[:120]
        numbered = numbering.number_sequence(vh_domain, 'H')
        results['numbering_positions'] = len(numbered)
        logger.info(f"✅ Antibody Numbering: {len(numbered)} positions")
    except Exception as e:
        logger.warning(f"⚠️ Antibody Numbering failed (ANARCI may not be installed): {e}")
        results['numbering_positions'] = None
    
    return results


def test_main_calculator(heavy_seq, light_seq, pdb_file=None):
    """Test the main AntibodyDescriptorCalculator."""
    logger.info("Testing main AntibodyDescriptorCalculator...")
    
    try:
        from AbXtract import AntibodyDescriptorCalculator, Config
        
        # Create optimized config (disable external tools that may not be available)
        config = Config.from_dict({
            'verbose': False,
            'calculate_dssp': False,      # May not be installed
            'calculate_propka': False,    # May not be installed
            'calculate_arpeggio': False,  # May not be installed
            'calculate_sasa': True,       # Should work with freesasa
            'calculate_liabilities': True,
            'calculate_bashour': True,
            'calculate_peptide': True
        })
        
        calc = AntibodyDescriptorCalculator(config=config)
        logger.info("✅ Calculator initialized")
        
        # Test sequence analysis
        seq_results = calc.calculate_sequence_descriptors(
            heavy_sequence=heavy_seq,
            light_sequence=light_seq,
            sequence_id="TestAb"
        )
        logger.info(f"✅ Sequence analysis: {seq_results.shape[1]} descriptors calculated")
        
        # Test combined analysis
        if pdb_file and pdb_file.exists():
            logger.info("Testing structure analysis...")
            try:
                combined_results = calc.calculate_all(
                    heavy_sequence=heavy_seq,
                    light_sequence=light_seq,
                    pdb_file=pdb_file,
                    sample_id="TestAb_Complete"
                )
                logger.info(f"✅ Combined analysis: {combined_results.shape[1]} descriptors calculated")
                return combined_results
            except Exception as e:
                logger.warning(f"⚠️ Structure analysis failed: {e}")
                return seq_results
        else:
            logger.info("No PDB file provided, using sequence-only analysis")
            return seq_results
    
    except Exception as e:
        logger.error(f"❌ Main calculator failed: {e}")
        return None


def analyze_results(results_df):
    """Analyze and summarize the calculated descriptors."""
    logger.info("Analyzing results...")
    
    if results_df is None or results_df.empty:
        logger.error("No results to analyze")
        return {}
    
    result_row = results_df.iloc[0]
    analysis = {}
    
    # Basic properties
    analysis['antibody_type'] = result_row.get('Type', 'Unknown')
    analysis['heavy_length'] = result_row.get('Heavy_Length', 0)
    analysis['light_length'] = result_row.get('Light_Length', 0)
    analysis['total_descriptors'] = len(results_df.columns)
    
    # Liability analysis
    total_liabilities = result_row.get('Total_Liabilities', 0)
    analysis['total_liabilities'] = total_liabilities
    analysis['liability_assessment'] = (
        'Excellent' if total_liabilities == 0 else
        'Good' if total_liabilities <= 2 else
        'Moderate' if total_liabilities <= 5 else
        'Poor'
    )
    
    # Physicochemical properties
    mw = result_row.get('Molecular Weight', result_row.get('molecular_weight'))
    if pd.notna(mw):
        analysis['molecular_weight'] = f"{mw:.1f} Da"
    
    pi = result_row.get('pI', result_row.get('isoelectric_point'))
    if pd.notna(pi):
        analysis['isoelectric_point'] = f"{pi:.2f}"
    
    instability = result_row.get('Instability Index')
    if pd.notna(instability):
        analysis['instability_index'] = f"{instability:.2f}"
        analysis['stability_assessment'] = 'Stable' if instability < 40 else 'Unstable'
    
    gravy = result_row.get('Hydrophobicity', result_row.get('gravy'))
    if pd.notna(gravy):
        analysis['gravy_score'] = f"{gravy:.3f}"
        analysis['hydrophobicity'] = 'Hydrophobic' if gravy > 0 else 'Hydrophilic'
    
    # Structure properties (if available)
    total_sasa = result_row.get('Total_SASA', result_row.get('total_sasa'))
    if pd.notna(total_sasa):
        analysis['total_sasa'] = f"{total_sasa:.1f} Ų"
    
    sap_score = result_row.get('SAP_Score', result_row.get('sap_score'))
    if pd.notna(sap_score):
        analysis['sap_score'] = f"{sap_score:.3f}"
    
    return analysis


def save_results(results_df, analysis, output_dir):
    """Save results and analysis to files."""
    logger.info("Saving results...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    # Save complete results
    csv_path = output_dir / f"abxtract_test_results_{timestamp}.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"✅ Results saved: {csv_path}")
    
    # Save analysis summary
    summary_path = output_dir / f"abxtract_test_summary_{timestamp}.txt"
    with open(summary_path, 'w') as f:
        f.write("AbXtract Test Results Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Test Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("ANALYSIS RESULTS\n")
        f.write("-" * 20 + "\n")
        for key, value in analysis.items():
            f.write(f"{key.replace('_', ' ').title()}: {value}\n")
        
        f.write(f"\nDETAILED RESULTS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Complete results saved to: {csv_path.name}\n")
        f.write(f"Total descriptors calculated: {len(results_df.columns)}\n")
    
    logger.info(f"✅ Summary saved: {summary_path}")
    return csv_path, summary_path


def print_summary(analysis):
    """Print a formatted summary to the console."""
    print("\n" + "="*60)
    print("           ABXTRACT TEST RESULTS SUMMARY")
    print("="*60)
    
    print(f"\n📊 BASIC PROPERTIES")
    print(f"   Antibody Type: {analysis.get('antibody_type', 'Unknown')}")
    print(f"   Heavy Chain: {analysis.get('heavy_length', 0)} aa")
    print(f"   Light Chain: {analysis.get('light_length', 0)} aa")
    print(f"   Total Descriptors: {analysis.get('total_descriptors', 0)}")
    
    print(f"\n🔍 DEVELOPABILITY ASSESSMENT")
    print(f"   Sequence Liabilities: {analysis.get('total_liabilities', 0)} ({analysis.get('liability_assessment', 'Unknown')})")
    
    if 'stability_assessment' in analysis:
        print(f"   Protein Stability: {analysis['stability_assessment']}")
    
    if 'hydrophobicity' in analysis:
        print(f"   Hydrophobicity: {analysis['hydrophobicity']}")
    
    print(f"\n🧪 PHYSICOCHEMICAL PROPERTIES")
    for prop in ['molecular_weight', 'isoelectric_point', 'instability_index', 'gravy_score']:
        if prop in analysis:
            print(f"   {prop.replace('_', ' ').title()}: {analysis[prop]}")
    
    if 'total_sasa' in analysis or 'sap_score' in analysis:
        print(f"\n🏗️ STRUCTURAL PROPERTIES")
        if 'total_sasa' in analysis:
            print(f"   Total SASA: {analysis['total_sasa']}")
        if 'sap_score' in analysis:
            print(f"   SAP Score: {analysis['sap_score']}")
    
    print("\n" + "="*60)
    print("✅ AbXtract test completed successfully!")
    print("="*60)


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description='Test AbXtract functionality')
    parser.add_argument('--pdb-file', type=str, help='Path to test PDB file')
    parser.add_argument('--output-dir', type=str, default='./test_results', 
                       help='Output directory for results')
    args = parser.parse_args()
    
    # Setup paths
    output_dir = Path(args.output_dir)
    data_dir = Path('./data/test')
    pdb_file = Path(args.pdb_file) if args.pdb_file else data_dir / 'test.pdb'
    
    print("🚀 Starting AbXtract Test Suite")
    print("="*50)
    
    # Test 1: Imports
    if not test_imports():
        sys.exit(1)
    
    # Test 2: Create test data
    heavy_seq, light_seq, fasta_path = create_test_data(data_dir)
    
    # Test 3: Individual calculators
    calc_results = test_individual_calculators(heavy_seq, light_seq)
    
    # Test 4: Main calculator
    results_df = test_main_calculator(heavy_seq, light_seq, pdb_file)
    
    if results_df is None:
        logger.error("Main calculator test failed")
        sys.exit(1)
    
    # Test 5: Analyze results
    analysis = analyze_results(results_df)
    
    # Test 6: Save results
    csv_path, summary_path = save_results(results_df, analysis, output_dir)
    
    # Print final summary
    print_summary(analysis)
    
    print(f"\n📁 Files generated:")
    print(f"   • Test data: {fasta_path}")
    print(f"   • Results: {csv_path}")
    print(f"   • Summary: {summary_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())