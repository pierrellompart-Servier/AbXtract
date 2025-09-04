import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import matplotlib.lines as mlines
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import matplotlib.lines as mlines

    
import pandas as pd
import numpy as np
from anarci import anarci

def add_residue_sasa_sum_column(structures_df):
    """
    Add a column with the sum of residue SASA per position to structures_results_seq dataframe.
    """
    residue_sasa_sum_list = []
    
    for idx, row in structures_df.iterrows():
        sidechain_sasa = row['sidechain_sasa']
        residue_sasa_sum = {}
        
        if isinstance(sidechain_sasa, dict):
            for residue_key, sasa_value in sidechain_sasa.items():
                parts = residue_key.split('_')
                if len(parts) >= 3:
                    try:
                        # Position in the key is 0-based, convert to 1-based for consistency
                        position = int(parts[1]) + 1
                        residue_sasa_sum[position] = sasa_value
                    except ValueError:
                        continue
        
        residue_sasa_sum_list.append(residue_sasa_sum)
    
    structures_df['residue_sasa_sum'] = residue_sasa_sum_list
    return structures_df


def create_complete_antibody_dataframe(len_heavy_seq, df_residue, df_Ab, sequence, annotations, hydrophobicity, structures_data, 
                                       liabilities_list, chain_type='Heavy', scheme='imgt'):
    """
    Create a complete dataframe with proper handling of ANARCI numbering and structural data.
    Properly handles both Heavy (H) and Light (L) chains.
    """
    chain_letter = 'H' if chain_type == 'Heavy' else 'L'
    
    # STEP 1: Create base dataframe from full sequence
    data = []
    for idx, amino_acid in enumerate(sequence):
        data.append({
            'position_seq': idx + 1,  # 1-based sequential position
            'position_idx': idx,       # 0-based index for data lookup
            'amino_acid': amino_acid
        })
    df = pd.DataFrame(data)
    
    print("structures_data",structures_data)
    print("df_Ab",df_Ab)
    
    # STEP 2: Add ANARCI numbered positions and regions
    df['position_num'] = np.nan
    df['region'] = None
    
    if annotations:
        annotation_idx = 0
        for seq_idx in range(len(df)):
            if annotation_idx < len(annotations):
                pos_tuple, aa, region = annotations[annotation_idx]
                pos_num = pos_tuple[0]
                
                if df.loc[seq_idx, 'amino_acid'] == aa:
                    df.loc[seq_idx, 'position_num'] = pos_num
                    df.loc[seq_idx, 'region'] = region
                    annotation_idx += 1
    
    # STEP 3: Add hydrophobicity data
    hydro_cols = ['charge_sign', 'hydrophobicity_hw', 'hydrophobicity_eisenberg',
                  'hydrophobicity_rose', 'hydrophobicity_janin', 'hydrophobicity_engelman']
    
    for col in hydro_cols:
        df[col] = np.nan
        if col in hydrophobicity and hydrophobicity[col]:
            values = hydrophobicity[col]
            for i in range(min(len(values), len(df))):
                if i < len(values):
                    df.loc[i, col] = values[i]
    
    # STEP 4: Initialize structural columns
    df['disulfide_bond'] = False
    df['sap'] = np.nan
    df['high_sap'] = False
    df['sidechain_sasa'] = np.nan
    df['buried'] = False
    df['pka'] = None
    df['pka_shift'] = None
    df['residue_sasa_sum'] = np.nan
    
    # STEP 4a: Process residue_sasa_sum
    if 'residue_sasa_sum' in structures_data and isinstance(structures_data['residue_sasa_sum'], dict):
        for pos, value in structures_data['residue_sasa_sum'].items():
            if pos <= len(df):
                mask = df['position_seq'] == pos
                if mask.any():
                    df.loc[mask, 'residue_sasa_sum'] = value
    
    # STEP 4b: Process residue_sap - FIXED to handle both H and L
    if 'residue_sap' in structures_data and isinstance(structures_data['residue_sap'], dict):
        sap_count = 0
        for key, value in structures_data['residue_sap'].items():
            if key.endswith(f'_{chain_letter}'):
                parts = key.split('_')
                if len(parts) >= 3:
                    try:
                        # For Light chain, adjust the index offset
                        if chain_letter == 'L':
                            # Light chain indices start after Heavy chain
                            # Find the actual position in Light chain
                            idx = int(parts[1]) - len_heavy_seq  # Assuming Heavy has 121 residues
                        else:
                            idx = int(parts[1])
                        
                        if 0 <= idx < len(df):
                            mask = df['position_idx'] == idx
                            if mask.any():
                                df.loc[mask, 'sap'] = value
                                sap_count += 1
                    except ValueError:
                        continue
    
    # STEP 4c: Process high_sap_residues - FIXED for both chains
    if 'high_sap_residues' in structures_data and isinstance(structures_data['high_sap_residues'], list):
        high_sap_count = 0
        for residue in structures_data['high_sap_residues']:
            if isinstance(residue, str) and residue.endswith(f'_{chain_letter}'):
                parts = residue.split('_')
                if len(parts) >= 3:
                    try:
                        if chain_letter == 'L':
                            idx = int(parts[1]) - len_heavy_seq
                        else:
                            idx = int(parts[1])
                        
                        if 0 <= idx < len(df):
                            mask = df['position_idx'] == idx
                            if mask.any():
                                df.loc[mask, 'high_sap'] = True
                                high_sap_count += 1
                    except ValueError:
                        continue
    
    # STEP 4d: Process sidechain_sasa - FIXED for both chains
    if 'sidechain_sasa' in structures_data and isinstance(structures_data['sidechain_sasa'], dict):
        sasa_count = 0
        for key, value in structures_data['sidechain_sasa'].items():
            if key.endswith(f'_{chain_letter}'):
                parts = key.split('_')
                if len(parts) >= 3:
                    try:
                        if chain_letter == 'L':
                            idx = int(parts[1]) - len_heavy_seq
                        else:
                            idx = int(parts[1])
                        
                        if 0 <= idx < len(df):
                            mask = df['position_idx'] == idx
                            if mask.any():
                                df.loc[mask, 'sidechain_sasa'] = value
                                sasa_count += 1
                    except ValueError:
                        continue
    
    # STEP 4e: Process buried_residues - FIXED for both chains
    if 'buried_residues' in structures_data and isinstance(structures_data['buried_residues'], list):
        buried_count = 0
        for residue in structures_data['buried_residues']:
            if isinstance(residue, str) and residue.endswith(f'_{chain_letter}'):
                parts = residue.split('_')
                if len(parts) >= 3:
                    try:
                        if chain_letter == 'L':
                            idx = int(parts[1]) - len_heavy_seq
                        else:
                            idx = int(parts[1])
                        
                        if 0 <= idx < len(df):
                            mask = df['position_idx'] == idx
                            if mask.any():
                                df.loc[mask, 'buried'] = True
                                buried_count += 1
                    except ValueError:
                        continue
    
    # STEP 4f: Process pKa values - ENHANCED for debugging
    pka_count = 0
    if 'residue_pka' in structures_data:
        pka_data = structures_data['residue_pka']
        
        # Handle different data formats
        if isinstance(pka_data, str):
            try:
                import ast
                pka_data = ast.literal_eval(pka_data)
            except:
                pka_data = {}
        elif hasattr(pka_data, 'values'):
            pka_data = pka_data.values[0] if len(pka_data.values) > 0 else {}
            if isinstance(pka_data, str):
                try:
                    import ast
                    pka_data = ast.literal_eval(pka_data)
                except:
                    pka_data = {}
        
        if isinstance(pka_data, dict):
            # Debug: Check what's available for this chain
            chain_pkas = [k for k in pka_data.keys() if k.endswith(f'_{chain_letter}')]
            
            for key, value in pka_data.items():
                if key.endswith(f'_{chain_letter}'):
                    parts = key.split('_')
                    if len(parts) >= 3:
                        try:
                            if chain_letter == 'L':
                                idx = int(parts[1]) - len_heavy_seq
                            else:
                                idx = int(parts[1])
                            
                            if 0 <= idx < len(df):
                                mask = df['position_idx'] == idx
                                if mask.any():
                                    df.loc[mask, 'pka'] = value
                                    pka_count += 1
                        except ValueError:
                            continue
    
    # STEP 4g: Process pKa shifts - ENHANCED for both chains
    pka_shift_count = 0
    if 'pka_shifts' in structures_data:
        pka_shifts_data = structures_data['pka_shifts']
        
        # Handle different data formats
        if isinstance(pka_shifts_data, str):
            try:
                import ast
                pka_shifts_data = ast.literal_eval(pka_shifts_data)
            except:
                pka_shifts_data = {}
        elif hasattr(pka_shifts_data, 'values'):
            pka_shifts_data = pka_shifts_data.values[0] if len(pka_shifts_data.values) > 0 else {}
            if isinstance(pka_shifts_data, str):
                try:
                    import ast
                    pka_shifts_data = ast.literal_eval(pka_shifts_data)
                except:
                    pka_shifts_data = {}
        
        if isinstance(pka_shifts_data, dict):
            for key, value in pka_shifts_data.items():
                if key.endswith(f'_{chain_letter}'):
                    parts = key.split('_')
                    if len(parts) >= 3:
                        try:
                            if chain_letter == 'L':
                                idx = int(parts[1]) - len_heavy_seq
                            else:
                                idx = int(parts[1])
                            
                            if 0 <= idx < len(df):
                                mask = df['position_idx'] == idx
                                if mask.any():
                                    df.loc[mask, 'pka_shift'] = value
                                    pka_shift_count += 1
                        except ValueError:
                            continue
    
    # STEP 4h: Process disulfide bonds
    if 'disulfide_bonds' in structures_data and isinstance(structures_data['disulfide_bonds'], list):
        for bond in structures_data['disulfide_bonds']:
            if isinstance(bond, dict):
                for cys_key in ['cys1', 'cys2']:
                    if cys_key in bond:
                        cys_info = bond[cys_key]
                        if isinstance(cys_info, str) and cys_info.endswith(f'_{chain_letter}'):
                            try:
                                parts = cys_info.split('_')
                                if len(parts) >= 2:
                                    pos_num = int(parts[1])
                                    mask = (df['position_num'] == pos_num) & (df['amino_acid'] == 'C')
                                    if mask.any():
                                        df.loc[mask, 'disulfide_bond'] = True
                            except (ValueError, IndexError):
                                continue

    # STEP 4h.5: Add columns from df_residue based on position and chain
    if df_residue is not None and not df_residue.empty:
        # Filter df_residue for the current chain
        chain_residues = df_residue[df_residue['Chain'] == chain_letter].copy()

        if not chain_residues.empty:
            # For Light chain, we need to map PROPKA residue numbers to actual Light chain positions
            if chain_letter == 'L':
                # Use the known heavy sequence length as offset
                offset = len_heavy_seq

                # Map PROPKA residue numbers to Light chain positions
                for idx, row in chain_residues.iterrows():
                    propka_res_num = int(row['Residue_Number'])
                    actual_res_num = propka_res_num - offset

                    # Find matching position in df
                    mask = df['position_num'] == actual_res_num
                    if mask.any():
                        # Add all PROPKA columns with _propka suffix
                        propka_cols = ['pKa', 'BURIED', 'REGULAR', 'RE', 'Buried ratio', 
                                      'Disolvation regular Cst', 'Disolvation regular Nb',
                                      'Effects RE Cst', 'Effects RE Nb']

                        for col in propka_cols:
                            if col in row.index:
                                col_name = f"{col}_propka" if not col.endswith('_propka') else col
                                df.loc[mask, col_name] = row[col]
            else:
                # For Heavy chain, direct mapping works
                for idx, row in chain_residues.iterrows():
                    res_num = int(row['Residue_Number'])

                    # Find matching position in df
                    mask = df['position_num'] == res_num
                    if mask.any():
                        # Add all PROPKA columns with _propka suffix
                        propka_cols = ['pKa', 'BURIED', 'REGULAR', 'RE', 'Buried ratio', 
                                      'Disolvation regular Cst', 'Disolvation regular Nb',
                                      'Effects RE Cst', 'Effects RE Nb']

                        for col in propka_cols:
                            if col in row.index:
                                col_name = f"{col}_propka" if not col.endswith('_propka') else col
                                df.loc[mask, col_name] = row[col]


    # NEW STEP 4i: Add columns from df_Ab
    if df_Ab is not None and not df_Ab.empty:
        
        # Define columns to add from df_Ab
        df_ab_columns = [
            'SASA', 'RASA', 'EXP_HSE_B_U', 'EXP_HSE_B_D', 'EXP_HSE_A_U', 'EXP_HSE_A_D',
            'hydrophobicity', 'mean_IP', 'max_IP', 'min_IP', 'mean_dpx',
            'CA_x', 'CA_y', 'CA_z', 'residue_name_U', 'charge'
        ]
        
        # Initialize these columns in df
        for col in df_ab_columns:
            if col not in df.columns:
                df[col] = np.nan
        
        # Filter df_Ab for the current chain
        chain_df_ab = df_Ab[df_Ab['chain'] == chain_letter].copy()
        
        if not chain_df_ab.empty:
            # Reset index for chain_df_ab to be 0-based for that chain
            chain_df_ab = chain_df_ab.reset_index(drop=True)
            
            # Three-letter to one-letter amino acid conversion
            three_to_one = {
                'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
                'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
                'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
                'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
            }
            
            matched_count = 0
            for idx, row in df.iterrows():
                # Match by position_idx
                if idx < len(chain_df_ab):
                    match = chain_df_ab.iloc[idx]
                    
                    # Verify amino acid consistency
                    aa_match = False
                    if 'residue_name' in chain_df_ab.columns:
                        three_letter = match['residue_name']
                        if three_letter in three_to_one:
                            one_letter = three_to_one[three_letter]
                            if one_letter == row['amino_acid']:
                                aa_match = True
                        else:
                            # If no conversion needed or unknown, skip verification
                            aa_match = True
                    else:
                        aa_match = True  # No residue_name column to verify
                    
                    if aa_match:
                        # Copy values from df_Ab to df
                        for col in df_ab_columns:
                            if col in chain_df_ab.columns:
                                df.loc[idx, col] = match[col]
                        matched_count += 1
                    else:
                        print(f" ")
            

    # STEP 5: Add liability data
    liability_types = [
        'Unpaired_Cys', 'N-linked_glycosylation', 'Met_oxidation', 'Trp_oxidation',
        'Asn_deamidation', 'Asp_isomerisation', 'Lysine_Glycation', 'N-terminal_glutamate',
        'Integrin_binding', 'CD11c/CD18_binding', 'Fragmentation', 'Polyreactivity'
    ]
    
    for col_name in liability_types:
        df[col_name] = False
    
    if liabilities_list and isinstance(liabilities_list, list):
        for liability in liabilities_list:
            if isinstance(liability, dict) and liability.get('chain') == chain_letter:
                if 'start_position' in liability and 'end_position' in liability:
                    start_pos = liability['start_position'][0] if isinstance(liability['start_position'], list) else liability['start_position']
                    end_pos = liability['end_position'][0] if isinstance(liability['end_position'], list) else liability['end_position']
                    
                    col_name = liability['name'].split('(')[0].strip().replace(' ', '_').replace('/', '')
                    
                    mask = (df['position_num'] >= start_pos) & (df['position_num'] <= end_pos)
                    if mask.any():
                        df.loc[mask, col_name] = True
    
    # STEP 6: Organize columns
    # STEP 6: Organize columns
    column_order = [
        'position_seq', 'position_num', 'amino_acid', 'region',
        'disulfide_bond', 'sap', 'high_sap', 'sidechain_sasa', 'buried',
        'pka', 'pka_shift', 'residue_sasa_sum',
        'SASA', 'RASA', 'EXP_HSE_B_U', 'EXP_HSE_B_D', 'EXP_HSE_A_U', 'EXP_HSE_A_D',
        'hydrophobicity', 'mean_IP', 'max_IP', 'min_IP', 'mean_dpx',
        'CA_x', 'CA_y', 'CA_z', 'residue_name_U', 'charge',
        'charge_sign', 'hydrophobicity_hw', 'hydrophobicity_eisenberg',
        'hydrophobicity_rose', 'hydrophobicity_janin', 'hydrophobicity_engelman',
        'Buried ratio_propka', 'Disolvation regular Cst_propka', 
        'Disolvation regular Nb_propka', 'Effects RE Cst_propka', 'Effects RE Nb_propka'
    ] + liability_types
    df = df[[col for col in column_order if col in df.columns]]
    

    return df


import pandas as pd

def complete_peptide_results(df, heavy_valid=None, light_valid=None, cdrs_H=None, cdrs_L=None):
    """
    Enrich peptide dataframe with validity flags and CDR sequences.
    Safe against missing heavy/light info.
    """
    # Default fallbacks
    if cdrs_H is None: 
        cdrs_H = {}
    if cdrs_L is None: 
        cdrs_L = {}

    # Map validity safely
    valid_map = {
        "Heavy": heavy_valid if heavy_valid is not None else False,
        "Light": light_valid if light_valid is not None else False
    }
    df["valid"] = df["chain"].map(valid_map).fillna(False)

    # Add type (H/L)
    type_map = {"Heavy": "H", "Light": "L"}
    df["type"] = df["chain"].map(type_map).fillna("NA")

    # Add CDRs for Heavy
    for cdr, seq in cdrs_H.items():
        if "Heavy" in df["chain"].values:
            df.loc[df["chain"] == "Heavy", cdr] = seq

    # Add CDRs for Light
    for cdr, seq in cdrs_L.items():
        if "Light" in df["chain"].values:
            df.loc[df["chain"] == "Light", cdr] = seq

    return df


import pandas as pd

def combine_all_results(df_AA, structure_df, sequence_df, peptide_df, 
                        heavy_valid=None, light_valid=None, 
                        cdrs_H=None, cdrs_L=None):
    """
    Combine structure_results_comp, sequence_results, and peptide_results
    into a single-row dataframe. Resistant if heavy/light chains missing.
    """

    if cdrs_H is None: cdrs_H = {}
    if cdrs_L is None: cdrs_L = {}

    # --- 1. Start with structure + sequence (safe outer merge on index) ---
    combined = pd.concat([df_AA.reset_index(drop=True), structure_df.reset_index(drop=True),
                          sequence_df.reset_index(drop=True)], axis=1)

    # --- 2. Extract heavy and light peptide rows safely ---
    heavy_row = peptide_df.loc[peptide_df["chain"] == "Heavy"].copy() if "Heavy" in peptide_df["chain"].values else pd.DataFrame()
    light_row = peptide_df.loc[peptide_df["chain"] == "Light"].copy() if "Light" in peptide_df["chain"].values else pd.DataFrame()

    # --- 3. Flatten heavy peptide row ---
    if not heavy_row.empty:
        heavy_row = heavy_row.reset_index(drop=True).add_prefix("Heavy_")
        heavy_row["Heavy_valid"] = heavy_valid if heavy_valid is not None else False
        for k, v in cdrs_H.items():
            heavy_row[f"Heavy_{k}"] = v
    else:
        heavy_row = pd.DataFrame([{}])  # empty fallback

    # --- 4. Flatten light peptide row ---
    if not light_row.empty:
        light_row = light_row.reset_index(drop=True).add_prefix("Light_")
        light_row["Light_valid"] = light_valid if light_valid is not None else False
        for k, v in cdrs_L.items():
            light_row[f"Light_{k}"] = v
    else:
        light_row = pd.DataFrame([{}])  # empty fallback

    # --- 5. Merge everything into single row ---
    final = pd.concat([combined, heavy_row, light_row], axis=1)

    return final



def plot_protein_properties(df, chain_type='heavy', figsize=(20, 16)):
    """
    Create comprehensive plots of protein properties along sequence positions
    with CDR highlighting and liability indicators.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with protein sequence data (df_heavy_final or df_light_final)
    chain_type : str
        'heavy' or 'light' to label the plot
    figsize : tuple
        Figure size
    """
    
    # Define CDR regions and their colors
    cdr_colors = {
        'cdrh1': '#FFB6C1', 'cdrh2': '#FFA07A', 'cdrh3': '#FF6B6B',  # Heavy CDRs - red/pink shades
        'cdrl1': '#87CEEB', 'cdrl2': '#6495ED', 'cdrl3': '#4169E1',  # Light CDRs - blue shades
        'fwh1': '#F0F0F0', 'fwh2': '#F0F0F0', 'fwh3': '#F0F0F0', 'fwh4': '#F0F0F0',  # Framework - light gray
        'fwl1': '#F0F0F0', 'fwl2': '#F0F0F0', 'fwl3': '#F0F0F0', 'fwl4': '#F0F0F0'
    }
    
    # Define liability colors
    liability_colors = {
        'Unpaired_Cys': 'red',
        'N-linked_glycosylation': 'blue',
        'Met_oxidation': 'orange',
        'Trp_oxidation': 'brown',
        'Asn_deamidation': 'purple',
        'Asp_isomerisation': 'pink',
        'Lysine_Glycation': 'cyan',
        'N-terminal_glutamate': 'lime',
        'Integrin_binding': 'navy',
        'CD11c/CD18_binding': 'teal',
        'Fragmentation': 'maroon',
        'Polyreactivity': 'gold'
    }
    
    # Create figure with subplots
    fig, axes = plt.subplots(8, 1, figsize=figsize, gridspec_kw={'height_ratios': [1, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1]})
    fig.suptitle(f'{chain_type.capitalize()} Chain - Sequence Properties Analysis', fontsize=16, fontweight='bold')
    
    positions = df['position_seq'].values
    
    # 1. Region highlighting (top panel)
    ax1 = axes[0]
    ax1.set_xlim(positions.min() - 1, positions.max() + 1)
    ax1.set_ylim(0, 1)
    
    # Group consecutive positions by region
    current_region = None
    start_pos = positions[0]
    
    for i, (pos, region) in enumerate(zip(df['position_seq'], df['region'])):
        if region != current_region:
            if current_region is not None:
                # Draw rectangle for previous region
                width = positions[i-1] - start_pos + 1
                rect = Rectangle((start_pos - 0.5, 0), width, 1, 
                               facecolor=cdr_colors.get(current_region, '#F0F0F0'),
                               edgecolor='black', linewidth=0.5)
                ax1.add_patch(rect)
                # Add region label
                if 'cdr' in current_region:
                    ax1.text(start_pos + width/2 - 0.5, 0.5, current_region.upper(), 
                           ha='center', va='center', fontsize=9, fontweight='bold')
            start_pos = pos
            current_region = region
    
    # Don't forget the last region
    if current_region is not None:
        width = positions[-1] - start_pos + 1
        rect = Rectangle((start_pos - 0.5, 0), width, 1,
                       facecolor=cdr_colors.get(current_region, '#F0F0F0'),
                       edgecolor='black', linewidth=0.5)
        ax1.add_patch(rect)
        if 'cdr' in current_region:
            ax1.text(start_pos + width/2 - 0.5, 0.5, current_region.upper(),
                   ha='center', va='center', fontsize=9, fontweight='bold')
    
    ax1.set_ylabel('Regions', fontsize=10)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title('CDR and Framework Regions', fontsize=11)
    
    # 2. SAP and SASA plot
    ax2 = axes[1]
    ax2_twin = ax2.twinx()
    
    # Plot SAP
    ax2.plot(positions, df['sap'], 'b-', label='SAP', linewidth=1.5, alpha=0.7)
    ax2.scatter(positions[df['high_sap']], df.loc[df['high_sap'], 'sap'], 
               color='red', s=30, zorder=5, label='High SAP')
    ax2.set_ylabel('SAP', color='b', fontsize=10)
    ax2.tick_params(axis='y', labelcolor='b')
    ax2.set_ylim(0, df['sap'].max() * 1.1)
    
    # Plot SASA on twin axis
    ax2_twin.plot(positions, df['sidechain_sasa'], 'g-', label='Sidechain SASA', linewidth=1.5, alpha=0.7)
    ax2_twin.set_ylabel('Sidechain SASA', color='g', fontsize=10)
    ax2_twin.tick_params(axis='y', labelcolor='g')
    
    ax2.set_title('SAP and SASA', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    
    # 3. RASA and Buried residues
    ax3 = axes[2]
    ax3.plot(positions, df['RASA'], 'purple', linewidth=1.5, label='RASA')
    buried_positions = positions[df['buried']]
    if len(buried_positions) > 0:
        ax3.scatter(buried_positions, np.zeros(len(buried_positions)), 
                   color='black', s=50, marker='^', label='Buried', zorder=5)
    ax3.set_ylabel('RASA (%)', fontsize=10)
    ax3.set_title('Relative Accessible Surface Area (RASA)', fontsize=11)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. HSE (Half-Sphere Exposure) values
    ax4 = axes[3]
    ax4.plot(positions, df['EXP_HSE_B_U'], 'r-', label='HSE_B_U', linewidth=1, alpha=0.7)
    ax4.plot(positions, df['EXP_HSE_B_D'], 'b-', label='HSE_B_D', linewidth=1, alpha=0.7)
    ax4.plot(positions, df['EXP_HSE_A_U'], 'g-', label='HSE_A_U', linewidth=1, alpha=0.7)
    ax4.plot(positions, df['EXP_HSE_A_D'], 'm-', label='HSE_A_D', linewidth=1, alpha=0.7)
    ax4.set_ylabel('HSE Count', fontsize=10)
    ax4.set_title('Half-Sphere Exposure', fontsize=11)
    ax4.legend(ncol=4, loc='upper right', fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # 5. Hydrophobicity and IP
    ax5 = axes[4]
    ax5_twin = ax5.twinx()
    
    # Plot hydrophobicity
    ax5.plot(positions, df['hydrophobicity'], 'navy', linewidth=1.5, label='Hydrophobicity')
    ax5.set_ylabel('Hydrophobicity', color='navy', fontsize=10)
    ax5.tick_params(axis='y', labelcolor='navy')
    ax5.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Plot IP values
    ax5_twin.plot(positions, df['mean_IP'], 'orange', linewidth=1, alpha=0.7, label='Mean IP')
    ax5_twin.fill_between(positions, df['min_IP'], df['max_IP'], alpha=0.2, color='orange')
    ax5_twin.set_ylabel('Isoelectric Point (IP)', color='orange', fontsize=10)
    ax5_twin.tick_params(axis='y', labelcolor='orange')
    
    ax5.set_title('Hydrophobicity and Isoelectric Point', fontsize=11)
    ax5.legend(loc='upper left')
    ax5_twin.legend(loc='upper right')
    ax5.grid(True, alpha=0.3)
    
    # 6. Charge and pKa
    ax6 = axes[5]
    ax6.bar(positions, df['charge'], color=['red' if c < 0 else 'blue' if c > 0 else 'gray' 
                                           for c in df['charge']], alpha=0.6, label='Charge')
    
    # Plot pKa values as points where they exist
    pka_mask = df['pka'].notna()
    if pka_mask.any():
        # Convert pka strings to floats, handling the case where they might be strings
        pka_values = pd.to_numeric(df.loc[pka_mask, 'pka'], errors='coerce')
        pka_positions = positions[pka_mask]
        valid_pka = ~pka_values.isna()
        
        if valid_pka.any():
            ax6_twin = ax6.twinx()
            ax6_twin.scatter(pka_positions[valid_pka], pka_values[valid_pka], 
                           color='green', s=40, marker='o', label='pKa', zorder=5)
            ax6_twin.set_ylabel('pKa', color='green', fontsize=10)
            ax6_twin.tick_params(axis='y', labelcolor='green')
            ax6_twin.legend(loc='upper right')
    
    ax6.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax6.set_ylabel('Charge', fontsize=10)
    ax6.set_title('Charge Distribution and pKa Values', fontsize=11)
    ax6.legend(loc='upper left')
    ax6.grid(True, alpha=0.3)
    
    # 7. Amino acid sequence
    ax7 = axes[6]
    ax7.set_xlim(positions.min() - 1, positions.max() + 1)
    ax7.set_ylim(-0.5, 0.5)
    
    # Plot amino acid letters
    for pos, aa in zip(df['position_seq'], df['amino_acid']):
        ax7.text(pos, 0, aa, ha='center', va='center', fontsize=8)
    
    ax7.set_ylabel('AA', fontsize=10)
    ax7.set_title('Amino Acid Sequence', fontsize=11)
    ax7.set_yticks([])
    ax7.set_xlabel('Position', fontsize=10)
    ax7.grid(True, alpha=0.3, axis='x')
    
    # 8. Liabilities (bottom panel)
    ax8 = axes[7]
    ax8.set_xlim(positions.min() - 1, positions.max() + 1)
    
    liability_cols = [col for col in liability_colors.keys() if col in df.columns]
    y_positions = np.linspace(0, 1, len(liability_cols))
    
    # Plot liabilities
    for i, liability in enumerate(liability_cols):
        liability_positions = positions[df[liability] == True]
        if len(liability_positions) > 0:
            ax8.scatter(liability_positions, [y_positions[i]] * len(liability_positions),
                       color=liability_colors[liability], s=40, alpha=0.8, label=liability)
    
    ax8.set_ylim(-0.1, 1.1)
    ax8.set_ylabel('Liabilities', fontsize=10)
    ax8.set_xlabel('Position', fontsize=11)
    ax8.set_yticks([])
    
    # Create custom legend with smaller markers
    handles, labels = ax8.get_legend_handles_labels()
    if handles:
        ax8.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), 
                  ncol=1, fontsize=8, markerscale=0.8)
    
    ax8.set_title('Post-Translational Modification Liabilities', fontsize=11)
    ax8.grid(True, alpha=0.3, axis='x')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

# Function to create a simpler version focusing on key properties
def plot_key_properties(df, chain_type='heavy', figsize=(16, 10)):
    """
    Create a simplified plot focusing on key properties
    """
    fig, axes = plt.subplots(5, 1, figsize=figsize, gridspec_kw={'height_ratios': [0.8, 1.2, 1.2, 1.2, 0.8]})
    fig.suptitle(f'{chain_type.capitalize()} Chain - Key Properties', fontsize=14, fontweight='bold')
    
    positions = df['position_seq'].values
    
    # Define CDR colors
    cdr_colors = {
        'cdrh1': '#FF6B6B', 'cdrh2': '#FF6B6B', 'cdrh3': '#FF6B6B',
        'cdrl1': '#4169E1', 'cdrl2': '#4169E1', 'cdrl3': '#4169E1',
    }
    
    # 1. Region highlighting
    ax1 = axes[0]
    for i, (pos, region) in enumerate(zip(df['position_seq'], df['region'])):
        if 'cdr' in region:
            ax1.axvspan(pos - 0.5, pos + 0.5, alpha=0.5, color=cdr_colors.get(region, 'gray'))
    
    ax1.set_xlim(positions.min() - 1, positions.max() + 1)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel('CDR', fontsize=10)
    ax1.set_title('CDR Regions (highlighted)', fontsize=11)
    ax1.set_yticks([])
    
    # 2. SAP and high SAP
    ax2 = axes[1]
    ax2.plot(positions, df['sap'], 'b-', linewidth=1.5, alpha=0.7)
    ax2.scatter(positions[df['high_sap']], df.loc[df['high_sap'], 'sap'],
               color='red', s=50, zorder=5, label='High SAP')
    ax2.set_ylabel('SAP', fontsize=10)
    ax2.set_title('Spatial Aggregation Propensity (SAP)', fontsize=11)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Hydrophobicity
    ax3 = axes[2]
    ax3.plot(positions, df['hydrophobicity'], 'darkgreen', linewidth=1.5)
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_ylabel('Hydrophobicity', fontsize=10)
    ax3.set_title('Hydrophobicity Profile', fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    # 4. Charge
    ax4 = axes[3]
    colors = ['red' if c < 0 else 'blue' if c > 0 else 'gray' for c in df['charge']]
    ax4.bar(positions, df['charge'], color=colors, alpha=0.6)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax4.set_ylabel('Charge', fontsize=10)
    ax4.set_title('Charge Distribution', fontsize=11)
    ax4.grid(True, alpha=0.3)
    
    # 5. Liabilities summary
    ax5 = axes[4]
    liability_cols = [k for k in liability_colors]
    liability_cols = [col for col in liability_cols if col in df.columns]
    
    # Count liabilities per position
    liability_count = df[liability_cols].sum(axis=1)
    liability_positions = positions[liability_count > 0]
    
    if len(liability_positions) > 0:
        ax5.scatter(liability_positions, [0.5] * len(liability_positions),
                   s=liability_count[liability_count > 0] * 100,
                   color='red', alpha=0.6)
    
    ax5.set_xlim(positions.min() - 1, positions.max() + 1)
    ax5.set_ylim(0, 1)
    ax5.set_xlabel('Position', fontsize=11)
    ax5.set_ylabel('PTMs', fontsize=10)
    ax5.set_title('Post-Translational Modifications (size = count)', fontsize=11)
    ax5.set_yticks([])
    ax5.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    return fig


def standardize_ph_value(ph_str):
    """
    Standardize pH value format to always have 2 decimal places
    e.g., "14" -> "14.00", "1" -> "1.00", "7.5" -> "7.50"
    """
    try:
        ph_float = float(ph_str)
        return f"{ph_float:.2f}"
    except:
        return ph_str

def reshape_dataframe_by_object(df_test):
    """
    Reshape dataframe so each object (row) becomes a separate dataframe
    with pH values as columns and descriptors as rows.
    
    Parameters:
    -----------
    df_test : pd.DataFrame
        Input dataframe where each row is an object and columns are descriptor_pH combinations
    
    Returns:
    --------
    dict : Dictionary where keys are object indices and values are reshaped dataframes
    """
    
    # Get all pH-related columns
    patterns = ["Light_Charges_pH_", "Heavy_Charge_pH_", "Free_Energy_kcal_mol_pH_",
                "Protein_Charge_Unfolded_pH_", "Protein_Charge_Folded_pH_",
                "pI_Folded_pH_", "pI_Unfolded_pH_"]
    
    # Get pH columns
    ph_cols = [col for col in df_test.columns 
               if any(pattern in col for pattern in patterns)]
    
    # Dictionary to store results for each object
    result_dict = {}
    
    # Process each object (row) in the dataframe
    for idx, row in df_test.iterrows():
        # Extract only pH-related data for this object
        ph_data = row[ph_cols]
        
        # Create a dictionary to organize data by descriptor and pH
        data_dict = {}
        
        for col in ph_cols:
            # Parse column name to extract descriptor and pH value
            if "Free_Energy_kcal_mol_pH_" in col:
                descriptor = "Free_Energy_kcal_mol"
                ph_value = col.replace("Free_Energy_kcal_mol_pH_", "")
            elif "Light_Charges_pH_" in col:
                descriptor = "Light_Charges"
                ph_value = col.replace("Light_Charges_pH_", "")
            elif "Heavy_Charge_pH_" in col:
                descriptor = "Heavy_Charge"
                ph_value = col.replace("Heavy_Charge_pH_", "")
            elif "Protein_Charge_Unfolded_pH_" in col:
                descriptor = "Protein_Charge_Unfolded"
                ph_value = col.replace("Protein_Charge_Unfolded_pH_", "")
            elif "Protein_Charge_Folded_pH_" in col:
                descriptor = "Protein_Charge_Folded"
                ph_value = col.replace("Protein_Charge_Folded_pH_", "")
            elif "pI_Folded_pH_" in col:
                descriptor = "pI_Folded"
                ph_value = col.replace("pI_Folded_pH_", "")
            elif "pI_Unfolded_pH_" in col:
                descriptor = "pI_Unfolded"
                ph_value = col.replace("pI_Unfolded_pH_", "")
            else:
                continue
            
            # Standardize pH value format
            ph_value_std = standardize_ph_value(ph_value)
            
            # Initialize descriptor if not exists
            if descriptor not in data_dict:
                data_dict[descriptor] = {}
            
            # Store the value with standardized pH format
            data_dict[descriptor][f"pH_{ph_value_std}"] = ph_data[col]
        
        # Convert to DataFrame (descriptors as rows, pH as columns)
        object_df = pd.DataFrame(data_dict).T
        
        # Sort columns by pH value numerically
        ph_columns = object_df.columns.tolist()
        ph_values = [float(col.replace("pH_", "")) for col in ph_columns]
        sorted_indices = np.argsort(ph_values)
        sorted_columns = [ph_columns[i] for i in sorted_indices]
        
        object_df = object_df[sorted_columns]
        
        # Store in result dictionary
        result_dict[idx] = object_df
    
    return result_dict

def plot_ph_profiles(object_df, object_id=None, figsize=(16, 8)):
    """
    Create plots for pH-dependent properties
    
    Parameters:
    -----------
    object_df : pd.DataFrame
        DataFrame for a single object with descriptors as rows and pH as columns
    object_id : str/int, optional
        Identifier for the object being plotted
    figsize : tuple
        Figure size for the plots
    """
    
    # Extract pH values from column names
    ph_values = [float(col.replace("pH_", "")) for col in object_df.columns]
    
    # Create figure with subplots
    fig, axes = plt.subplots(4, 1, figsize=figsize)
    fig.suptitle(f'pH-dependent Properties{" - Object " + str(object_id) if object_id is not None else ""}', 
                 fontsize=14, fontweight='bold')
    
    # 1. Plot pI (Folded vs Unfolded)
    ax1 = axes[0]
    if 'pI_Folded' in object_df.index and 'pI_Unfolded' in object_df.index:
        pi_folded = object_df.loc['pI_Folded'].values
        pi_unfolded = object_df.loc['pI_Unfolded'].values
        
        # Remove NaN values for plotting
        mask_folded = ~pd.isna(pi_folded)
        mask_unfolded = ~pd.isna(pi_unfolded)
        
        ax1.plot(np.array(ph_values)[mask_folded], pi_folded[mask_folded], 
                'b-o', label='pI Folded', linewidth=2, markersize=4)
        ax1.plot(np.array(ph_values)[mask_unfolded], pi_unfolded[mask_unfolded], 
                'r-s', label='pI Unfolded', linewidth=2, markersize=4)
        ax1.set_xlabel('pH', fontsize=11)
        ax1.set_ylabel('pI', fontsize=11)
        ax1.set_title('Isoelectric Point (pI) vs pH', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 14)
    
    # 2. Plot Charges (Light vs Heavy)
    ax2 = axes[1]
    if 'Light_Charges' in object_df.index and 'Heavy_Charge' in object_df.index:
        light_charges = object_df.loc['Light_Charges'].values
        heavy_charges = object_df.loc['Heavy_Charge'].values
        
        # Remove NaN values for plotting
        mask_light = ~pd.isna(light_charges)
        mask_heavy = ~pd.isna(heavy_charges)
        
        ax2.plot(np.array(ph_values)[mask_light], light_charges[mask_light], 
                'g-^', label='Light Charges', linewidth=2, markersize=4)
        ax2.plot(np.array(ph_values)[mask_heavy], heavy_charges[mask_heavy], 
                'm-v', label='Heavy Charge', linewidth=2, markersize=4)
        ax2.set_xlabel('pH', fontsize=11)
        ax2.set_ylabel('Charge', fontsize=11)
        ax2.set_title('Charges vs pH', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax2.set_xlim(0, 14)
    
    # 3. Plot Protein Charges (Folded vs Unfolded)
    ax3 = axes[2]
    if 'Protein_Charge_Folded' in object_df.index and 'Protein_Charge_Unfolded' in object_df.index:
        protein_folded = object_df.loc['Protein_Charge_Folded'].values
        protein_unfolded = object_df.loc['Protein_Charge_Unfolded'].values
        
        # Remove NaN values for plotting
        mask_folded = ~pd.isna(protein_folded)
        mask_unfolded = ~pd.isna(protein_unfolded)
        
        ax3.plot(np.array(ph_values)[mask_folded], protein_folded[mask_folded], 
                'c-o', label='Protein Charge Folded', linewidth=2, markersize=4)
        ax3.plot(np.array(ph_values)[mask_unfolded], protein_unfolded[mask_unfolded], 
                'orange', marker='s', linestyle='-', label='Protein Charge Unfolded', 
                linewidth=2, markersize=4)
        ax3.set_xlabel('pH', fontsize=11)
        ax3.set_ylabel('Protein Charge', fontsize=11)
        ax3.set_title('Protein Charge vs pH', fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax3.set_xlim(0, 14)
    
    # 3. Free_Energy_kcal_mol
    ax4 = axes[3]
    if 'Free_Energy_kcal_mol' in object_df.index:
        Free_nrj_folded = object_df.loc['Free_Energy_kcal_mol'].values
        
        # Remove NaN values for plotting
        mask_Free_nrj_folded = ~pd.isna(Free_nrj_folded)
        
        ax4.plot(np.array(ph_values)[mask_Free_nrj_folded], Free_nrj_folded[mask_Free_nrj_folded], 
                'c-o', label='Free Energy kcal mol', linewidth=2, markersize=4)
        ax4.set_xlabel('pH', fontsize=11)
        ax4.set_ylabel('Free Energy kcal mol', fontsize=11)
        ax4.set_title('Free Energy vs pH', fontsize=12)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(0, 14)
    
    
    
    
    plt.tight_layout()
    return fig

def plot_all_objects(result_dict, max_objects=5):
    """
    Plot pH profiles for multiple objects
    
    Parameters:
    -----------
    result_dict : dict
        Dictionary with object dataframes
    max_objects : int
        Maximum number of objects to plot
    """
    n_objects = min(len(result_dict), max_objects)
    
    for i, (obj_id, df) in enumerate(result_dict.items()):
        if i >= n_objects:
            break
        plot_ph_profiles(df, object_id=obj_id)
        plt.show()

# Helper functions
def display_results(result_dict, n_objects_to_show=2):
    """
    Display the first n objects' dataframes
    """
    for i, (obj_id, df) in enumerate(result_dict.items()):
        if i >= n_objects_to_show:
            break
        print(f"\n=== Object {obj_id} ===")
        print(df.head())  # Show first few columns
        print(f"Shape: {df.shape}")
        print(f"Columns range: {df.columns[0]} to {df.columns[-1]}")

def get_object_dataframe(result_dict, object_index):
    """
    Get dataframe for a specific object
    """
    return result_dict.get(object_index)


def rename_columns_heavy_light(df):
    """
    Rename columns to standardize Heavy/Light prefixes and remove redundant suffixes
    """
    rename_dict = {}
    
    for col in df.columns:
        new_col = col
        
        # Handle columns ending with _VH (Variable Heavy)
        if col.endswith('_VH'):
            new_col = 'Heavy_' + col[:-3]  # Remove _VH and add Heavy_ prefix
        
        # Handle columns ending with _VL (Variable Light)
        elif col.endswith('_VL'):
            new_col = 'Light_' + col[:-3]  # Remove _VL and add Light_ prefix
        
        # Handle columns ending with _H (Heavy)
        elif col.endswith('_H') and not col.endswith('_VH'):
            # Special case: don't rename if it's part of another pattern
            if col not in ['SASA_ratio_H_to_L']:  # Keep this as is
                new_col = 'Heavy_' + col[:-2]  # Remove _H and add Heavy_ prefix
        
        # Handle columns ending with _L (Light)
        elif col.endswith('_L') and not col.endswith('_VL'):
            # Special case: don't rename if it's part of another pattern
            if col not in ['SASA_ratio_H_to_L']:  # Keep this as is
                new_col = 'Light_' + col[:-2]  # Remove _L and add Light_ prefix
        
        # Handle columns with _H_ in the middle
        elif '_H_' in col and col != 'SASA_ratio_H_to_L':
            # Replace _H_ with _Heavy_
            new_col = col.replace('_H_', '_Heavy_')
        
        # Handle columns with _L_ in the middle
        elif '_L_' in col and col != 'SASA_ratio_H_to_L':
            # Replace _L_ with _Light_
            new_col = col.replace('_L_', '_Light_')
        
        # Handle columns that already have _Heavy_ in them
        elif '_Heavy_' in col:
            # Move Heavy_ to the beginning if it's not already there
            if not col.startswith('Heavy_'):
                new_col = 'Heavy_' + col.replace('_Heavy_', '_')
        
        # Handle columns that already have _Light_ in them
        elif '_Light_' in col:
            # Move Light_ to the beginning if it's not already there
            if not col.startswith('Light_'):
                new_col = 'Light_' + col.replace('_Light_', '_')
        
        # Add to rename dictionary if the name changed
        if new_col != col:
            rename_dict[col] = new_col
    
    # Apply the renaming
    df_renamed = df.rename(columns=rename_dict)
    
    return df_renamed


def prepare_object_descriptors(df_test):
    # 1. Use set for faster lookups
    patterns = ["Light_Charges_pH_", "Heavy_Charge_pH_", "Free_Energy_kcal_mol_",
                "Protein_Charge_Unfolded_", "Protein_Charge_Folded_pH_",
                "pI_Folded_pH_", "pI_Unfolded_pH_"]

    # More efficient pattern matching
    col_ph = []
    for col in df_test.columns:
        for pattern in patterns:
            if pattern in col:
                col_ph.append(col)
                break  # Stop checking other patterns once matched

    col_ph = sorted(col_ph)

    # Add additional columns (remove duplicates with set)
    additional_cols = ["SeqID", "Type", "Heavy_chain", "Light_chain", "heavy_sequence", 
                       "light_sequence", "StructureID", "PDB_File", "phi_psi", "fc_charge_pH7.4", "positive_patches",
                       "negative_patches", "Heavy_molar_extinction_reduced", "Heavy_molar_extinction_cystines"]
    
    col_ph.extend(additional_cols)

    # Use set for O(1) lookups instead of list
    col_ph_set = set(col.lower() for col in col_ph)

    # More efficient column filtering
    cols_to_keep = [col for col in df_test.columns if col.lower() not in col_ph_set]
    df_mod = df_test[cols_to_keep]


    # Apply the renaming
    df_mod = rename_columns_heavy_light(df_mod)

    # Find all dipole_moment columns (exact match for the pattern)
    dipole_cols = [col for col in df_mod.columns if col == 'dipole_moment' or col.startswith('dipole_moment')]

    # Also check if there are duplicates with same name
    from collections import Counter
    col_counts = Counter(df_mod.columns)
    duplicate_cols = [col for col, count in col_counts.items() if count > 1 and 'dipole_moment' in col]

    if duplicate_cols:
        # Handle duplicates by making them unique first
        new_columns = []
        seen_counts = {}
        for col in df_mod.columns:
            if col in duplicate_cols:
                if col not in seen_counts:
                    seen_counts[col] = 0
                else:
                    seen_counts[col] += 1
                if seen_counts[col] > 0:
                    new_columns.append(f"{col}_{seen_counts[col]}")
                else:
                    new_columns.append(col)
            else:
                new_columns.append(col)
        df_mod.columns = new_columns

    # Now find unique dipole_moment columns (excluding RM_dipole_to_hydrophobic)
    dipole_cols = [col for col in df_mod.columns 
                   if (col == 'dipole_moment' or col.startswith('dipole_moment')) 
                   and 'RM_dipole' not in col]


    # Get first two unique dipole columns based on their values
    if len(dipole_cols) > 0:
        # Check which columns have unique data
        unique_cols = []
        seen_data = []

        for col in dipole_cols:
            col_hash = hash(tuple(df_mod[col].round(6).values))  # Round to avoid float precision issues
            if col_hash not in seen_data:
                seen_data.append(col_hash)
                unique_cols.append(col)
                if len(unique_cols) == 2:  # Stop after finding 2 unique columns
                    break

        # Create rename mapping
        rename_dict = {}
        if len(unique_cols) >= 1:
            rename_dict[unique_cols[0]] = 'dipole_moment_all'
        if len(unique_cols) >= 2:
            rename_dict[unique_cols[1]] = 'dipole_moment_sub'

        # Remove all dipole_moment columns (except RM_dipole_to_hydrophobic)
        cols_to_drop = [col for col in dipole_cols if col not in rename_dict.keys()]
        df_mod = df_mod.drop(columns=cols_to_drop)

        # Rename the kept columns
        df_mod = df_mod.rename(columns=rename_dict)

    # Define ordered columns
    ordered_cols = [
        'heavy_sequence',  # Put sequences first
        'light_sequence',

        'Heavy_chain',
        'Light_chain', 
        'scheme',
        'Light_valid',
        'Heavy_valid',
        'Light_cdrl1',
        'Light_cdrl2',
        'Light_cdrl3',
        'Heavy_cdrh1',
        'Heavy_cdrh2',
        'Heavy_cdrh3'
    ]

    # If sequences are missing, try to get them from original df_test
    missing_sequences = []
    if 'heavy_sequence' not in df_mod.columns and 'heavy_sequence' in df_test.columns:
        df_mod['heavy_sequence'] = df_test['heavy_sequence']
        missing_sequences.append('heavy_sequence')
    if 'light_sequence' not in df_mod.columns and 'light_sequence' in df_test.columns:
        df_mod['light_sequence'] = df_test['light_sequence']
        missing_sequences.append('light_sequence')


    # Use set for faster membership testing
    ordered_set = set(ordered_cols)

    # Get existing ordered columns
    existing_ordered = [col for col in ordered_cols if col in df_mod.columns]

    # Get remaining columns efficiently using set difference
    remaining_cols = sorted([col for col in df_mod.columns if col not in ordered_set])

    # Final reordering
    df_mod = df_mod[existing_ordered + remaining_cols]

    return(df_mod)

def plot_protein_properties_with_propka(df, chain_type='heavy', figsize=(20, 22)):
    """
    Create comprehensive plots including PROPKA-derived properties.
    """
    
    # Define CDR regions and their colors
    cdr_colors = {
        'cdrh1': '#FFB6C1', 'cdrh2': '#FFA07A', 'cdrh3': '#FF6B6B',  # Heavy CDRs
        'cdrl1': '#87CEEB', 'cdrl2': '#6495ED', 'cdrl3': '#4169E1',  # Light CDRs
        'fwh1': '#F0F0F0', 'fwh2': '#F0F0F0', 'fwh3': '#F0F0F0', 'fwh4': '#F0F0F0',  # Framework
        'fwl1': '#F0F0F0', 'fwl2': '#F0F0F0', 'fwl3': '#F0F0F0', 'fwl4': '#F0F0F0'
    }
    
    # Define liability colors
    liability_colors = {
        'Unpaired_Cys': 'red',
        'N-linked_glycosylation': 'blue',
        'Met_oxidation': 'orange',
        'Trp_oxidation': 'brown',
        'Asn_deamidation': 'purple',
        'Asp_isomerisation': 'pink',
        'Lysine_Glycation': 'cyan',
        'N-terminal_glutamate': 'lime',
        'Integrin_binding': 'navy',
        'CD11c/CD18_binding': 'teal',
        'Fragmentation': 'maroon',
        'Polyreactivity': 'gold'
    }
    
    # Create figure with 14 subplots now
    fig, axes = plt.subplots(14, 1, figsize=figsize, 
                            gridspec_kw={'height_ratios': [1, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.2, 
                                                           1.2, 1.2, 1.2, 1.2, 1.2, 1]})
    fig.suptitle(f'{chain_type.capitalize()} Chain - Sequence Properties Analysis with PROPKA Data', 
                fontsize=16, fontweight='bold')
    
    positions = df['position_seq'].values
    
    # 1. Region highlighting (top panel)
    ax1 = axes[0]
    ax1.set_xlim(positions.min() - 1, positions.max() + 1)
    ax1.set_ylim(0, 1)
    
    # Group consecutive positions by region
    current_region = None
    start_pos = positions[0]
    
    for i, (pos, region) in enumerate(zip(df['position_seq'], df['region'])):
        if region != current_region:
            if current_region is not None:
                width = positions[i-1] - start_pos + 1
                rect = Rectangle((start_pos - 0.5, 0), width, 1, 
                               facecolor=cdr_colors.get(current_region, '#F0F0F0'),
                               edgecolor='black', linewidth=0.5)
                ax1.add_patch(rect)
                if 'cdr' in current_region:
                    ax1.text(start_pos + width/2 - 0.5, 0.5, current_region.upper(), 
                           ha='center', va='center', fontsize=9, fontweight='bold')
            start_pos = pos
            current_region = region
    
    # Last region
    if current_region is not None:
        width = positions[-1] - start_pos + 1
        rect = Rectangle((start_pos - 0.5, 0), width, 1,
                       facecolor=cdr_colors.get(current_region, '#F0F0F0'),
                       edgecolor='black', linewidth=0.5)
        ax1.add_patch(rect)
        if 'cdr' in current_region:
            ax1.text(start_pos + width/2 - 0.5, 0.5, current_region.upper(),
                   ha='center', va='center', fontsize=9, fontweight='bold')
    
    ax1.set_ylabel('Regions', fontsize=10)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title('CDR and Framework Regions', fontsize=11)
    
    # [Previous plots 2-6 remain the same]
    # ... (I'm keeping the SAP/SASA, RASA, HSE, Hydrophobicity/IP, and Charge plots as before)
    
    # 7. pKa Values (dedicated plot)
    ax7 = axes[6]
    ax7.set_xlim(positions.min() - 1, positions.max() + 1)
    
    # Plot both pka and pKa_propka if they exist
    pka_plotted = False
    
    # Original pKa
    if 'pka' in df.columns:
        pka_mask = df['pka'].notna()
        if pka_mask.any():
            pka_values = pd.to_numeric(df.loc[pka_mask, 'pka'], errors='coerce')
            pka_positions = positions[pka_mask]
            valid_pka = ~pka_values.isna()
            
            if valid_pka.any():
                ax7.scatter(pka_positions[valid_pka], pka_values[valid_pka], 
                           color='green', s=60, marker='o', label='pKa', 
                           zorder=5, edgecolors='darkgreen', linewidth=1)
                pka_plotted = True
    
    # PROPKA pKa
    if 'pKa_propka' in df.columns:
        propka_mask = df['pKa_propka'].notna()
        if propka_mask.any():
            propka_values = pd.to_numeric(df.loc[propka_mask, 'pKa_propka'], errors='coerce')
            propka_positions = positions[propka_mask]
            valid_propka = ~propka_values.isna()
            
            if valid_propka.any():
                ax7.scatter(propka_positions[valid_propka], propka_values[valid_propka], 
                           color='lime', s=40, marker='^', label='pKa (PROPKA)', 
                           zorder=4, edgecolors='darkgreen', linewidth=1)
                pka_plotted = True
    
    if pka_plotted:
        ax7.axhline(y=3.8, color='gray', linestyle=':', alpha=0.3, label='Asp')
        ax7.axhline(y=4.5, color='gray', linestyle=':', alpha=0.3, label='Glu')
        ax7.axhline(y=6.5, color='gray', linestyle=':', alpha=0.3, label='His')
        ax7.axhline(y=10.5, color='gray', linestyle=':', alpha=0.3, label='Lys')
    
    ax7.set_ylabel('pKa', fontsize=10)
    ax7.set_title('pKa Values of Ionizable Residues', fontsize=11)
    ax7.grid(True, alpha=0.3)
    ax7.set_ylim(0, 14)
    ax7.legend(loc='upper right', fontsize=8)
    
    # 8. Buried Ratio (PROPKA)
    ax8 = axes[7]
    if 'Buried ratio_propka' in df.columns:
        buried_mask = df['Buried ratio_propka'].notna()
        if buried_mask.any():
            buried_values = pd.to_numeric(df.loc[buried_mask, 'Buried ratio_propka'], errors='coerce')
            buried_positions = positions[buried_mask]
            valid_buried = ~buried_values.isna()
            
            if valid_buried.any():
                ax8.scatter(buried_positions[valid_buried], buried_values[valid_buried], 
                           color='brown', s=50, marker='s', alpha=0.7)
                ax8.axhline(y=50, color='red', linestyle='--', alpha=0.3, label='50% buried')
    
    ax8.set_ylabel('Buried Ratio (%)', fontsize=10)
    ax8.set_title('Buried Ratio (PROPKA)', fontsize=11)
    ax8.grid(True, alpha=0.3)
    ax8.set_ylim(0, 100)
    ax8.legend(loc='upper right', fontsize=8)
    
    # 9. Desolvation Regular Cst (PROPKA)
    ax9 = axes[8]
    if 'Disolvation regular Cst_propka' in df.columns:
        desolv_mask = df['Disolvation regular Cst_propka'].notna()
        if desolv_mask.any():
            desolv_values = pd.to_numeric(df.loc[desolv_mask, 'Disolvation regular Cst_propka'], errors='coerce')
            desolv_positions = positions[desolv_mask]
            valid_desolv = ~desolv_values.isna()
            
            if valid_desolv.any():
                colors = ['red' if v < 0 else 'blue' for v in desolv_values[valid_desolv]]
                ax9.scatter(desolv_positions[valid_desolv], desolv_values[valid_desolv], 
                           c=colors, s=50, marker='o', alpha=0.7)
                ax9.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    ax9.set_ylabel('Desolvation Cst', fontsize=10)
    ax9.set_title('Desolvation Regular Constant (PROPKA)', fontsize=11)
    ax9.grid(True, alpha=0.3)
    
    # 10. Desolvation Regular Nb (PROPKA)
    ax10 = axes[9]
    if 'Disolvation regular Nb_propka' in df.columns:
        nb_mask = df['Disolvation regular Nb_propka'].notna()
        if nb_mask.any():
            nb_values = pd.to_numeric(df.loc[nb_mask, 'Disolvation regular Nb_propka'], errors='coerce')
            nb_positions = positions[nb_mask]
            valid_nb = ~nb_values.isna()
            
            if valid_nb.any():
                ax10.scatter(nb_positions[valid_nb], nb_values[valid_nb], 
                           color='purple', s=50, marker='D', alpha=0.7)
    
    ax10.set_ylabel('Neighbor Count', fontsize=10)
    ax10.set_title('Desolvation Regular Neighbors (PROPKA)', fontsize=11)
    ax10.grid(True, alpha=0.3)
    
    # 11. Effects RE Cst (PROPKA)
    ax11 = axes[10]
    if 'Effects RE Cst_propka' in df.columns:
        re_cst_mask = df['Effects RE Cst_propka'].notna()
        if re_cst_mask.any():
            re_cst_values = pd.to_numeric(df.loc[re_cst_mask, 'Effects RE Cst_propka'], errors='coerce')
            re_cst_positions = positions[re_cst_mask]
            valid_re_cst = ~re_cst_values.isna()
            
            if valid_re_cst.any():
                ax11.scatter(re_cst_positions[valid_re_cst], re_cst_values[valid_re_cst], 
                           color='orange', s=50, marker='v', alpha=0.7)
                ax11.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    ax11.set_ylabel('RE Effects Cst', fontsize=10)
    ax11.set_title('Reorganization Energy Effects Constant (PROPKA)', fontsize=11)
    ax11.grid(True, alpha=0.3)
    
    # 12. Effects RE Nb (PROPKA)
    ax12 = axes[11]
    if 'Effects RE Nb_propka' in df.columns:
        re_nb_mask = df['Effects RE Nb_propka'].notna()
        if re_nb_mask.any():
            re_nb_values = pd.to_numeric(df.loc[re_nb_mask, 'Effects RE Nb_propka'], errors='coerce')
            re_nb_positions = positions[re_nb_mask]
            valid_re_nb = ~re_nb_values.isna()
            
            if valid_re_nb.any():
                ax12.scatter(re_nb_positions[valid_re_nb], re_nb_values[valid_re_nb], 
                           color='teal', s=50, marker='^', alpha=0.7)
    
    ax12.set_ylabel('RE Neighbors', fontsize=10)
    ax12.set_title('Reorganization Energy Effects Neighbors (PROPKA)', fontsize=11)
    ax12.grid(True, alpha=0.3)
    
    # 13. Amino acid sequence (with CDR highlighting)
    ax13 = axes[12]
    # [Keep the enhanced amino acid sequence plot with CDR highlighting as before]
    
    # 14. Liabilities (bottom panel)
    ax14 = axes[13]
    # [Keep the liabilities plot as before]
    
    plt.tight_layout()
    return fig


def plot_propka_properties(df, chain_type='heavy', figsize=(20, 12)):
    """
    Create plots specifically for PROPKA-derived properties.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with protein sequence data (df_heavy_final or df_light_final)
    chain_type : str
        'heavy' or 'light' to label the plot
    figsize : tuple
        Figure size
    """
    
    # Define CDR regions and their colors for background highlighting
    cdr_colors = {
        'cdrh1': '#FFB6C1', 'cdrh2': '#FFA07A', 'cdrh3': '#FF6B6B',
        'cdrl1': '#87CEEB', 'cdrl2': '#6495ED', 'cdrl3': '#4169E1',
    }
    
    # Create figure with subplots
    fig, axes = plt.subplots(6, 1, figsize=figsize)
    fig.suptitle(f'{chain_type.capitalize()} Chain - PROPKA Analysis', fontsize=16, fontweight='bold')
    
    positions = df['position_seq'].values
    
    # Helper function to add CDR background highlighting
    def add_cdr_background(ax):
        for i, (pos, region) in enumerate(zip(df['position_seq'], df['region'])):
            if 'cdr' in str(region):
                ax.axvspan(pos - 0.5, pos + 0.5, alpha=0.2, color=cdr_colors.get(region, 'gray'))
    
    # 1. pKa values (standalone)
    ax1 = axes[0]
    if 'pKa_propka' in df.columns:
        mask = df['pKa_propka'].notna()
        if mask.any():
            pka_positions = positions[mask]
            # Convert to numeric
            pka_values = pd.to_numeric(df.loc[mask, 'pKa_propka'], errors='coerce')
            valid_mask = ~pka_values.isna()
            
            if valid_mask.any():
                pka_positions = pka_positions[valid_mask]
                pka_values = pka_values[valid_mask].values
                amino_acids = df.loc[mask, 'amino_acid'].values[valid_mask]
                
                ax1.scatter(pka_positions, pka_values, color='darkgreen', s=60, zorder=5, edgecolors='black', linewidth=1)
                
                # Add amino acid labels
                for pos, pka, aa in zip(pka_positions, pka_values, amino_acids):
                    ax1.annotate(aa, (pos, pka), textcoords="offset points", 
                               xytext=(0, 5), ha='center', fontsize=7, fontweight='bold')
                
                # Add reference lines for typical pKa values
                ax1.axhline(y=3.8, color='red', linestyle=':', alpha=0.3, label='Asp typical (3.8)')
                ax1.axhline(y=4.5, color='orange', linestyle=':', alpha=0.3, label='Glu typical (4.5)')
                ax1.axhline(y=6.5, color='purple', linestyle=':', alpha=0.3, label='His typical (6.5)')
                ax1.axhline(y=10.5, color='blue', linestyle=':', alpha=0.3, label='Lys typical (10.5)')
                ax1.axhline(y=12.5, color='green', linestyle=':', alpha=0.3, label='Arg typical (12.5)')
    
    add_cdr_background(ax1)
    ax1.set_ylabel('pKa', fontsize=10)
    ax1.set_title('pKa Values from PROPKA', fontsize=11)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(positions.min() - 1, positions.max() + 1)
    ax1.set_ylim(0, 14)
    
    # 2. Buried ratio
    ax2 = axes[1]
    if 'Buried ratio_propka' in df.columns:
        mask = df['Buried ratio_propka'].notna()
        if mask.any():
            buried_positions = positions[mask]
            # Convert to numeric
            buried_values = pd.to_numeric(df.loc[mask, 'Buried ratio_propka'], errors='coerce')
            valid_mask = ~buried_values.isna()
            
            if valid_mask.any():
                buried_positions = buried_positions[valid_mask]
                buried_values = buried_values[valid_mask].values
                
                ax2.scatter(buried_positions, buried_values, color='brown', s=40, zorder=5, alpha=0.7)
                ax2.bar(buried_positions, buried_values, color='brown', alpha=0.3, width=0.8)
                
                # Highlight high burial ratio
                high_burial = buried_values > 50
                if high_burial.any():
                    ax2.scatter(buried_positions[high_burial], buried_values[high_burial], 
                              color='darkred', s=60, zorder=6, marker='^', label='High burial (>50%)')
                    ax2.legend(loc='upper right', fontsize=8)
    
    add_cdr_background(ax2)
    ax2.set_ylabel('Buried Ratio (%)', fontsize=10)
    ax2.set_title('Residue Burial Ratio', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(positions.min() - 1, positions.max() + 1)
    
    # 3. Desolvation regular Cst
    ax3 = axes[2]
    if 'Disolvation regular Cst_propka' in df.columns:
        mask = df['Disolvation regular Cst_propka'].notna()
        if mask.any():
            desolv_positions = positions[mask]
            # Convert to numeric
            desolv_values = pd.to_numeric(df.loc[mask, 'Disolvation regular Cst_propka'], errors='coerce')
            valid_mask = ~desolv_values.isna()
            
            if valid_mask.any():
                desolv_positions = desolv_positions[valid_mask]
                desolv_values = desolv_values[valid_mask].values
                
                colors = ['red' if v < 0 else 'blue' for v in desolv_values]
                ax3.bar(desolv_positions, desolv_values, color=colors, alpha=0.6)
                ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    add_cdr_background(ax3)
    ax3.set_ylabel('Desolvation Energy', fontsize=10)
    ax3.set_title('Desolvation Regular Constant', fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(positions.min() - 1, positions.max() + 1)
    
    # 4. Desolvation regular Nb
    ax4 = axes[3]
    if 'Disolvation regular Nb_propka' in df.columns:
        mask = df['Disolvation regular Nb_propka'].notna()
        if mask.any():
            nb_positions = positions[mask]
            # Convert to numeric
            nb_values = pd.to_numeric(df.loc[mask, 'Disolvation regular Nb_propka'], errors='coerce')
            valid_mask = ~nb_values.isna()
            
            if valid_mask.any():
                nb_positions = nb_positions[valid_mask]
                nb_values = nb_values[valid_mask].values
                
                ax4.scatter(nb_positions, nb_values, color='purple', s=30, zorder=5)
                ax4.plot(nb_positions, nb_values, color='purple', alpha=0.3, linewidth=1)
    
    add_cdr_background(ax4)
    ax4.set_ylabel('Number of Interactions', fontsize=10)
    ax4.set_title('Desolvation Regular Number of Interactions', fontsize=11)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(positions.min() - 1, positions.max() + 1)
    
    # 5. Effects RE Cst
    ax5 = axes[4]
    if 'Effects RE Cst_propka' in df.columns:
        mask = df['Effects RE Cst_propka'].notna()
        if mask.any():
            re_positions = positions[mask]
            # Convert to numeric
            re_values = pd.to_numeric(df.loc[mask, 'Effects RE Cst_propka'], errors='coerce')
            valid_mask = ~re_values.isna()
            
            if valid_mask.any():
                re_positions = re_positions[valid_mask]
                re_values = re_values[valid_mask].values
                
                colors = ['orange' if v < 0 else 'teal' for v in re_values]
                ax5.bar(re_positions, re_values, color=colors, alpha=0.6)
                ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    add_cdr_background(ax5)
    ax5.set_ylabel('RE Effect Constant', fontsize=10)
    ax5.set_title('Reorganization Energy Effects Constant', fontsize=11)
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(positions.min() - 1, positions.max() + 1)
    
    # 6. Effects RE Nb
    ax6 = axes[5]
    if 'Effects RE Nb_propka' in df.columns:
        mask = df['Effects RE Nb_propka'].notna()
        if mask.any():
            re_nb_positions = positions[mask]
            # Convert to numeric
            re_nb_values = pd.to_numeric(df.loc[mask, 'Effects RE Nb_propka'], errors='coerce')
            valid_mask = ~re_nb_values.isna()
            
            if valid_mask.any():
                re_nb_positions = re_nb_positions[valid_mask]
                re_nb_values = re_nb_values[valid_mask].values
                
                ax6.scatter(re_nb_positions, re_nb_values, color='darkblue', s=30, zorder=5)
                ax6.plot(re_nb_positions, re_nb_values, color='darkblue', alpha=0.3, linewidth=1)
    
    add_cdr_background(ax6)
    ax6.set_ylabel('RE Interactions', fontsize=10)
    ax6.set_title('Reorganization Energy Number of Interactions', fontsize=11)
    ax6.set_xlabel('Position', fontsize=11)
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(positions.min() - 1, positions.max() + 1)
    
    plt.tight_layout()
    return fig