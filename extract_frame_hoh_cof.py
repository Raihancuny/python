import mdtraj as md
import MDAnalysis as mda
import os

# Load the DCD file
traj = md.load_dcd('/home/muddin/raihan_work/ndh-1/md_analysis/run_ndh1ms_last100PatriciaSaura.dcd', 
                    top='/home/muddin/raihan_work/ndh-1/md_analysis/ndh-1msPatriciaSaura_parmed.psf')

# Specify the frames you want
frames_wanted = [10]

# Loop over the desired frames
for frame in frames_wanted:
    # Create a directory for each frame
    directory_name = f'fr{frame}'
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)

    # Extract the specific frame and save it as a PDB file using mdtraj
    pdb_filename = os.path.join(directory_name, f'frame{frame}.pdb')
    frame_traj = traj[frame]
    frame_traj.save_pdb(pdb_filename)

    # Load the newly saved PDB frame with MDAnalysis
    u = mda.Universe(pdb_filename)

    # Select specific subunits NDHA, NDHC, NDHG, NDHE, NDHL, NDHH, and cofactors (PL9)
    subunits = u.select_atoms("segid NDHA NDHC NDHG NDHE NDHL NDHH or resname PL9")

    # Select all water molecules within 5 Å of any selected subunit (allow duplicates)
    waters = u.select_atoms("resname HOH and around 5.0 (segid NDHA NDHC NDHG NDHE NDHL NDHH or resname PL9)")

    # Renumber water residues uniquely
    for i, residue in enumerate(waters.residues):
        residue.resid = 10000 + i  # Assign new unique residue numbers starting from 10000

    # Combine subunits and renumbered water molecules
    combined_selection = subunits + waters

    # Save the cleaned-up subunit PDB with uniquely numbered waters
    output_filename = os.path.join(directory_name, f'fr{frame}_subunits_waters_renum.pdb')
    combined_selection.write(output_filename)

    # Optionally, save another PDB containing only the selected core subunits
    selected_pdb_filename = os.path.join(directory_name, f'fr{frame}_acge.pdb')
    selected_subunits = u.select_atoms("segid NDHA NDHC NDHG NDHE NDHL")
    selected_subunits.write(selected_pdb_filename)

    print(f"Frame {frame} processed and saved in directory '{directory_name}'.")

print("All frames processed and saved in individual directories.")

