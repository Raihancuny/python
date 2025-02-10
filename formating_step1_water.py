def convert_pdb_format(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        water_residue_counter = 1  # Start numbering water residues from 1

        for line in infile:
            if not line.startswith("ATOM"):  
                continue  # Process only ATOM lines
            
            # Extract data using fixed PDB column widths
            atom_type = "HETATM "  # Extra space after "HETATM"
            atom_serial = int(line[6:11].strip()) - 1000  # Adjusting serial number
            atom_name = line[12:16].strip()
            resname = "HOH"  # Keep water residue name as HOH
            chain = line[21]  # Extract chain ID directly
            
            # Assign a new sequential residue number (always four digits, zero-padded)
            resid = f"{water_residue_counter:04d}"  
            water_residue_counter += 1  # Increment counter

            # Extract coordinates, ensuring correct column alignment
            try:
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
            except ValueError:
                print(f"Skipping line due to invalid coordinates: {line.strip()}")
                continue

            # Default values for occupancy & temp factor
            occupancy = 1.00
            temp_factor = 0.00

            # **Removed two spaces just before x-coordinate**
            formatted_line = (
                f"{atom_type:<7}{atom_serial:5d} {atom_name:<4}{resname:>3} {chain:1} {resid:>4}   "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  {occupancy:4.2f}  {temp_factor:4.2f}           O\n"
            )
            outfile.write(formatted_line)

# Run the script
convert_pdb_format("water.pdb", "output.pdb")

print("File conversion complete!")

