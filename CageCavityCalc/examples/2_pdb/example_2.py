from CageCavityCalc.CageCavityCalc import cavity

cav = cavity()
cav.read_file("cage_ACIE_2006_45_901.pdb")
cav.calculate_volume()

# 设置 `k` 的值为 5
pore_center_of_mass, pore_radius = cav.calculate_center_and_radius(k=5)

cav.print_to_file("cage_cavity.pdb")
print(f"Cavity volume: {cav.volume}")

