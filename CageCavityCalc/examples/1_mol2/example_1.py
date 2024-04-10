from CageCavityCalc.CageCavityCalc import cavity
import time  # 导入 time 模块，用于计时
import logging


cav = cavity()
#cav.read_file("cage_1_JACS_2006_128_14120.mol2")
cav.read_file("file.mol2")
start_time = time.time()
volume = cav.calculate_volume()
print("Volume=", volume)
print(f"--- Total time {(time.time() - start_time):.0f} seconds ---" )
cav.print_to_file("cage_cavity.pdb")
cav.print_to_pymol("cage_cavity.pml")
