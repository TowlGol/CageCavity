load cage_cavity.pdb
extract cavity, resname CV
alter name D, vdw=1
show_as surface, cavity
