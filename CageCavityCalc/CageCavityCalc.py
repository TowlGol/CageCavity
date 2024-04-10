import os
import rdkit.Chem.AllChem as rdkit  # 导入 RDKit 化学库的 AllChem 模块
import rdkit.Chem.rdmolops as rdkitrdmolops  # 导入 RDKit 化学库的 rdmolops 模块
from rdkit.Geometry import Point3D  # 从 RDKit 中导入 Point3D 类
from rdkit.Chem import Descriptors  # 从 RDKit 中导入 Descriptors 模块
from rdkit import Chem  # 导入 RDKit 化学库的 Chem 模块
from copy import deepcopy  # 导入 deepcopy 函数，用于复制对象
import numpy as np  # 导入 NumPy 库，用于处理数组和矩阵
import math  # 导入 math 模块，用于数学运算
import time  # 导入 time 模块，用于计时
from scipy.spatial import KDTree  # 从 SciPy 库中导入 KDTree 类，用于快速最近邻搜索
from scipy.spatial import distance_matrix  # 从 SciPy 库中导入 distance_matrix 函数，用于计算点之间的距离矩阵
from sklearn.cluster import DBSCAN  # 导入 DBSCAN 聚类算法
from CageCavityCalc.data import hydrophValuesGhose1998, hydrophValuesCrippen1999, vdw_radii  # 导入自定义模块中的数据
from CageCavityCalc.calculations import sum_grid_volume  # 导入自定义模块中的函数
from CageCavityCalc.grid_classes import GridPoint, CageGrid  # 导入自定义模块中的类
from CageCavityCalc.hydrophobicity import assignHydrophobicValuesToCageAtoms, calc_single_hydrophobicity  # 导入自定义模块中的函数
from CageCavityCalc.electrostatics import calculate_partial_charges  # 导入自定义模块中的函数
from CageCavityCalc.input_output import read_positions_and_atom_names_from_file, read_positions_and_atom_names_from_array, print_to_file, read_cgbind, read_mdanalysis, print_pymol_file, convert_to_mol2  # 导入自定义模块中的函数
from CageCavityCalc.window_size import get_max_escape_sphere  # 导入自定义模块中的函数
from CageCavityCalc.log import logger  # 导入自定义模块中的日志记录器

class cavity():
    def __init__(self):
        '''
        Initilalize the methodology, set up all the variables
        '''
        self.grid_spacing = 1  # 网格点之间的间距
        self.distance_threshold_for_90_deg_angle = 7  # 90度角的距离阈值
        calculate_bfactor = True  # 是否计算 B 因子
        compute_aromatic_contacts = False  # 是否计算芳香性接触
        compute_hydrophobicity = True  # 是否计算疏水性
        self.dummy_atom_radii = 1  # 虚拟原子的半径
        self.clustering_to_remove_cavity_noise = "false"  # 用于移除腔内噪声的聚类方法
        self.distanceFromCOMFactor = 1  # 考虑球形腔时的质心距离因子
        self.threads_KDThree = 4  # KDTree 的线程数
        printLevel = 2  # 打印级别，1 表示正常打印，2 表示打印所有信息
        self.hydrophMethod = "Ghose"  # 疏水性计算方法，"Ghose" 或 "Crippen"
        self.positions = None  # 分子结构的位置信息
        self.atom_names = None  # 分子结构的原子名称
        self.atom_masses = None  # 分子结构的原子质量
        self.atom_vdw = None  # 分子结构的范德华半径
        self.n_atoms = 0  # 原子数量
        self.dummy_atoms_positions = []  # 虚拟原子的位置列表
        self.dummy_atoms_temperature = None  # 虚拟原子的温度
        self.volume = None  # 腔的体积
        self.filename = None  # 分子结构的文件名
        self.compute_hydrophobicity = False  # 是否计算疏水性
        self.KDTree_dict = None  # KDTree 字典
        self.distThreshold_atom_contacts = 20.0  # 原子间接触的距离阈值
        self.distance_function = "Fauchere"  # 疏水性势能的距离函数，"Audry"、"Fauchere"、"Fauchere2" 或 "OnlyValues"
        self.hydrophobicity = []  # 疏水性列表
        self.aromatic_constacts = []  # 芳香性接触列表
        self.solvent_accessibility = []  # 溶剂可及性列表
        self.esp_grid = []  # 电势网格列表

    def read_file(self, filename):
        logger.info("--- Reading the file ---")  # 记录日志
        print("--- Reading the file ---")  # 记录日志
        self.filename = filename  # 设置文件名
        # 从文件中读取位置和原子名称
        self.positions, self.atom_names, self.atom_masses, self.atom_vdw = read_positions_and_atom_names_from_file(filename)
        self.n_atoms = len(self.positions)  # 计算原子数量
        # 重置体积（如果已经计算了不同文件的体积）
        self.volume = None
        self.dummy_atoms = []

    def read_pos_name_array(self, positions, names):
        logger.info("--- Reading the arrays ---")  # 记录日志
        print("--- Reading the arrays ---")  # 记录日志
        # 从数组中读取位置和原子名称
        self.positions, self.atom_names, self.atom_masses, self.atom_vdw = read_positions_and_atom_names_from_array(np.array(positions), np.array(names))
        self.n_atoms = len(self.positions)  # 计算原子数量
        # 重置体积（如果已经计算了不同文件的体积）
        self.volume = None
        self.dummy_atoms = []

    def read_cgbind(self, cgbind_cage):
        # 从 CGbind 文件中读取位置和原子名称
        self.positions, self.atom_names, self.atom_masses, self.atom_vdw = read_cgbind(cgbind_cage)
        self.n_atoms = len(self.positions)  # 计算原子数量
        # 重置体积（如果已经计算了不同文件的体积）
        self.volume = None
        self.dummy_atoms = []

    def read_mdanalysis(self, syst):
        # 从 MDAnalysis 系统中读取位置和原子名称
        self.positions, self.atom_names, self.atom_masses, self.atom_vdw = read_mdanalysis(syst)
        self.n_atoms = len(self.positions)  # 计算原子数量
        # 重置体积（如果已经计算了不同文件的体积）
        self.volume = None
        self.dummy_atoms = []


    def volume(self):
        # 如果未计算过体积，则调用计算体积的方法，否则返回已经计算好的体积
        if self.volume is None:
            self.calculate_volume()
        else:
            return self.volume

    def calculate_volume(self):
        # 计算腔体的体积

        logger.info("--- Calculation of the cavity ---")
        print("--- Calculation of the cavity ---")

        # 确保至少有一个原子
        assert self.n_atoms > 0, "no atoms"

        start_time = time.time()

        # 计算腔体的质心和半径
        pore_center_of_mass, pore_radius = self.calculate_center_and_radius()

        # 设置网格
        calculatedGrid = self.set_up_grid(pore_center_of_mass)
        #self.create_empty_molecule()

        # 查找腔体内部的虚拟原子
        self.find_dummies_inside_cavity(calculatedGrid, pore_center_of_mass, pore_radius)

        # 求和计算腔体的体积
        self.volume = self.sum_up_volume()

        # 记录总计算时间
        logger.info(f"--- Total time {(time.time() - start_time):.0f} seconds ---" )
        print(f"--- Total time {(time.time() - start_time):.0f} seconds ---" )
        return self.volume

    def calculate_volume_by_balloon(self):
        logger.info("Start Ballon UP")
        return 

    def calculate_center_of_mass(self):
        # 计算腔体的质心
        pore_center_of_mass = np.array(sum(self.atom_masses[i]*self.positions[i] for i in range(self.n_atoms))) / sum(self.atom_masses)
        return pore_center_of_mass.tolist()
        
    def calculate_center_and_radius(self):
        # 计算腔体的质心和半径
        pore_center_of_mass = np.array(sum(self.atom_masses[i]*self.positions[i] for i in range(self.n_atoms))) / sum(self.atom_masses)
        logger.info(f"center_of_mass= {pore_center_of_mass:}")

        # 使用分子的质心作为腔体的质心
        kdtxyzAtoms = KDTree(self.positions, leafsize=20)  # 计算分子原子位置的 KD 树\
        """
        KDTree.query() 方法用于在 KD 树中查询最近邻或最近邻列表。下面是该方法的参数及其含义：
            pore_center_of_mass：查询点的坐标，即要查找最近邻的点。
            k：要返回的最近邻点的数量。如果设置为 None，则返回所有点的距离。
            p：用于计算距离的参数。默认值为 2，表示使用欧氏距离。
        """
        #因此 distancesFromCOM的作用是用于记录当前给定距离的所有点
        distancesFromCOM = kdtxyzAtoms.query(pore_center_of_mass, k=None, p=2)
        pore_radius = distancesFromCOM[0][0] * self.distanceFromCOMFactor

        # 确定质心距离的原子类型，并根据原子半径和虚拟原子半径修正腔体半径
        """
        vdw_radii[self.atom_names[distancesFromCOM[1][0]]] 表示与质心距离最近的原子的范德华半径。
        self.dummy_atom_radii 是虚拟原子的半径。   
        所谓虚拟原子半径指的是，就在空腔内滚动的虚拟圆球 
        """
        pore_radius = pore_radius - 1.01 * vdw_radii[self.atom_names[distancesFromCOM[1][0]]] - 1.01 * self.dummy_atom_radii
        if pore_radius < 0:
            pore_radius = 0

        #返回质心、半径
        return pore_center_of_mass, pore_radius


    def set_up_grid(self, pore_center_of_mass):
        # 设置网格，计算笼子的包围盒
        x_coordinates, y_coordinates, z_coordinates = zip(*self.positions)
        box_size = [min(x_coordinates), max(x_coordinates), min(y_coordinates), max(y_coordinates), min(z_coordinates), max(z_coordinates)]
        logger.info(f"Box x min/max= {box_size[0]:f}, {box_size[1]:f}")
        logger.info(f"Box y min/max= {box_size[2]:f}, {box_size[3]:f}")
        logger.info(f"Box z min/max= {box_size[4]:f}, {box_size[5]:f}")

        calculatedGrid = CageGrid(pore_center_of_mass, box_size, delta=0, grid_spacing=self.grid_spacing)
        return calculatedGrid

        # 为每种原子类型创建 KD 树，并存储在字典中
        self.atom_type_list = []  # 存储原子类型的列表
        coords_dict = {}           # 存储每种原子类型的坐标字典
        vdwR_dict = {}            # 存储每种原子类型的范德华半径字典
        self.atom_idx_dict = {}   # 存储每种原子类型的索引字典

        # 遍历每个原子，将原子位置和相关信息存储在字典中
        for atom_idx, cage_name in enumerate(self.atom_names):
            atom_type = cage_name
            pos = self.positions[atom_idx]

            if atom_type not in self.atom_type_list:
                self.atom_type_list.append(atom_type)  # 添加新的原子类型到列表中

                vdwR_dict.setdefault(atom_type, []).append(self.atom_vdw[atom_idx])  # 添加范德华半径到字典中

            coords_dict.setdefault(atom_type, []).append(list(pos))   # 添加原子位置到字典中
            self.atom_idx_dict.setdefault(atom_type, []).append(atom_idx)  # 添加原子索引到字典中

        # 创建每种原子类型的 KD 树并存储在字典中
        self.KDTree_dict = {}
        for atom_type in self.atom_type_list:
            self.KDTree_dict[atom_type] = KDTree(coords_dict[atom_type], leafsize=20)

        # 计算形成腔体的虚拟原子，同时检查原子是否与笼子重叠
        vdwRdummy = self.dummy_atom_radii  # 定义腔体中虚拟原子的半径
        for i in calculatedGird.grid:  # 遍历笼子网格中的每个点
            dist1 = np.linalg.norm(np.array(pore_center_of_mass) - np.array(i.pos))  # 计算笼子质心到当前点的距离
            if dist1 == 0:
                i.inside_cavity = 1  # 如果距离为0，则表示当前点在腔体内部
            if dist1 > 0:
                vect1Norm = (np.array(pore_center_of_mass) - np.array(i.pos)) / dist1  # 计算当前点到质心的单位向量
            if dist1 < pore_radius:
                i.inside_cavity = 1  # 如果距离小于腔体半径，则表示当前点在腔体内部
            if i.inside_cavity == 0:  # 如果当前点不在腔体内部
                summAngles = []  # 存储角度之和
                distancesAngles = []  # 存储距离加权的角度
                for atom_type in self.atom_type_list:
                    # 查询当前点附近的原子
                    xyzAtomsSet2 = self.KDTree_dict[atom_type].query(i.pos, k=None, p=2, distance_upper_bound=self.distance_threshold_for_90_deg_angle)
                    if xyzAtomsSet2[1]:
                        vdwR = vdwR_dict[atom_type][0]  # 获取当前原子类型的范德华半径
                        distThreshold = vdwR + vdwRdummy  # 计算与虚拟原子的范德华半径之和
                        if xyzAtomsSet2[0][0] < distThreshold:
                            i.overlapping_with_cage = 1  # 如果距离小于阈值，则表示当前原子与笼子重叠

                        for atom_pos_index in xyzAtomsSet2[1]:
                            atom_pos = coords_dict[atom_type][atom_pos_index]  # 获取原子的位置
                            dist2 = np.linalg.norm(np.array(atom_pos) - np.array(i.pos))  # 计算原子到当前点的距离
                            vect2Norm = (np.array(atom_pos) - np.array(i.pos)) / dist2  # 计算原子到当前点的单位向量
                            angle = np.arccos(np.dot(vect1Norm, vect2Norm))  # 计算当前点与原子的夹角
                            summAngles.append(angle)  # 将角度加入列表中
                            distancesAngles.append(1 / (1 + dist2))  # 将距离加权的角度加入列表中
                if (summAngles):
                    averageSummAngles = np.average(summAngles, axis=None, weights=distancesAngles)  # 计算加权平均角度
                    averageSummAngles_deg = np.degrees(averageSummAngles)  # 将角度转换为度
                    i.vector_angle = averageSummAngles_deg  # 存储加权平均角度
                    if i.overlapping_with_cage == 0:
                        if averageSummAngles_deg > 90:
                            i.inside_cavity = 1  # 如果加权平均角度大于90度，则表示当前点在腔体内部

        # 创建腔体虚拟原子的 KD 树
        calculatedGirdContacts = []
        for i in calculatedGird.grid:
            if i.inside_cavity == 1:
                calculatedGirdContacts.append(i.pos)  # 将腔体内部的点添加到列表中
        calculatedGirdContactsKDTree = KDTree(calculatedGirdContacts, leafsize=20)  # 创建 KD 树

        # 计算每个虚拟原子的邻居数目，从 1 到 7（其中 1 个是自己）
        for i, dummy_atom in enumerate(calculatedGird.grid):
            if dummy_atom.inside_cavity == 1:
                xyzDummySet = calculatedGirdContactsKDTree.query(dummy_atom.pos, k=None, p=2, distance_upper_bound=1.1 * self.grid_spacing)
                dummy_atom.number_of_neighbors = len(xyzDummySet[1])  # 存储邻居数目

        # 如果需要进行聚类来去除腔体噪声
        if self.clustering_to_remove_cavity_noise != "false":
            cavity_dummy_atoms_positions = []
            cavity_dummy_atoms_index = []
            for i, dummy_atom in enumerate(calculatedGird.grid):
                if dummy_atom.inside_cavity == 1:
                    cavity_dummy_atoms_positions.append([dummy_atom.x, dummy_atom.y, dummy_atom.z])  # 将腔体内部虚拟原子的位置添加到列表中
                    cavity_dummy_atoms_index.append(i)  # 将腔体内部虚拟原子的索引添加到列表中

            clusters = DBSCAN(eps=self.grid_spacing * 1.1).fit(np.array(cavity_dummy_atoms_positions))  # 使用 DBSCAN 进行聚类
            cluster_labels = clusters.labels_  # 获取聚类标签
            number_of_clusters = len(np.unique(cluster_labels))  # 获取聚类数目
            number_of_noise = np.sum(np.array(cluster_labels) == -1, axis=0)  # 获取噪声点数目
            largest_cluster = max(set(cluster_labels.tolist()), key=cluster_labels.tolist().count)  # 获取最大的聚类标签
            print('Clusters labels:', np.unique(cluster_labels))  # 打印聚类标签
            print('Number of clusters: %d' % number_of_clusters)  # 打印聚类数目
            print('Number of noise points: %d' % number_of_noise)  # 打印噪声点数目

            # 创建一个字典，存储腔体虚拟原子的索引和聚类标签
            cavity_dummy_atoms_clusters = {cavity_dummy_atoms_index[i]: cluster_labels[i] for i in range(len(cavity_dummy_atoms_index))}

        # 如果选择了根据距离删除腔体噪声
        if self.clustering_to_remove_cavity_noise == "dist":
            inv_map_cavity_dummy_atoms_clusters = {}
            for k, v in cavity_dummy_atoms_clusters.items():
                inv_map_cavity_dummy_atoms_clusters[v] = inv_map_cavity_dummy_atoms_clusters.get(v, []) + [k]

            all_grid_dummy_atoms_positions = []
            for i, dummy_atom in enumerate(calculatedGird.grid):
                all_grid_dummy_atoms_positions.append([dummy_atom.x, dummy_atom.y, dummy_atom.z])
            all_grid_dummy_atoms_positions = np.array(all_grid_dummy_atoms_positions)

            cluster_centroids = []
            cluster_centroids_distance_to_COM = []
            center_of_mass = self.calculate_center_of_mass()  # 计算笼子的质心
            for i in np.unique(cluster_labels):
                cluster_centroids.append(np.mean(all_grid_dummy_atoms_positions[inv_map_cavity_dummy_atoms_clusters[i]], axis=0))  # 计算聚类中心点

            for i in cluster_centroids:
                cluster_centroids_distance_to_COM.append(np.linalg.norm(i - center_of_mass))  # 计算聚类中心点到质心的距离

            min_cluster_centroids_distance_to_COM = cluster_centroids_distance_to_COM[1:].index(
                min(cluster_centroids_distance_to_COM[1:]))  # 获取距离质心最近的聚类中心点的索引

        # 如果选择了根据聚类大小删除腔体噪声
        if self.clustering_to_remove_cavity_noise == "size":
            print('Saving only largest cluster of the cavity')
            print('Number of points in the cluster: %d' % cluster_labels.tolist().count(largest_cluster))
        # 如果选择了根据距离删除腔体噪声
        elif self.clustering_to_remove_cavity_noise == "dist":
            print('Saving only cluster closest to the center of mass of the cavity')
            print('Number of points in the cluster: %d' % cluster_labels.tolist().count(
                min_cluster_centroids_distance_to_COM))

        # 遍历腔体内部的虚拟原子
        for i, dummy_atom in enumerate(calculatedGird.grid):
            if (dummy_atom.inside_cavity == 1 and dummy_atom.overlapping_with_cage == 0):
                # 如果选择不进行聚类或者虚拟原子的邻居数目大于1或者选择根据聚类大小删除腔体噪声且虚拟原子位于最大聚类中或者选择根据距离删除腔体噪声且虚拟原子位于距离质心最近的聚类中心
                if ((self.clustering_to_remove_cavity_noise == "false" and dummy_atom.number_of_neighbors > 1) or (
                        self.clustering_to_remove_cavity_noise == "size" and dummy_atom.number_of_neighbors > 1 and cavity_dummy_atoms_clusters[
                    i] == largest_cluster) or (
                            self.clustering_to_remove_cavity_noise == "dist" and dummy_atom.number_of_neighbors > 1 and cavity_dummy_atoms_clusters[
                        i] == min_cluster_centroids_distance_to_COM)):
                    # 将虚拟原子的位置添加到列表中
                    self.dummy_atoms_positions.append([dummy_atom.x, dummy_atom.y, dummy_atom.z])

    def sum_up_volume(self):
        logger.info("Summing the volume")  # 记录日志，提示正在计算体积

        # 计算笼体积，打印并保存
        if len(self.dummy_atoms_positions) > 0:
            cageCavityVolume = sum_grid_volume(np.array(self.dummy_atoms_positions),
                                               radius=self.dummy_atom_radii,
                                               volume_grid_size=self.grid_spacing/2)

            logger.info(f"Cage cavity volume = {cageCavityVolume:.2f}  A3")  # 记录日志，显示笼体积
            return cageCavityVolume  # 返回笼体积值
        else:
            logger.info("Cavity with no volume")  # 记录日志，提示空腔没有体积


    def get_property_values(self, property_name):
        property_values = None  # 初始化属性值为空

        # 根据属性名称获取相应的属性值
        if property_name == "aromaticity" or property_name == "aromatic_contast" or property_name == "a":
            property_values = np.append(np.zeros((len(self.positions))), self.aromatic_constacts)
        elif property_name == "solvent_accessibility" or property_name == "solvent" or property_name == "s":
            property_values = np.append(np.zeros((len(self.positions))), self.solvent_accessibility)
        elif property_name == "hydrophobicity" or property_name == "hydro" or property_name == "h":
            property_values = np.append(np.zeros((len(self.positions))), self.hydrophobicity)
        elif property_name == "electrostatics" or property_name == "esp":
            property_values = np.append(np.zeros((len(self.positions))), self.esp_grid)

        return property_values  # 返回属性值


    def print_to_file(self, filename, property_name=None):
        logger.info("Printing to file")  # 记录日志，提示正在保存到文件

        assert self.n_atoms > 0, "no atoms"  # 断言，如果没有原子则报错

        if len(self.dummy_atoms_positions) == 0:
            logger.info("No cavity, saving just the input!")  # 记录日志，提示没有空腔，只保存输入
            print_to_file(filename, self.positions, self.atom_names)  # 调用保存到文件函数，保存原子位置和名称
        else:
            property_values = self.get_property_values(property_name)  # 获取属性值

            positions = np.vstack([self.positions, self.dummy_atoms_positions])  # 堆叠位置数组
            atom_names = np.append(self.atom_names, np.array(['D'] * len(self.dummy_atoms_positions)))  # 追加原子名称数组
            print_to_file(filename, positions, atom_names, property_values)  # 调用保存到文件函数，保存位置、原子名称和属性值


    def print_to_pymol(self, filename, property_name=None):
        logger.info("Printing to pymol")  # 记录日志，提示正在保存到pymol

        property_values = self.get_property_values(property_name)  # 获取属性值

        # 首先保存为pdb文件
        positions = np.vstack([self.positions, self.dummy_atoms_positions])  # 堆叠位置数组
        atom_names = np.append(self.atom_names, np.array(['D'] * len(self.dummy_atoms_positions)))  # 追加原子名称数组
        print_to_file(filename[:filename.find('.')]+".pdb", positions, atom_names, property_values)  # 调用保存到文件函数，保存pdb文件

        # 然后保存到pymol
        if property_values is not None:
            print_pymol_file(filename, property_values[len(self.positions):], self.dummy_atom_radii)  # 调用保存到pymol函数，保存pymol文件
        else:
            print_pymol_file(filename, None, self.dummy_atom_radii)  # 调用保存到pymol函数，保存pymol文件


    def calculate_hydrophobicity(self):
        logger.info("--- Calculation of the hydrophobicity ---")  # 记录日志，提示正在计算疏水性

        if self.hydrophMethod == "Ghose":
            self.hydrophValues = hydrophValuesGhose1998
        elif self.hydrophMethod == "Crippen":
            self.hydrophValues = hydrophValuesCrippen1999

        assert self.volume is not None, "Cavity not calculated (use calculate_volume())"  # 断言，确保已计算空腔体积

        self.compute_hydrophobicity = True  # 设置标志位，表示正在计算疏水性

        if self.filename is not None and self.filename.endswith('.mol2'):
            rdkit_cage = rdkit.MolFromMol2File(self.filename, removeHs=False)
        else:
            # 使用openbabel转换为mol2格式
            logger.info("No .mol2 file, we will convert it using openbabel")  # 记录日志，提示没有.mol2文件，将使用openbabel转换
            rdkit_cage = convert_to_mol2(self.positions, self.atom_names)

        # 计算疏水性
        # 具体实现可参考实际函数代码

        return self.hydrophobicity  # 返回疏水性


    def calculate_esp(self, metal_name=None, metal_charge=None, method='eem', max_memory=1e9):
        # 计算电荷密度
        # 具体实现可参考实际函数代码


    def calculate_window(self):
        self.window_radius = get_max_escape_sphere(self.positions, self.atom_names)  # 计算逃逸球半径
        return self.window_radius  # 返回逃逸球半径


