# Modules
from files.read_file import read_json
from statistics import mean
import pandas as pd


def count_nt(data_train):
	"""

	"""
	nucleotide = ["A", "U", "C", "G"]

	nt_index_dict = {}
	sequences = []
	for seq in data_train["sequence"]:
		sequences.append(seq)

	for seq in sequences:
		for i, nt in enumerate(seq):
			pos = i + 1
			
			if pos not in nt_index_dict.keys():
				nt_index_dict[pos] = {}
			
			if nt not in nt_index_dict[pos].keys():
				nt_index_dict[pos][nt] = 0
			nt_index_dict[pos][nt] += 1


	for nt in nucleotide:
		for pos in nt_index_dict.keys():	
			if nt not in nt_index_dict[pos]:
				nt_index_dict[pos][nt] = 0


	return nt_index_dict


def count_structure(data_train):
	"""
	
	"""
	structs = [".", "(", ")"]

	struct_index_dict = {}
	structures = []
	
	for struct in data_train["structure"]:
		structures.append(struct)

	for structure in structures:
		for i, struct in enumerate(structure):
			pos = i + 1

			if pos not in struct_index_dict.keys():
				struct_index_dict[pos] = {}
			
			if struct not in struct_index_dict[pos].keys():
				struct_index_dict[pos][struct] = 0
			struct_index_dict[pos][struct] += 1

	
	for s in structs:
		for pos in struct_index_dict.keys():
			if s not in struct_index_dict[pos]:
				struct_index_dict[pos][s] = 0


	return struct_index_dict			


def count_loop_type(data_train):
	"""
	
	"""
	loops = ["E", "I", "B", "S", "H", "X", "M"]

	loop_index_dict = {}
	predicted_loops = []
	
	for loop in data_train["predicted_loop_type"]:
		predicted_loops.append(loop)

	for loop in predicted_loops:
		for i, loop_type in enumerate(loop):
			pos = i + 1
			
			if pos not in loop_index_dict.keys():
				loop_index_dict[pos] = {}
			
			if loop_type not in loop_index_dict[pos].keys():
				loop_index_dict[pos][loop_type] = 0
			loop_index_dict[pos][loop_type] += 1
	
	for l in loops:
		for pos in loop_index_dict.keys():
			if l not in loop_index_dict[pos]:
				loop_index_dict[pos][l] = 0

	return loop_index_dict


def save_loop(data_train_deg):
	"""

	"""

	len_deg = len(data_train_deg["deg_50C"][0])

	predicted_loops = [loop[:len_deg] for loop in data_train_deg["predicted_loop_type"]]

	sequences = [seq[:len_deg] for seq in data_train_deg["sequence"]]

	reactivities = [react[:len_deg] for react in data_train_deg["reactivity"]]
	reactivities_err = [react_err[:len_deg] for react_err in data_train_deg["reactivity_error"]]

	degradations_pH10 = [deg for deg in data_train_deg["deg_pH10"]]
	degradations_50C = [deg for deg in data_train_deg["deg_50C"]]
	
	deg_error_pH10 = [deg_error for deg_error in data_train_deg["deg_error_pH10"]]
	deg_error_50C = [deg_error for deg_error in data_train_deg["deg_error_50C"]]

	loops_seq = {}
	loops_deg_pH10 = {}
	loops_deg_50C = {}
	loops_reac = {}
	loops_reac_error = {}
	loops_deg_error_pH10 = {}
	loops_deg_error_50C = {}

	for i, loop in enumerate(predicted_loops):
		for j, ltype in enumerate(loop):
			
			if ltype not in loops_seq.keys():
				
				loops_seq[ltype] = []
				loops_deg_50C[ltype] = []
				loops_deg_pH10[ltype] = []
				loops_reac[ltype] = []
				loops_reac_error[ltype] = []
				loops_deg_error_50C[ltype] = []
				loops_deg_error_pH10[ltype] = []
			
			loops_seq[ltype].append(sequences[i][j])
			loops_deg_50C[ltype].append(degradations_50C[i][j])
			loops_deg_pH10[ltype].append(degradations_pH10[i][j])
			loops_reac[ltype].append(reactivities[i][j])
			loops_reac_error[ltype].append(reactivities_err[i][j])
			loops_deg_error_50C[ltype].append(deg_error_50C[i][j])
			loops_deg_error_pH10[ltype].append(deg_error_pH10[i][j])


	return loops_seq, loops_reac, loops_reac_error, loops_deg_pH10, \
	loops_deg_50C, loops_deg_error_pH10, loops_deg_error_50C


def count_nt_pred_JP(jp_pred, data_train):
	"""

	"""
	loops_pred_jp = [loop for loop in jp_pred["loop_pred"]]
	sequences = [seq for seq in data_train["sequence"]]

	loop_seq = {}

	for i, loop in enumerate(loops_pred_jp):
		for j, ltype in enumerate(loop):
			if ltype not in loop_seq.keys():
				loop_seq[ltype] = []

			loop_seq[ltype].append(sequences[i][j])


	count_l_s = count_dict_loop_seq(loop_seq)


	return count_l_s


def extract_diff_between_two_pred(data_train, jp_pred):
	"""

	"""
	loops_init = [loop for loop in data_train["predicted_loop_type"]]
	loops_pred_jp = [loop for loop in jp_pred["loop_pred"]]

	struct_init = [s for s in data_train["structure"]]
	struct_pred_jp = [s for s in jp_pred["structure_pred"]]

	ids = [i for i in data_train["id"]]

	diff_struct = {}
	indices = range(107)
	difference_struct = {}.fromkeys(set(indices), 0)



	for i, struct in enumerate(struct_pred_jp):
		for j, s in enumerate(struct):

			if ids[i] not in diff_struct.keys():
				diff_struct[ids[i]] = {}
			
				if j not in diff_struct[ids[i]].keys():
					diff_struct[ids[i]][j] = ""

			if s != struct_init[i][j]:
				diff_struct[ids[i]][j] = f"'{s}' --> '{struct_init[i][j]}'"
				difference_struct[j] += 1

			else:
				diff_struct[ids[i]][j] = "="


	diff_loop = {}
	difference_loop = {}.fromkeys(set(indices), 0)


	for i, loop in enumerate(loops_pred_jp):
		for j, l in enumerate(loop):

			if ids[i] not in diff_loop.keys():
				diff_loop[ids[i]] = {}
			
				if j not in diff_loop[ids[i]].keys():
					diff_loop[ids[i]][j] = ""

			if l != loops_init[i][j]:
				diff_loop[ids[i]][j] = f"{l} --> {loops_init[i][j]}"
				difference_loop[j] += 1

			else:
				diff_loop[ids[i]][j] = "="

	
	
	return diff_struct, difference_struct, diff_loop, difference_loop


def count_dict_loop_seq(loops_seq):
	"""

	"""
	count_loop_seq = {}

	for cle, valeur in loops_seq.items():
		if cle not in count_loop_seq.keys():
			count_loop_seq[cle] = {}
			for val in valeur:
				if val not in count_loop_seq[cle].keys():
					count_loop_seq[cle][val] = 0
				count_loop_seq[cle][val] += 1

	return count_loop_seq


def calc_mean(dict1, dict2):
	"""

	"""
	mean_dict1 = {}
	mean_dict2 = {}

	for cle in dict1.keys():
		if cle not in mean_dict1.keys():
			mean_dict1[cle] = round(mean(dict1[cle]), 4)

	for cle in dict2.keys():
		if cle not in mean_dict2.keys():
			mean_dict2[cle] = round(mean(dict2[cle]), 4)


	return mean_dict1, mean_dict2


def main():
	
#	loop type :

#	E = End
#	I = Internal
#	B = Bulge
#	S = Stem-loop
#	H = hairpin
#	X = external
#	M = Multiloop

	
	##### SAVE DATA #####

	data_train = read_json('../Data/train.json')
	data_train = data_train.query("SN_filter == 1")
	
	jp_pred = pd.read_csv("../Data/spotrna_train.tsv", sep = "\t", header = None)
	jp_pred.columns = ["id_pred", "structure_pred", "loop_pred"]

	data_train_deg = data_train[["sequence", "predicted_loop_type",
	"reactivity_error", "reactivity", "deg_error_pH10", "deg_error_50C",
	"deg_pH10", "deg_50C"]]

	data_train = data_train[["index", "id", "sequence", 
		"structure", "predicted_loop_type"]]

	

	##### CREATE DICT #####

	nt_index_dict = count_nt(data_train)

	loops_seq, loops_reac, loops_reac_error, \
	loops_deg_pH10, loops_deg_50C, loops_deg_error_pH10, \
	loops_deg_error_50C = save_loop(data_train_deg)

	count_loop_seq = count_dict_loop_seq(loops_seq)
	mean_reac, mean_reac_error = calc_mean(loops_reac, loops_reac_error)
	mean_deg_50C, mean_deg_error_50C = calc_mean(loops_deg_50C, loops_deg_error_50C)
	mean_deg_pH10, mean_deg_error_pH10 = calc_mean(loops_deg_pH10, loops_deg_error_pH10)

	diff_struct, difference_struct, diff_loop, difference_loop = extract_diff_between_two_pred(data_train, jp_pred)

	struct_index_dict = count_structure(data_train)
	loop_index_dict = count_loop_type(data_train)

	count_l_s = count_nt_pred_JP(jp_pred, data_train)
	
	##### CREATE DATAFRAME #####

	count_ls_JP_df = pd.DataFrame(count_l_s)

	nt_df = pd.DataFrame(nt_index_dict)
	struct_df = pd.DataFrame(struct_index_dict)
	loop_df = pd.DataFrame(loop_index_dict)
	
	count_df = pd.DataFrame(count_loop_seq)
	
	row4 = pd.Series(data = mean_reac, name = "mean reactivity")
	count_df = count_df.append(row4)

	row5 = pd.Series(data = mean_reac_error, name = "mean reactivity error")
	count_df = count_df.append(row5)

	row6 = pd.Series(data = mean_deg_50C, name = "mean deg 50C")
	count_df = count_df.append(row6)

	row7 = pd.Series(data = mean_deg_error_50C, name = "mean deg error 50C")
	count_df = count_df.append(row7)

	row8 = pd.Series(data = mean_deg_pH10, name = "mean deg pH10")
	count_df = count_df.append(row8)

	row9 = pd.Series(data = mean_deg_error_pH10, name = "mean deg error pH10")
	count_df = count_df.append(row9)

	diff_struct_df = pd.DataFrame(diff_struct)
	diff_loop_df = pd.DataFrame(diff_loop)

	diff_struct_tot_df = pd.DataFrame.from_dict(difference_struct, orient = "index")
	diff_loop_tot_df = pd.DataFrame.from_dict(difference_loop, orient = "index")
	

	##### DATAFRAME TO CSV #####

	count_ls_JP_df.to_csv("../Stats/count_new_pred.csv", index = True, header = True)
	diff_struct_df.to_csv("../Stats/diff_struct.csv", index = True, header = True)
	diff_loop_df.to_csv("../Stats/diff_loop.csv", index = True, header = True)
	diff_struct_tot_df.to_csv("../Stats/diff_struct_par_i.csv", index = True, header = True)
	diff_loop_tot_df.to_csv("../Stats/diff_loop_par_i.csv", index = True, header = True)
	count_df.to_csv("../Stats/count.csv", index = True, header = True)
	nt_df.to_csv("../Stats/nt_count.csv", index = True, header = True)
	struct_df.to_csv("../Stats/struct_count.csv", index = True, header = True)
	loop_df.to_csv("../Stats/loop_count.csv", index = True, header = True)

if __name__ == '__main__':
	main()
