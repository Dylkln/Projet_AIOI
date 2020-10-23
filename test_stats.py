# Modules
from aioi.files.read_file import read_json
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


def main():
	
	data_train = read_json('./Data/train.json')
	data_train = data_train[["index", "id", "sequence", 
		"structure", "predicted_loop_type"]]

	nt_index_dict = count_nt(data_train)

#	loop type :

#	E = External
#	I = Internal
#	B = Bulge
#	S = Stem-loop
#	H =	Pseudoknots
#	X = tetraboucles
#	M = Multiloop

	struct_index_dict = count_structure(data_train)
	loop_index_dict = count_loop_type(data_train)

	nt_df = pd.concat({k : pd.DataFrame.from_dict(v, "index") for k,v in nt_index_dict.items()}, axis = 1)
	struct_df = pd.concat({k : pd.DataFrame.from_dict(v, "index") for k,v in struct_index_dict.items()}, axis = 1)
	loop_df = pd.concat({k : pd.DataFrame.from_dict(v, "index") for k,v in loop_index_dict.items()}, axis = 1)

	nt_df.to_csv("nt_count.csv", index = True, header = True)
	struct_df.to_csv("struct_count.csv", index = True, header = True)
	loop_df.to_csv("loop_count.csv", index = True, header = True)

if __name__ == '__main__':
	main()