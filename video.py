from run_model import run_model_multiples
import tempfile


def make_consecutive(prototxt, weights, input_list_file, output_folder, start=0):
	"""
	input_list_file : list of photos (consecutive)
	
	"""
	
	photo_pairs = []
	with open(input_list_file, "r") as f:
		lines = [line.strip() for line in f.readlines()]
		for i in range(len(lines) - 1):
			photo_pairs.append(" ".join([lines[i], lines[i+1]]))
	print("photos : ", len(lines))

	f = tempfile.NamedTemporaryFile(delete=False)
	f.write("\n".join(photo_pairs))
	f.close()
	run_model_multiples(prototxt, weights, f.name, output_folder, blobs=[], save_image=True, start=start, save_warp=True)

