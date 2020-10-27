# --------------- Yusuf Ari Bahtiar -----------------
# ---------- Adrian Rosebock - Source Code ----------

# impor paket yang dibutuhkan
from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os

# persiapan kontruksi bahan
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="dataset")
ap.add_argument("-e", "--enkoding", required=True, help="enkoding_rupa.py")
ap.add_argument("-m", "--metode-deteksi", type=str, default="hog", help="hog")
args = vars(ap.parse_args())

# input gambar dataset
print("[INFO] jumlah gambar...")
imagePaths = list(paths.list_images(args["dataset"]))

# inisialisasi enkoding dan nama
knownEncodings = []
knownNames = []

# perulangan direktori gambar
for (i, imagePath) in enumerate(imagePaths):
	# ekstrasi nama dari direktori
	print("[INFO] gambar diproses {}/{}".format(i + 1,len(imagePaths)))
	nama = imagePath.split(os.path.sep) [-2]
	
	# mengambil gambar dan mengubah dari BGR (OpenCV)
	# ke dlib (RGB)
	gambar = cv2.imread(imagePath)
	rgb = cv2.cvtColor(gambar, cv2.COLOR_BGR2RGB)
	
	# deteksi kordinat (x,y) pada kotak untuk setiap wajah
	kotak = face_recognition.face_location(rgb, model=args["metode_deteksi"])
	
	# menghitung pencocokan  wajah
	enkodings = face_recognition.face_encodings(rgb, kotak)
	
	# perulangan proses enkoding
	for enkoding in enkodings:
		#menambahkan setiap enkoding + nama
		knownEncodings.append(enkoding)
		knownNames.append(nama)

#menyimpan enkoding wajah
print("[INFO] enkoding serial data....")
data = {"enkodings": knownEncodings, "namas": knownNames}
f = open(args["enkodings"],"wb")
f.write(pickle.dumps(data))
f.close()	
	
	