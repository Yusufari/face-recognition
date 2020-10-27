# --------------- Yusuf Ari Bahtiar -----------------
# ---------- Adrian Rosebock - Source Code ----------

#impor paket / library
from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2

# kontruksi argumen parse
ap = argparse.ArgumentParser()
ap.add_argument("-c","--cascade", required=True, 
		help = "haarcascade_frontalface_default.xml")
ap.add_argument("-e","--enkodings", required=True, help = "enkodings.pickle")
args = vars(ap.parse_args())

# mengambil wajah yang sudah diketahui dengan OpenCV's Haar
print("[INFO] ambil enkoding + pendeteksi wajah ....")
data = pickle.loads(open(args["enkodings"],"rb").read())
pendeteksi = cv2.CascadeClassifier(args["cascade"])

# inisialisasi video stream dan  mengaktifkan kamera
print("[INFO] memulai video stream...")
vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

# mulai hitung FPS
fps = FPS().start()

# proses perulangan pembingkaian file video
while True:
	#ambil gambar dari video dan diperkecil ukurannya ke 500px 
	# untuk mempercepat proses
	figur = vs.read()
	figur = imutils.resize(figur, width=500)
	
	#mengubah input figur dari BGR ke keabuan & BGR ke RGB
	gray = cv2.cvtColor (figur, cv2.COLOR_BGR2GRAY)
	rgb = cv2.cvtColor(figur, cv2.COLOR_BGR2RGB)
	
	#deteksi wajah pada figur keabuan
	defa = pendeteksi.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
			minSize=(30,30)) # ~ ~ , flags=cv2.CASCADE_SCALE_IMAGE)
	
	#menentukan kordinat kotak deteksi berupa (atas, kanan, bawah, kiri)
	kotak = [(y,x + w, y + h, x) for (x,y,w,h) in defa]
	
	#komputasi wajah melalui tangkapan kotak
	enkodings = face_recognition.face_encodings(rgb, kotak)
	namas = []
	
	#perulangan embed wajah
	for enkoding in enkodings:
			#mencocokan setiap input gambar dengan database enkoding
			cocok = face_recognition.compare_faces(data["enkodings"], enkoding)
			nama = "Kamu Siapa"
			
			#Cek jika menemukan kecocokan
			if True in cocok:
					#temukan indek kecocokan dan menghitung nilai kecocokan
					cocokInd = [i for (i,b) in enumerate(cocok) if b]
					hitung = {}
					
					#ulangi pencocokan untuk menemukan indeks yang benar-benar pas
					for i in cocokInd:
							nama = data["namas"][i]
							hitung[nama] = hitung.get(nama,0) + 1
					
					#menentukan kecocokan dari nilai maksimal
					nama = max(hitung, key=hitung.get)
			
			#update daftar nama
			namas.append(nama)
	
	#mengatur penampilan kotak  pencocokan wajah
	for ((atas, kanan, bawah, kiri), nama) in zip(kotak,namas):
			#Tulis prediksi nama
			cv2.rectangle(figur,(kiri, atas), (kanan, bawah), (0, 255, 0), 2)
			y = atas - 15 if atas - 15 > 15 else atas + 15
			cv2.putText(figur, nama, (kiri,y), cv2.FONT_HERSHEY_SIMPLEX, 
					0.75, (0, 255, 0), 2)
			
	#tampilkan kotak di layar
	cv2.imshow("Figur", figur)
	key = cv2.waitKey(1) & 0xFF
	
	#jika 'q' ditekan, pengulangan berhenti
	if key == ord("q"):
			break
	
	#memperbarui penghitung FPS
	fps.update()

#menghentikan display informasi
fps.stop()
print("[INFO]  waktu dibutuhkan: {:.2f}".format(fps.elapsed()))
print("[INFO] perkiraan FPS: {:.2f}".format(fps.fps()))

#bersihkan layar
cv2.destroyAllWindows()
vs.stop()