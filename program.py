import random
import sys
import math
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import preprocessing

"""
Pada Kode ini akan dilakukan 3 pemrosesan data, yaitu
    A. Melakukan generasi dataset untuk setiap kursus dan mengekspor ke csv
    B. Melakukan preprocessing data untuk pemformatan ke dalam .npz
    C. Melakukan evaluasi data dan prediksi data input tes dengan TensorFlow
"""

# A. Kode generasi dataset
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

    def append(self, data):
        new_node = Node(data)
        if self.head is None:
            self.head = new_node
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = new_node

def read_names_from_file(file_name):
    with open(file_name, 'r') as input_file:
        lines = list(set(line.strip() for line in input_file.readlines()))
    return lines

def generate_combined_names(names, num):
    combined_names = LinkedList()
    for _ in range(num):
        if not names:
            break
        num_strings = min(random.randint(1, 5), len(names))
        selected_names = random.sample(names, num_strings)
        combined_name = ' '.join(selected_names).strip()
        combined_id = hash(combined_name) & sys.maxsize
        target = 0  
        combined_names.append((combined_id, selected_names, target))
        names = list(set(names) - set(selected_names))
    return combined_names

def generate_course_data(num_courses):
    course_data = []
    for _ in range(num_courses):
        course_id = random.randint(1000, 9999)
        harga_kursus = (random.randint(50, 500)) * 1000
        lama_kursus = random.randint(30, 300)
        category_id = random.randint(1, 30)
        course_data.append((course_id, category_id, harga_kursus, lama_kursus))
    return course_data

def calculate_probability(num_selected_courses):
    return (3*math.exp(-(3 * num_selected_courses)))

def assign_course_ids(combined_names, course_data, num_courses):
    combined_list = LinkedList()

    current = combined_names.head
    while current:
        combined_id, combined_name, target = current.data
        num_selected_courses = 1
        while random.random() <= calculate_probability(num_selected_courses):
            num_selected_courses += 1
        selected_courses = random.sample(course_data, num_selected_courses)
        combined_list.append((combined_id, selected_courses, target))
        current = current.next

    return combined_list

def mean_dan_tambahkan_data(combined_list):
    bilangan_positif = []
    current = combined_list.head
    while current:
        combined_id, course_data, target = current.data
        for i in range(len(course_data)):
            course_id, category_id, harga_kursus, lama_kursus = course_data[i]
            persentase_penyelesaian = round(random.uniform(0, 1), 2)
            lama_user_di_kursus = round(lama_kursus * persentase_penyelesaian * 1.25)
            review_int = 1 if random.random() <= 0.2 else 0
            review_float = float(random.randint(1, 10)) if review_int == 1 else -1
            kunjungan_terakhir = random.randint(0, 365) if random.random() <= 0.8 else 0
            if review_float > 0:
                bilangan_positif.append(review_float)
            data = [combined_id, course_id, category_id, harga_kursus, persentase_penyelesaian, lama_user_di_kursus,
                    lama_kursus, review_int, review_float, kunjungan_terakhir, target]
            course_data[i] = data
        current = current.next
        if len(bilangan_positif) > 0:
            mean = sum(bilangan_positif) / len(bilangan_positif)
        else:
            mean = 0
    return mean

def ubah_tampilkan_data(combined_list, mean, file):
    current = combined_list.head
    while current:
        combined_id, course_data, target = current.data
        probabilitas_target = 0.15
        for i in range(len(course_data)):
            if course_data[i][7] == 0:
                course_data[i][8] = mean
            price2probtarget = 0.12 * (1 - (((course_data[i][3] / 50000) - 1) * (1 / 4)))
            if price2probtarget < 0:
                price2probtarget = price2probtarget * 4 / 3
            percent2probtarget = 0.1 * (((course_data[i][4] / 0.2) - 1)*(1 / 4))
            dur2probtarget = 0.1 * (course_data[i][6] / 120)
            if course_data[i][6] <= 120:
                dur2probtarget = 0.1 * (2 * course_data[i][6] - 0.01*course_data[i][6]**2)
            if mean > 5:
                rev2probtarget = 0.2 * ((course_data[i][8] - mean) / (10.0 - mean))
                if rev2probtarget < 0:
                    rev2probtarget *= (3/4)
            elif mean <= 5:
                rev2probtarget = 0.15 * ((course_data[i][8] - 5) / (10.0 - 5))
                if rev2probtarget < 0:
                    rev2probtarget *= (3/4)
            last2probtarget = 0
            if course_data[i][9] > 60:
                last2probtarget = (0.05 * (course_data[i][9]/365)) - 0.008
            probabilitas_target += (price2probtarget + percent2probtarget + dur2probtarget + rev2probtarget + last2probtarget)
            course_data[i][10] = 1 if random.random() <= probabilitas_target else 0
        current = current.next
    
    return combined_list

def ekspor_data(combined_list, file_csv, file):
    try:
        data = []
        current = combined_list.head
        while current:
            combined_id, course_data, target = current.data
            for data_row in course_data:
                data.append(data_row)
            current = current.next
        
        df = pd.DataFrame(data, columns=['Combined ID', 'Course ID', 'Category ID', 'Harga Kursus', 'Persentase Penyelesaian', 'Lama User di Kursus', 'Lama Kursus', 'Review Int', 'Review Float', 'Kunjungan Terakhir', 'Target'])
        df.to_csv(file_csv, index=False, header=False)
        
        print(f"Data berhasil diekspor ke file {file_csv}")
        return data
    except Exception as e:
        print(f"Terjadi kesalahan saat mengekspor data: {str(e)}")
        return None

def generasi_dataset(file):
    input_file = 'dictionary_nama_simplified.txt'
    num_nama = int(input("Masukkan jumlah nama diinginkan: "))
    num_courses = int(input("Masukkan jumlah kursus diinginkan: "))
    file_csv = input("Nama file .csv yang akan diekspor: ") + '.csv'
    names = read_names_from_file(input_file)
    combined_names = generate_combined_names(names, num_nama)
    course_data = generate_course_data(num_courses)
    combined_list = assign_course_ids(combined_names, course_data, num_courses)
    mean = mean_dan_tambahkan_data(combined_list)
    ubah_tampilkan_data(combined_list, mean, file)
    data_awal = ekspor_data(combined_list, file_csv, file)
    return data_awal, file_csv

# Akhir kode generasi dataset

# B. Kode preprocessing dataset
def muat_data_csv(file_csv, file):
    data_csv = np.loadtxt(file_csv, delimiter=',')
    print("Data CSV Raw:", file=file)
    for i in data_csv:
        print(i, file=file)
    return data_csv

def ambil_input_awal(data_csv, file):
    input_awal = data_csv[:,3:-1]
    data_input = input_awal.copy()
    print("Input Awal Raw:", file=file)
    for i in input_awal:
        print(i, file=file)
    return input_awal, data_input

def ambil_target_awal(data_csv, file):
    target_awal = data_csv[:,-1]
    print("Target Awal Raw: ", file=file)
    print(target_awal, file=file)
    return target_awal

def pengacakan_index(input_awal, target_awal, file):
    index_diacak = np.arange(input_awal.shape[0])
    print("Index Sebelum Diacak: ", file=file)
    print(index_diacak, file=file)
    np.random.shuffle(index_diacak)
    print("Idex Setelah Diacak: ", file=file)
    for i in index_diacak:
        print(i, file=file)
    input_diacak = input_awal[index_diacak]
    print("Input Diacak: ", file=file)
    for i in input_diacak:
        print(i, file=file)
    target_diacak = target_awal[index_diacak]
    print("Target Diacak: ", file=file)
    print(target_diacak, file=file)
    return input_diacak, target_diacak
    
def penyeimbangan_data(input_diacak, target_diacak, file):
    angka_satu = int(np.sum(target_diacak))
    angka_nol = 0
    hapus_index = []
    for i in range(target_diacak.shape[0]):
        if target_diacak[i] == 0:
            angka_nol += 1
            if angka_nol > angka_satu:
                hapus_index.append(i)
    input_imbang = np.delete(input_diacak, hapus_index, axis=0)
    print("Input Setelah Dihapus:", file=file)
    for i in input_imbang:
        print(i, file=file)
    target_imbang = np.delete(target_diacak, hapus_index, axis=0)
    print("Target Setelah Diahpus", file=file)
    print(target_imbang)
    return input_imbang, target_imbang

def pengacakan_standarisasi_data(input_imbang, target_imbang, file):
        input_standard = preprocessing.scale(input_imbang)
        print("Input Setelah Distandardisasi:", file=file)
        for i in input_standard:
            print(i, file=file)
        mean = input_imbang.mean(axis=0)
        std = input_imbang.std(axis=0)
        index_diacak = np.arange(input_standard.shape[0])
        np.random.shuffle(index_diacak)
        input_diacak = input_standard[index_diacak]
        print("Input Setelah Diacak:", file=file)
        for i in input_diacak:
            print(i, file=file)
        target_diacak = target_imbang[index_diacak]
        print("Target Setelah Diacak:", file=file)
        print(target_diacak, file=file)
        return input_diacak, target_diacak, mean, std

def bagi_data(input_diacak, target_diacak, file):
    banyak_sampel = input_diacak.shape[0]
    print("Banyak Sampel:", file=file)
    print(banyak_sampel, file=file)
    banyak_sampel_training = int(0.8 * banyak_sampel)
    print("Banyak Sampel Training:", file=file)
    print(banyak_sampel_training, file=file)
    banyak_sampel_validasi = int(0.1 * banyak_sampel)
    print("Banyak Sampel Validasi:", file=file)
    print(banyak_sampel_validasi, file=file)
    banyak_sampel_tes = banyak_sampel - banyak_sampel_training - banyak_sampel_validasi
    print("Banyak Sampel Tes:", file=file)
    print(banyak_sampel_tes, file=file)
    input_training = input_diacak[:banyak_sampel_training]
    print("Input Training:", file=file)
    print(input_training, file=file)
    target_training = target_diacak[:banyak_sampel_training]
    print("Target Training:", file=file)
    print(target_training, file=file)
    input_validasi = input_diacak[banyak_sampel_training:banyak_sampel_training+banyak_sampel_validasi]
    print("Input Validasi:", file=file)
    print(input_validasi, file=file)
    target_validasi = target_diacak[banyak_sampel_training:banyak_sampel_training+banyak_sampel_validasi]
    print("Target Validasi:", file=file)
    print(target_validasi, file=file)
    input_tes = input_diacak[banyak_sampel_training+banyak_sampel_validasi:]
    print("Input Tes:", file=file)
    print(input_tes, file=file)
    target_tes = target_diacak[banyak_sampel_training+banyak_sampel_validasi:]
    print("Target Tes:", file=file)
    print(target_tes, file=file)
    print("Pembagian Data: ", file=file)
    print(np.sum(target_training), banyak_sampel_training, np.sum(target_training) / banyak_sampel_training, file=file)
    print(np.sum(target_validasi), banyak_sampel_validasi, np.sum(target_validasi) / banyak_sampel_validasi, file=file)
    print(np.sum(target_tes), banyak_sampel_tes, np.sum(target_tes) / banyak_sampel_tes, file=file)
    return input_training, target_training, input_validasi, target_validasi, input_tes, target_tes

def ekspor_npz(file_csv, input_training, target_training, input_validasi, target_validasi, input_tes, target_tes):
    file_training = file_csv[:-4] + '_training'
    np.savez(file_training, inputs=input_training, targets=target_training)
    file_vlidasi = file_csv[:-4] + '_validasi'
    np.savez(file_vlidasi, inputs=input_validasi, targets=target_validasi)
    file_tes = file_csv[:-4] + '_tes'
    np.savez(file_tes, inputs=input_tes, targets=target_tes)
    return file_training, file_vlidasi, file_tes

def konversi_npz(file_csv, file):
    try:
        data_csv = muat_data_csv(file_csv, file)
        input_awal, data_input = ambil_input_awal(data_csv, file)
        target_awal = ambil_target_awal(data_csv, file)
        input_diacak, target_diacak = pengacakan_index(input_awal, target_awal, file)
        input_imbang, target_imbang = penyeimbangan_data(input_diacak, target_diacak, file)
        input_diacak, target_diacak, mean, std = pengacakan_standarisasi_data(input_imbang, target_imbang, file)
        input_training, target_training, input_validasi, target_validasi, input_tes, target_tes = bagi_data(input_diacak, target_diacak, file)
        file_training, file_validasi, file_tes = ekspor_npz(file_csv, input_training, target_training, input_validasi, target_validasi, input_tes, target_tes)
        return file_training, file_validasi, file_tes, data_input, mean, std
    except Exception as e:
        print(f"Terjadi kesalahan saat memuat data: {str(e)}", file=file)

# Akhir kode preprocessing dataset

# C. Kode machine learning

def hashmap(data_input, data_awal):
    diksi = {}
    for i in range(len(data_awal)):
        diksi[str(i)] = []
        diksi[str(i)].append(data_awal[i])
        diksi[str(i)].append(list(data_input[i]))  
    return diksi

def bongkar_data(file_training, file_validasi, file_tes):
    npz = np.load(file_training + ".npz")
    input_training, target_training = npz['inputs'].astype(float), npz['targets'].astype(int)
    print("Input Training:", file=file)
    print(input_training, file=file)
    print("Target Training:", file=file)
    print(target_training, file=file)
    npz = np.load(file_validasi + '.npz')
    input_validasi, target_validasi = npz['inputs'].astype(float), npz['targets'].astype(int)
    print("Input Validasi:", file=file)
    print(input_validasi, file=file)
    print("Target Validasi:", file=file)
    print(target_validasi, file=file)
    npz = np.load(file_tes + '.npz')
    input_tes, target_tes = npz['inputs'].astype(float), npz['targets'].astype(int)
    print("Input Tes:", file=file)
    print(input_tes, file=file)
    print("Target Tes:", file=file)
    print(target_tes, file=file)
    return input_training, target_training, input_validasi, target_validasi, input_tes, target_tes

def infrastruktur_model():
    output_size = 1
    hidden_layer_size = 16
    # Un-comment kode di bawah ini untuk mencoba dengan ukuran lainnya
    # output_size = int(input("Masukkan ukuran untuk model outputnya"))
    # hidden_layer_size = int(input("Masukkan untuk ukuran model yang disembunyikan"))
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
        tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
        tf.keras.layers.Dense(output_size, activation='softmax')
    ])
    return model

def kompiler_model(model):
    optimizer = 'adam'
    loss = 'binary_crossentropy'
    metrics = ['accuracy']
    # # Un-comment kode di bawah ini untuk coba-coba
    # optimizer = input('Masukkan optimizer yang akan digunakan: ')
    # loss = input('Masukkan kompiler loss yang akan digunakan: ')
    # metrics = input('Masukkan metrik yang akan digunakan: ')
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model

def training_model(input_training, target_training, input_validasi, target_validasi, model):
    batch_size = 32
    max_epochs = 100
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=2)
    # Un-comment kode di bawah ini untuk coba-coba
    # batch_size = int(input("Masukkan banyaknya data maksimal yang dikerjakan per epoch: "))
    # max_epochs = int(input("Masukkan banyaknya iterasi yang dilakukan: "))
    # patience = int(input("Masukkan banyaknya data maksimal yang dikerjakan per epoch: "))
    # early_stopping = tf.keras.callbacks.EarlyStopping(patience=patience)
    model.fit(
            input_training,
            target_training,
            batch_size,
            max_epochs,
            callbacks=[early_stopping],
            validation_data=(input_validasi, target_validasi),
            verbose = 2
            )  
    return model 

def prediksi(model, input_tes, diksi, mean, std):
    predictions = model.predict(input_tes)
    predicted_targets = np.argmax(predictions, axis=1)
    predicted_list = input_tes[predicted_targets == 1]
    restored_data = predicted_list * std + mean
    predicted_list_original = []
    for i in range(len(restored_data)):
        for key, value in diksi.items():
            if np.array_equal(restored_data[i], value[1]):
                predicted_list_original.append(value[0])
    # Print out the predicted list in the original form.
    print("List prediksi dalam bentuk data awal:", file=file)
    for i in range(len(predicted_list_original)):
        print(predicted_list_original[i], file=file)

def machine_learning(file_training, file_validasi, file_tes, file, data_awal, data_input, mean, std):
    diksi = hashmap(data_input, data_awal)
    input_training, target_training, input_validasi, target_validasi, input_tes, target_tes = bongkar_data(file_training, file_validasi, file_tes)
    model = infrastruktur_model()
    model = kompiler_model(model)
    model = training_model(input_training, target_training, input_validasi, target_validasi, model)
    test_loss, test_accuracy = model.evaluate(input_tes, target_tes)
    print('\nTest loss: {0:.2f}. Test accuracy: {1:.2f}%'.format(test_loss, test_accuracy*100.), file=file)
    prediksi(model, input_tes, diksi, mean, std)

# Akhir kode machine learning

file = open("log1.txt", "w") # Comment bagian ini untuk menonaktifkan log

# Memanggil fungsi main untuk meghasilkan dataset
# A. Unpacking data dari dataset yang dibuat
data_awal, file_csv = generasi_dataset(file) 
# file_csv = input("Masukkan file csv yang akan digunakan: ")  
# /A. Comment dan un-comment kode di atas ini untuk menggunakan data sebelumnya

# Memprooses data .csv untuk diekspor ke dalam .npz
# B. Preprocessing data .csv ke .npz
file_training, file_validasi, file_tes, data_input, mean, std = konversi_npz(file_csv, file) 
# /B. 

# C. Melakukan analisis data dengan secara supervised dan unsupervised
machine_learning(file_training, file_validasi, file_tes, file, data_awal, data_input, mean, std)
# /C.

file.close()
