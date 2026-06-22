import streamlit as st

# 1. Konfigurasi Halaman & Menu Utama (Pastikan mentok kiri, tanpa spasi di baris pertama)
st.set_page_config(
    page_title="Studi Kasus Linked List",
    page_icon="📚",
    layout="wide"
)

menu = st.sidebar.selectbox(
    "Pilih Studi Kasus",
    [
        "Playlist Musik",
        "Browser History",
        "Daftar Anggota Koperasi",
        "Reservasi Hotel",
        "Antrian Rumah Sakit",
        "Editor Undo Redo"
    ]
)


# ==============================================================================
# 2. Modul 1: Playlist Musik
# ==============================================================================
if menu == "Playlist Musik":
    st.title("🎵 Playlist Musik")

    class NodePlaylist:
        def __init__(self, lagu):
            self.lagu = lagu
            self.next = None

    class Playlist:
        def __init__(self):
            self.head = None

        def tambah_lagu(self, lagu):
            node_baru = NodePlaylist(lagu)

            if self.head is None:
                self.head = node_baru
            else:
                temp = self.head
                while temp.next:
                    temp = temp.next
                temp.next = node_baru

        def get_playlist(self):
            daftar_lagu = []
            temp = self.head

            while temp:
                daftar_lagu.append(temp.lagu)
                temp = temp.next

            return daftar_lagu

    # Simpan playlist dalam session state
    if "playlist" not in st.session_state:
        st.session_state.playlist = Playlist()

    playlist = st.session_state.playlist

    # Input lagu
    st.subheader("Tambah Lagu")
    lagu = st.text_input("Masukkan Judul Lagu")

    if st.button("Tambah Lagu"):
        if lagu:
            playlist.tambah_lagu(lagu)
            st.success(f"Lagu '{lagu}' berhasil ditambahkan")
        else:
            st.warning("Masukkan judul lagu terlebih dahulu")

    # Tampilkan playlist
    st.subheader("Playlist Musik")
    daftar_lagu = playlist.get_playlist()

    if daftar_lagu:
        for i, lagu_item in enumerate(daftar_lagu, start=1):
            st.write(f"{i}. {lagu_item}")
    else:
        st.info("Playlist kosong")


# ==============================================================================
# 3. Modul 2: Browser History
# ==============================================================================
elif menu == "Browser History":
    st.title("🌐 Browser History")

    class NodeHalaman:
        def __init__(self, url, judul):
            self.url = url
            self.judul = judul
            self.next = None

    class BrowserHistory:
        def __init__(self):
            self.head = None
            self.current = None

        def kunjungi(self, url, judul):
            node_baru = NodeHalaman(url, judul)

            if self.head is None:
                self.head = node_baru
                self.current = node_baru
            else:
                temp = self.head
                while temp.next:
                    temp = temp.next
                temp.next = node_baru
                self.current = node_baru

        def back(self):
            if self.head is None or self.head is self.current:
                return False

            temp = self.head
            while temp.next and temp.next is not self.current:
                temp = temp.next

            self.current = temp
            return True

        def get_riwayat(self):
            data = []
            temp = self.head

            while temp:
                aktif = temp is self.current
                data.append((temp.judul, temp.url, aktif))
                temp = temp.next

            return data

        def cari_riwayat(self, keyword):
            hasil = []
            temp = self.head

            while temp:
                if keyword.lower() in temp.url.lower() or keyword.lower() in temp.judul.lower():
                    hasil.append((temp.judul, temp.url))
                temp = temp.next

            return hasil

    # Simpan objek browser
    if "browser" not in st.session_state:
        st.session_state.browser = BrowserHistory()

    browser = st.session_state.browser

    # Form kunjungi halaman
    st.subheader("Kunjungi Halaman")
    judul = st.text_input("Judul Halaman", key="browser_judul")
    url = st.text_input("URL", key="browser_url")

    if st.button("Kunjungi"):
        if judul and url:
            browser.kunjungi(url, judul)
            st.success(f"Berhasil membuka [{judul}]")
        else:
            st.warning("Isi judul dan URL terlebih dahulu.")

    # Tombol back
    if st.button("⬅ Back"):
        if browser.back():
            st.success("Kembali ke halaman sebelumnya.")
        else:
            st.warning("Tidak ada halaman sebelumnya.")

    # Halaman aktif
    st.subheader("Halaman Aktif")
    if browser.current:
        st.write(f"**{browser.current.judul}**")
        st.write(browser.current.url)
    else:
        st.info("Belum ada halaman yang dikunjungi.")

    # Riwayat
    st.subheader("Riwayat Kunjungan")
    riwayat = browser.get_riwayat()

    if riwayat:
        for i, (jdl, link, aktif) in enumerate(riwayat, 1):
            if aktif:
                st.write(f"{i}. [{jdl}] {link} ⭐")
            else:
                st.write(f"{i}. [{jdl}] {link}")
    else:
        st.info("Riwayat kosong.")

    # Pencarian
    st.subheader("Cari Riwayat")
    keyword = st.text_input("Masukkan kata kunci", key="browser_keyword")

    if st.button("Cari"):
        hasil = browser.cari_riwayat(keyword)
        if hasil:
            st.success(f"Ditemukan {len(hasil)} hasil")
            for jdl, link in hasil:
                st.write(f"- [{jdl}] {link}")
        else:
            st.warning("Tidak ditemukan hasil pencarian.")


# ==============================================================================
# 4. Modul 3: Daftar Anggota Koperasi
# ==============================================================================
elif menu == "Daftar Anggota Koperasi":
    st.title("👥 Daftar Anggota Koperasi")

    # Inisialisasi list Python untuk session state
    if "anggota" not in st.session_state:
        st.session_state.anggota = []

    # Form tambah anggota
    st.subheader("Tambah Anggota")
    nama_baru = st.text_input("Masukkan nama anggota", key="koperasi_nama")

    if st.button("➕ Tambah Anggota"):
        if nama_baru:
            st.session_state.anggota.append(nama_baru)
            st.success(f"{nama_baru} berhasil ditambahkan.")
        else:
            st.warning("Masukkan nama anggota terlebih dahulu.")

    # Hapus anggota
    st.subheader("Hapus Anggota")
    if st.session_state.anggota:
        anggota_hapus = st.selectbox(
            "Pilih anggota yang akan dihapus",
            st.session_state.anggota
        )

        if st.button("🗑 Hapus Anggota"):
            st.session_state.anggota.remove(anggota_hapus)
            st.success(f"{anggota_hapus} berhasil dihapus.")
            st.rerun()
    else:
        st.info("Belum ada anggota.")

    # Tampilkan daftar anggota
    st.subheader("Daftar Anggota")
    if st.session_state.anggota:
        for i, agt in enumerate(st.session_state.anggota, start=1):
            st.write(f"{i}. {agt}")
    else:
        st.write("Daftar anggota masih kosong.")


# ==============================================================================
# 5. Modul 4: Sistem Reservasi Hotel
# ==============================================================================
elif menu == "Reservasi Hotel":
    st.title("🏨 Sistem Reservasi Hotel")
    st.write("Implementasi Linked List untuk manajemen reservasi hotel.")

    class NodeTamu:
        def __init__(self, nama, no_kamar, tipe_kamar):
            self.nama = nama
            self.no_kamar = no_kamar
            self.tipe_kamar = tipe_kamar
            self.next = None

    class ReservasiHotel:
        def __init__(self):
            self.head = None

        def check_in(self, nama, no_kamar, tipe_kamar):
            baru = NodeTamu(nama, no_kamar, tipe_kamar)

            if not self.head:
                self.head = baru
                return

            sekarang = self.head
            while sekarang.next:
                sekarang = sekarang.next
            sekarang.next = baru

        def check_out(self, no_kamar):
            sekarang = self.head
            sebelumnya = None

            if sekarang and sekarang.no_kamar == no_kamar:
                self.head = sekarang.next
                return f"{sekarang.nama} berhasil check-out"

            while sekarang and sekarang.no_kamar != no_kamar:
                sebelumnya = angiogenesis = sekarang
                sekarang = sekarang.next

            if not sekarang:
                return None

            sebelumnya.next = sekarang.next
            return f"{sekarang.nama} berhasil check-out"

        def get_tamu(self):
            data = []
            sekarang = self.head

            while sekarang:
                data.append({
                    "Nama": sekarang.nama,
                    "No Kamar": sekarang.no_kamar,
                    "Tipe Kamar": sekarang.tipe_kamar
                })
                sekarang = sekarang.next

            return data

    if "hotel" not in st.session_state:
        st.session_state.hotel = ReservasiHotel()

    hotel = st.session_state.hotel

    # Form Check-In
    st.subheader("🔑 Check In")
    nama = st.text_input("Nama Tamu", key="hotel_nama")
    no_kamar = st.number_input("Nomor Kamar", min_value=1, step=1)
    tipe_kamar = st.selectbox("Tipe Kamar", ["Standard", "Deluxe", "Suite"])

    if st.button("Tambah Reservasi"):
        if nama:
            hotel.check_in(nama, no_kamar, tipe_kamar)
            st.success(f"{nama} berhasil check-in di kamar {no_kamar}")
        else:
            st.warning("Masukkan nama tamu terlebih dahulu.")

    # Form Check-Out
    st.subheader("🚪 Check Out")
    hapus_kamar = st.number_input("Nomor Kamar yang Check Out", min_value=1, step=1, key="checkout")

    if st.button("Proses Check Out"):
        hasil = hotel.check_out(hapus_kamar)
        if hasil:
            st.success(hasil)
        else:
            st.error("Nomor kamar tidak ditemukan.")

    # Daftar Tamu Aktif
    st.subheader("📋 Daftar Tamu Aktif")
    daftar = hotel.get_tamu()

    if daftar:
        st.dataframe(daftar, use_container_width=True)
        st.info(f"Total tamu aktif: {len(daftar)}")
    else:
        st.warning("Hotel kosong. Belum ada reservasi aktif.")


# ==============================================================================
# 6. Modul 5: Antrian Rumah Sakit
# ==============================================================================
elif menu == "Antrian Rumah Sakit":
    st.title("🏥 Antrian Rumah Sakit")

    class NodePasien:
        def __init__(self, nama_pasien):
            self.nama_pasien = nama_pasien
            self.next = None

    class AntrianRumahSakit:
        def __init__(self):
            self.head = None

        def tambah_pasien(self, nama):
            baru = NodePasien(nama)

            if self.head is None:
                self.head = baru
            else:
                current = self.head
                while current.next:
                    current = current.next
                current.next = baru

        def panggil_pasien(self):
            if self.head is None:
                return None

            pasien = self.head.nama_pasien
            self.head = self.head.next
            return pasien

        def get_antrian(self):
            data = []
            current = self.head

            while current:
                data.append(current.nama_pasien)
                current = current.next

            return data

    if "rs" not in st.session_state:
        st.session_state.rs = AntrianRumahSakit()

    rs = st.session_state.rs

    nama = st.text_input("Nama Pasien", key="rs_nama")

    if st.button("Tambah Pasien"):
        if nama:
            rs.tambah_pasien(nama)
            st.success(f"{nama} masuk antrian")

    if st.button("Panggil Pasien"):
        pasien = rs.panggil_pasien()
        if pasien:
            st.success(f"Pasien {pasien} dipanggil")
        else:
            st.warning("Antrian kosong")

    st.subheader("Daftar Antrian")
    antrian = rs.get_antrian()

    if antrian:
        for i, psn in enumerate(antrian, 1):
            st.write(f"{i}. {psn}")
    else:
        st.info("Antrian kosong")


# ==============================================================================
# 7. Modul 6: Editor Teks Undo & Redo
# ==============================================================================
elif menu == "Editor Undo Redo":
    st.title("📝 Editor Teks Undo & Redo")

    class NodeEditor:
        def __init__(self, aksi):
            self.aksi = aksi
            self.prev = None
            self.next = None

    class EditorTeks:
        def __init__(self):
            self.current = None

        def tambah_aksi(self, aksi):
            baru = NodeEditor(aksi)

            if self.current is None:
                self.current = baru
            else:
                self.current.next = baru
                baru.prev = self.current
                self.current = baru

        def undo(self):
            if self.current and self.current.prev:
                self.current = self.current.prev
                return True
            return False

        def redo(self):
            if self.current and self.current.next:
                self.current = self.current.next
                return True
            return False

        def aksi_sekarang(self):
            if self.current:
                return self.current.aksi
            return "Belum ada aksi"

    if "editor" not in st.session_state:
        st.session_state.editor = EditorTeks()

    editor = st.session_state.editor

    aksi = st.text_input("Masukkan Aksi", key="editor_aksi")

    if st.button("Tambah Aksi"):
        if aksi:
            editor.tambah_aksi(aksi)
            st.success("Aksi ditambahkan")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Undo"):
            if editor.undo():
                st.success("Undo berhasil")
            else:
                st.warning("Tidak dapat undo")

    with col2:
        if st.button("Redo"):
            if editor.redo():
                st.success("Redo berhasil")
            else:
                st.warning("Tidak dapat redo")

    st.subheader("Aksi Saat Ini")
    st.write(editor.aksi_sekarang())