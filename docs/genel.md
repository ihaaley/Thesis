# Longboard Trick Tanımlama — Genel Değerlendirme

## 1. Ne Yaptık ve Amaç Neydi?

Elimizde her biri yaklaşık 12 saniyelik ~1500+ etiketli video var. Her video bir klasörde, klasör adı o videonun trick adına karşılık geliyor. Örneğin `Caveman/` klasöründeki tüm videolar Caveman tricki içeriyor.

**Veri hazırlama adımları:**

1. Her video kare kare açıldı
2. Her karede **YOLOv8** ile kişinin 17 eklem noktası (omuz, diz, ayak bileği vb.) tespit edildi
3. Bu koordinatlar normalize edildi — kişinin kameraya uzaklığı ve boyu fark etmesin diye
4. Her 30 ardışık kare birleştirilerek bir "örnek" (sliding window) oluşturuldu
5. Tüm örnekler `X.npy` dosyasına, etiketler `y.npy`'ye kaydedildi
6. BiLSTM modeli bu verilerle eğitildi

**Amaç:** Modelin "bu 30 karelik hareket dizisi hangi trick?" sorusunu cevaplamasını öğrenmesi.

> **YOLOv8 Nedir?**
> YOLOv8, Ultralytics tarafından geliştirilen gerçek zamanlı nesne tespiti ve poz tahmini modelidir. "You Only Look Once" (yalnızca bir kez bak) prensibine dayanır — görüntüyü tek bir geçişte analiz ederek hem kişiyi hem de vücuttaki eklem noktalarını milisaniyeler içinde tespit eder. Bu projede `yolov8s-pose` modeli kullanılmıştır: her karede kişi başına COCO iskelet standardına göre 17 eklem noktası (burun, omuzlar, dirsekler, bilekler, kalçalar, dizler, ayak bilekleri) koordinat ve güven skoru olarak çıkarılmaktadır.

---

## 2. Neden BiLSTM Seçildi?

Trick tanımlama özünde bir **zaman serisi problemidir**. Bir trick tek bir kareyle tanımlanamaz — örneğin Caveman'de kişi önce tahtayı eline alır, sonra havaya atlar, sonra iner. Bu sıra önemlidir.

**BiLSTM (Bidirectional Long Short-Term Memory)** hem ileriye hem geriye doğru zamansal bağlantıları öğrenebildiği için bu tür ardışık hareketleri modellemek için uygundur.

**Diğer yaklaşımlar neden uygun değil?**

| Model               | Neden elendi                                                                                                                                                     |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Tek kare CNN        | Sadece o anki pozu görür, hareketin akışını göremez. "Havada duran kişi" her trick'te aynı görünür.                                                              |
| Random Forest / SVM | 30×51 boyutlu zaman serisini düzleştirip beslediğinde zamansal sıra bilgisi tamamen kaybolur.                                                                    |
| Transformer         | Teorik olarak daha güçlü ama ~20.000 örneklik veri setinde aşırı parametre sayısı nedeniyle ezber yapma (overfitting) riski yüksektir ve eğitim süresi çok uzar. |
| Tek yönlü LSTM      | Sadece geçmişe bakar. Oysa bazı trick'lerde önceki hareket, sonraki hareketi anlamlandırır — BiLSTM her iki yönü de kullandığı için daha yüksek doğruluk sağlar. |

Az veri, kısa sekans ve yorumlanabilirlik gereksinimi göz önünde bulundurulduğunda BiLSTM bu problem için pratik optimum seçimdir.

---

## 3. Mevcut Modelin Zayıf Noktası

12 saniyelik videonun içinde gerçek trick hareketi sadece **1-2 saniye** sürüyor. Geri kalan 10 saniye şunları içeriyor:

- Kişi longboarda yürüyerek biniyor
- Pozisyon alıyor
- Trick bittikten sonra duraksıyor
- Kamera hâlâ çekiyor

**Sayısal olarak:**

|                                                                  | Değer               |
| ---------------------------------------------------------------- | ------------------- |
| 12 sn × 15fps (her 2. kare)                                      | ~180 kare           |
| 30 karelik pencere, 15 stride → pencere sayısı                   | ~10 pencere / video |
| Gerçek trick içeren pencere                                      | **1-2 pencere**     |
| "Düz yürüyüş / bekleme" içeren ama trick etiketi taşıyan pencere | **8-9 pencere**     |

Modele gösterilen örneklerin **%80'i aslında o trick değil** — ama etiket öyle yazıyor. Model "Caveman nedir?" yerine "Caveman videosunda kişi nasıl yürür?" öğreniyor.

Bu kritik bir sorundur çünkü tüm trick'lerdeki "düz yürüyüş" birbirine çok benziyor. Model sınıflar arasındaki gerçek hareketi öğrenmek yerine bu gürültüyü ezberliyor.

**Mevcut test sonuçları:**

| Sınıf                    | Precision | Neden?                                                 |
| ------------------------ | --------- | ------------------------------------------------------ |
| Caveman                  | %92       | 436 video → en çok veri, model ezber yapabiliyor       |
| Ghostride                | %31       | Az veri + video içinde "sürüş" görüntüsü fazla         |
| NollieFrontside180Shuvit | %21       | 20 video × 8 gürültülü pencere = model hiç öğrenemiyor |

**Genel test doğruluğu: %57**

---

## 4. Videolar Kırpılırsa Ne Değişir?

Her videonun sadece gerçek trick'in olduğu kısmı (örneğin 4-6. saniyeler) işlenirse:

- Her video için ~10 pencere yerine **2-3 temiz pencere** kalır
- Bu pencerelerin **tamamı gerçek trick hareketi** içerir
- Model artık "Caveman'da kişi şu şekilde havaya kalkar" öğrenir, "Caveman videosunda kişi düz yürür" değil
- Sınıflar arasındaki "düz yürüyüş" gürültüsü büyük ölçüde ortadan kalkar

**Beklenen sonuç (tahmini):**

|                                        | Mevcut | Kırpma Sonrası |
| -------------------------------------- | ------ | -------------- |
| Test doğruluğu                         | %57    | %68–78         |
| Macro F1                               | %49    | %60–70         |
| NollieFrontside180 gibi zayıf sınıflar | %21–23 | %40–55         |

Kesin rakam kırpma kalitesine bağlıdır — doğru kırpılırsa üst değerlere yaklaşılır.

---

## 5. Sistem 1 Dakikalık Videoda Trick'i Bulabilir mi?

Evet. `main.py` ve `viewer.py` **sliding window** mantığıyla çalışır:

- Her kare işlenir, son 30 kare sürekli bir tamponda tutulur
- Tampon dolduğunda model çalışır → "Bu 30 kare hangi trick?" sorusunu cevaplar
- 1 dakikalık videoda model yaklaşık **900 kez** tahmin yapar

Ancak mevcut modelle şu sorun ortaya çıkar: videonun 58 saniyesi "düz sürüş" olduğundan model bu anlarda da bir şeyler "tahmin etmeye" çalışır ve yanlış pozitif üretir.

**Bu iki mekanizma ile kısmen engellenir:**

- `TRICK_FLASH_CONF_MIN = 0.50` — model %50'nin altında güvende ise ekranda göstermez
- `PRED_VOTE_BUF_LEN = 60` — 60 ardışık karenin çoğunluğu aynı trick'i söylüyorsa göster

**Kırpılmış veriyle eğitilmiş model** bu problemi kökten azaltır: model "trick nasıl görünür" öğrendiği için, trick olmayan anlarda güveni düşük kalır ve eşiği geçemez.

| Durum                   | Mevcut Model       | Kırpılmış Veriyle Eğitilmiş Model |
| ----------------------- | ------------------ | --------------------------------- |
| Düz sürüş anında tahmin | Sık yanlış pozitif | Düşük güven → sessiz kalır        |
| Trick anında tahmin     | %57 doğruluk       | %68–78 doğruluk (tahmini)         |
| Yanlış trick söyleme    | Sık                | Nadir                             |

---

## Özet

Mevcut yaklaşımda eğitim verisi, videonun tamamından elde edilen sliding window örneklerinden oluşmaktadır. Ancak etiketlenen hareket (trick), videonun yalnızca küçük bir bölümünde gerçekleşmektedir. Bu durum, eğitim örneklerinin büyük çoğunluğunun gürültüden ibaret olmasına ve modelin sınıflar arası gerçek farkları öğrenmekte zorlanmasına yol açmaktadır.

Trick segmentlerinin doğru şekilde belirlenerek yalnızca bu segmentler üzerinde eğitim yapılması, hem sınıflandırma doğruluğunu hem de gerçek zamanlı kullanımda yanlış pozitif oranını anlamlı ölçüde iyileştirmesi beklenmektedir.
