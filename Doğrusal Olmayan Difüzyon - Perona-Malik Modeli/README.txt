# Nonlinear Diffusion - Perona-Malik Model
## CMP 717 - Practical Assignment 1

### 📁 Dosya İçeriği

Bu klasörde aşağıdaki dosyalar bulunmaktadır:

1. **Nonlinear_Diffusion_Odevi_15_Gunluk_Plan.docx**
   - Türkçe'ye çevrilmiş ödev dökümanı
   - 15 günlük detaylı çalışma planı
   - Tüm problemlerin açıklamaları
   - Değerlendirme kriterleri ve teslim gereksinimleri

2. **nonlinear_diffusion_solution.py**
   - Ödevin tam Python çözümü
   - Perona-Malik modeli implementasyonu
   - Üç difüzivite fonksiyonu (PM Type 1, PM Type 2, Charbonnier)
   - Gri tonlamalı ve renkli görüntü desteği
   - Analiz ve görselleştirme fonksiyonları

---

## 🎯 Ödev Özeti

Bu ödev, doğrusal olmayan difüzyon filtreleme tekniklerini öğrenmeniz ve 
uygulamanız için tasarlanmıştır. Perona-Malik modeli kullanılarak görüntülerdeki 
kenarlar korunurken gürültü giderilmesi amaçlanmaktadır.

### Temel Konular:
- Nonlinear PDE'ler
- Perona-Malik difüzyon modeli
- Difüzivite fonksiyonları
- Görüntü yumuşatma ve kenar koruma
- Gradyan hesaplama ve divergence

---

## 📅 15 Günlük Çalışma Planı

### Hafta 1: Teori ve Hazırlık (Gün 1-5)
- Gün 1-2: Literatür araştırması
- Gün 3: Matematik temelleri
- Gün 4: Geliştirme ortamı kurulumu
- Gün 5: Linear difüzyon kodunu inceleme

### Hafta 2: İmplementasyon (Gün 6-10)
- Gün 6-7: Problem 1.1 - Perona-Malik modeli
- Gün 8: Test ve hata ayıklama
- Gün 9-10: Problem 1.2 - Karşılaştırmalı analiz

### Hafta 3: Renkli Görüntüler ve Raporlama (Gün 11-15)
- Gün 11-12: Problem 1.3 - Renkli görüntü desteği
- Gün 13: Rapor hazırlama
- Gün 14: Kod dokümantasyonu
- Gün 15: Final kontroller ve teslim

---

## 🔧 Kurulum ve Kullanım

### Gerekli Kütüphaneler:
```bash
pip install numpy opencv-python matplotlib scipy
```

### Kullanım:
```python
python nonlinear_diffusion_solution.py
```

Program çalıştırıldığında size demo seçenekleri sunulacaktır:
1. Gri tonlamalı görüntü demo
2. Renkli görüntü demo
3. Parametre karşılaştırması
4. Hepsi

---

## 📊 Problemler

### Problem 1.1: Perona-Malik Modeli
Üç farklı difüzivite fonksiyonu implement edilmiştir:

**A. PM Type 1:**
```
g(|x|) = exp(-|x|²/λ²)
```

**B. PM Type 2:**
```
g(|x|) = 1 / (1 + |x|²/λ²)
```

**C. Charbonnier:**
```
g(|x|) = 1 / √(1 + |x|²/λ²)
```

### Problem 1.2: Karşılaştırmalı Analiz
- Linear vs. Nonlinear difüzyon karşılaştırması
- Parametre etkilerinin analizi (λ, σ, T)
- İstatistiksel değişimler (ortalama, varyans, gradyan)

### Problem 1.3: Renkli Görüntü Desteği
- RGB kanalları için difüzyon
- Kanallar arası tutarlılık
- Renk korumalı yumuşatma

---

## 📝 Kod Yapısı

### Ana Sınıflar:

**NonlinearDiffusion**
- Gri tonlamalı görüntüler için
- Üç difüzivite fonksiyonu
- Gradyan ve divergence hesaplama
- İteratif difüzyon süreci

**ColorNonlinearDiffusion**
- Renkli görüntüler için genişletilmiş sınıf
- Kanal bazlı işleme
- Ortak difüzivite hesaplama

### Yardımcı Fonksiyonlar:
- `linear_diffusion()` - Karşılaştırma için linear difüzyon
- `plot_comparison()` - Sonuçların görselleştirilmesi
- `plot_statistics()` - İstatistiklerin grafikleştirilmesi
- `compare_parameters()` - Parametre analizi

---

## 📈 Beklenen Sonuçlar

Kod başarıyla çalıştırıldığında aşağıdaki çıktılar elde edilir:

1. **Görsel Karşılaştırmalar:**
   - Orijinal vs. filtrelenmiş görüntüler
   - Farklı difüzivite fonksiyonlarının sonuçları
   - Parametre varyasyonlarının etkileri

2. **İstatistiksel Grafikler:**
   - Ortalama yoğunluk değişimi
   - Varyans değişimi
   - Gradyan büyüklüğü değişimi

3. **Kaydedilen Dosyalar:**
   - comparison_grayscale.png
   - statistics_pm1.png
   - statistics_pm2.png
   - statistics_charbonnier.png
   - color_diffusion_result.png
   - lambda_comparison_*.png
   - sigma_comparison_*.png

---

## 🎓 Değerlendirme Kriterleri

**5 Puan:** Mükemmel çözüm
- Tüm problemler doğru çözülmüş
- Kod temiz ve iyi dokümante edilmiş
- Kapsamlı analiz ve yorumlar
- Ekstra özellikler eklenmiş

**4 Puan:** Tam çözüm
- Tüm gereksinimler karşılanmış
- Kod çalışıyor ve sonuçlar doğru
- İyi dokümantasyon

**3 Puan:** Çoğunlukla doğru
- Ana özellikler çalışıyor
- Bazı küçük hatalar var
- Temel dokümantasyon mevcut

**2 Puan:** Kısmen doğru
- Bazı bölümler eksik veya hatalı
- Kod kısmen çalışıyor

**1 Puan:** Deneme
- Çözüm denemesi var ancak çalışmıyor

**0 Puan:** Teslim edilmemiş

---

## 📚 Kaynaklar

1. **P. Perona and J. Malik** (1990)
   "Scale space and edge detection using anisotropic diffusion"
   IEEE Transactions on Pattern Analysis and Machine Intelligence, 12:629-639

2. **P. Charbonnier et al.** (1994)
   "Two deterministic half-quadratic regularization algorithms for computed imaging"
   Proc. 1994 IEEE International Conference on Image Processing

3. **J. Weickert** (1998)
   "Anisotropic Diffusion in Image Processing"

---

## ⚠️ Önemli Notlar

### Akademik Dürüstlük:
- Tüm çalışmalar bireysel olarak yapılmalıdır
- Kod paylaşımı yasaktır
- İntihal ciddi şekilde cezalandırılır

### Teslim:
- Son tarih: 15 gün içinde
- Format: ZIP dosyası (ad-soyad-pa1.zip)
- İçerik: kod/, html/, README.txt
- E-posta: erkut@cs.hacettepe.edu.tr

### Geç Teslim:
- Toplamda 5 uzatma günü kullanabilirsiniz
- Onaysız geç teslimler 0.5 katsayı ile puanlanır

---

## 💡 İpuçları

1. **Başlangıç için:**
   - Önce küçük test görüntüleri kullanın
   - Her adımı görselleştirerek kontrol edin
   - Linear difüzyon kodunu iyi anlayın

2. **Debug için:**
   - Gradyan değerlerini kontrol edin
   - Boundary koşullarını doğrulayın
   - İlk iterasyonları manuel kontrol edin

3. **Performans için:**
   - NumPy vektörizasyonu kullanın
   - For döngülerinden kaçının
   - Büyük görüntüleri downscale edin

4. **Rapor için:**
   - Her grafik ve şekli açıklayın
   - Gözlemlerinizi net ifade edin
   - Beklenmedik sonuçları da tartışın

---

## 📧 İletişim

Sorularınız için:
- Office hours kullanın
- Sınıf arkadaşlarınızla genel kavramları tartışın
- E-posta: erkut@cs.hacettepe.edu.tr

---

## ✅ Checklist

Teslimden önce kontrol edin:

- [ ] Tüm üç difüzivite fonksiyonu çalışıyor
- [ ] Gri tonlamalı görüntü desteği var
- [ ] Renkli görüntü desteği var
- [ ] Karşılaştırmalı analizler yapılmış
- [ ] İstatistiksel grafikler oluşturulmuş
- [ ] Kod iyi dokümante edilmiş
- [ ] HTML raporu hazırlanmış
- [ ] README.txt dosyası eklenmiş
- [ ] ZIP dosyası doğru şekilde oluşturulmuş
- [ ] Tüm dosyalar test edilmiş

---

**Başarılar dileriz!** 🎓✨
