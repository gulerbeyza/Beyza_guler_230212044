"""
Nonlinear Diffusion - Perona-Malik Model Implementation
CMP 717 - Practical Assignment 1

Bu kod, Perona-Malik doğrusal olmayan difüzyon modelini ve çeşitli difüzivite
fonksiyonlarını implement eder.

Yazar: [Your Name]
Tarih: [Date]
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import os


class NonlinearDiffusion:
    """
    Nonlinear Diffusion filtreleme için ana sınıf.
    
    Perona-Malik modeli ve çeşitli difüzivite fonksiyonlarını içerir.
    """
    
    def __init__(self, lambda_param=10.0, sigma=1.0, dt=0.25, num_iterations=50):
        """
        Nonlinear Diffusion filtresini initialize eder.
        
        Parameters:
        -----------
        lambda_param : float
            Kontrast eşiği (contrast threshold)
        sigma : float
            Gaussian yumuşatma için ölçek parametresi
        dt : float
            Zaman adımı (time step)
        num_iterations : int
            Difüzyon iterasyon sayısı
        """
        self.lambda_param = lambda_param
        self.sigma = sigma
        self.dt = dt
        self.num_iterations = num_iterations
        self.diffusivity_type = 'pm1'  # Varsayılan: PM Type 1
        
    def set_diffusivity(self, diff_type):
        """
        Difüzivite fonksiyon tipini ayarlar.
        
        Parameters:
        -----------
        diff_type : str
            'pm1', 'pm2', veya 'charbonnier'
        """
        if diff_type not in ['pm1', 'pm2', 'charbonnier']:
            raise ValueError("Geçersiz difüzivite tipi. 'pm1', 'pm2', veya 'charbonnier' olmalı.")
        self.diffusivity_type = diff_type
    
    # ========================================================================
    # DİFÜZİVİTE FONKSİYONLARI
    # ========================================================================
    
    def diffusivity_pm1(self, gradient_magnitude):
        """
        Perona-Malik Difüzivite - Tip 1
        
        g(|x|) = exp(-|x|²/λ²)
        
        Bu fonksiyon, Gauss benzeri bir yapıya sahiptir. Küçük gradyanları
        güçlü şekilde difüze ederken, büyük gradyanları (kenarları) korur.
        
        Parameters:
        -----------
        gradient_magnitude : ndarray
            Gradyan büyüklüğü
            
        Returns:
        --------
        ndarray
            Difüzivite değerleri [0, 1] aralığında
        """
        return np.exp(-(gradient_magnitude ** 2) / (self.lambda_param ** 2))
    
    def diffusivity_pm2(self, gradient_magnitude):
        """
        Perona-Malik Difüzivite - Tip 2
        
        g(|x|) = 1 / (1 + |x|²/λ²)
        
        Bu fonksiyon, PM Tip 1'e göre daha geniş bir aralıkta difüzyon sağlar.
        Büyük gradyanlarda bile bir miktar difüzyon yapar.
        
        Parameters:
        -----------
        gradient_magnitude : ndarray
            Gradyan büyüklüğü
            
        Returns:
        --------
        ndarray
            Difüzivite değerleri [0, 1] aralığında
        """
        return 1.0 / (1.0 + (gradient_magnitude ** 2) / (self.lambda_param ** 2))
    
    def diffusivity_charbonnier(self, gradient_magnitude):
        """
        Charbonnier Difüzivite
        
        g(|x|) = 1 / √(1 + |x|²/λ²)
        
        Konveks bir fonksiyondur ve optimizasyon açısından daha kararlıdır.
        PM Tip 2'nin karekök versiyonu olarak düşünülebilir.
        
        Parameters:
        -----------
        gradient_magnitude : ndarray
            Gradyan büyüklüğü
            
        Returns:
        --------
        ndarray
            Difüzivite değerleri [0, 1] aralığında
        """
        return 1.0 / np.sqrt(1.0 + (gradient_magnitude ** 2) / (self.lambda_param ** 2))
    
    def compute_diffusivity(self, gradient_magnitude):
        """
        Seçili difüzivite fonksiyonunu hesaplar.
        
        Parameters:
        -----------
        gradient_magnitude : ndarray
            Gradyan büyüklüğü
            
        Returns:
        --------
        ndarray
            Difüzivite değerleri
        """
        if self.diffusivity_type == 'pm1':
            return self.diffusivity_pm1(gradient_magnitude)
        elif self.diffusivity_type == 'pm2':
            return self.diffusivity_pm2(gradient_magnitude)
        elif self.diffusivity_type == 'charbonnier':
            return self.diffusivity_charbonnier(gradient_magnitude)
    
    # ========================================================================
    # GRADYAN HESAPLAMA
    # ========================================================================
    
    def compute_gradients(self, image):
        """
        Görüntünün x ve y yönlerindeki gradyanlarını hesaplar.
        
        Merkezi farklar (central differences) yöntemi kullanılır:
        - ∂u/∂x ≈ (u[i,j+1] - u[i,j-1]) / 2
        - ∂u/∂y ≈ (u[i+1,j] - u[i-1,j]) / 2
        
        Parameters:
        -----------
        image : ndarray
            Girdi görüntüsü
            
        Returns:
        --------
        grad_x, grad_y : tuple of ndarrays
            X ve Y yönündeki gradyanlar
        """
        # Padding ekleyerek boundary sorununu çöz
        padded = np.pad(image, 1, mode='edge')
        
        # X yönünde gradyan (yatay)
        grad_x = (padded[1:-1, 2:] - padded[1:-1, :-2]) / 2.0
        
        # Y yönünde gradyan (dikey)
        grad_y = (padded[2:, 1:-1] - padded[:-2, 1:-1]) / 2.0
        
        return grad_x, grad_y
    
    def compute_gradient_magnitude(self, grad_x, grad_y):
        """
        Gradyan büyüklüğünü hesaplar.
        
        |∇u| = √(grad_x² + grad_y²)
        
        Parameters:
        -----------
        grad_x, grad_y : ndarrays
            X ve Y yönündeki gradyanlar
            
        Returns:
        --------
        ndarray
            Gradyan büyüklüğü
        """
        return np.sqrt(grad_x ** 2 + grad_y ** 2)
    
    # ========================================================================
    # DİFÜZYON İŞLEMİ
    # ========================================================================
    
    def diffusion_step(self, image):
        """
        Tek bir difüzyon iterasyon adımı gerçekleştirir.
        
        PDE: ∂u/∂t = ∇·(g(|∇u_σ|)∇u)
        
        Burada:
        - u: görüntü yoğunluğu
        - g: difüzivite fonksiyonu
        - u_σ: Gaussian ile yumuşatılmış görüntü
        
        Parameters:
        -----------
        image : ndarray
            Mevcut görüntü
            
        Returns:
        --------
        ndarray
            Güncellenmiş görüntü
        """
        # Gaussian yumuşatma uygula (gradyan hesabı için)
        if self.sigma > 0:
            smoothed = gaussian_filter(image, sigma=self.sigma)
        else:
            smoothed = image.copy()
        
        # Gradyanları hesapla
        grad_x, grad_y = self.compute_gradients(smoothed)
        gradient_mag = self.compute_gradient_magnitude(grad_x, grad_y)
        
        # Difüzivite hesapla
        g = self.compute_diffusivity(gradient_mag)
        
        # Difüzyon katsayılı gradyanları hesapla
        grad_x_orig, grad_y_orig = self.compute_gradients(image)
        
        # g * ∇u hesapla
        diffusion_x = g * grad_x_orig
        diffusion_y = g * grad_y_orig
        
        # Divergence hesapla: ∇·(g∇u)
        # ∂/∂x(g * ∂u/∂x) + ∂/∂y(g * ∂u/∂y)
        div_x, _ = self.compute_gradients(diffusion_x)
        _, div_y = self.compute_gradients(diffusion_y)
        divergence = div_x + div_y
        
        # Explicit time stepping: u_new = u_old + dt * ∇·(g∇u)
        updated_image = image + self.dt * divergence
        
        # Değerleri [0, 255] aralığında tut
        updated_image = np.clip(updated_image, 0, 255)
        
        return updated_image
    
    def apply(self, image):
        """
        Görüntüye nonlinear difüzyon filtresi uygular.
        
        Parameters:
        -----------
        image : ndarray
            Girdi görüntüsü (gri tonlamalı, 0-255 arası)
            
        Returns:
        --------
        result : ndarray
            Filtrelenmiş görüntü
        history : dict
            İterasyonlar boyunca toplanan istatistikler
        """
        # Görüntüyü float'a dönüştür
        result = image.astype(np.float64)
        
        # İstatistikleri saklamak için
        history = {
            'mean': [],
            'variance': [],
            'gradient_magnitude': []
        }
        
        # İterasyonlar
        for i in range(self.num_iterations):
            # Difüzyon adımı
            result = self.diffusion_step(result)
            
            # İstatistikleri kaydet
            history['mean'].append(np.mean(result))
            history['variance'].append(np.var(result))
            
            # Toplam gradyan büyüklüğü
            gx, gy = self.compute_gradients(result)
            grad_mag = self.compute_gradient_magnitude(gx, gy)
            history['gradient_magnitude'].append(np.sum(grad_mag))
            
            # İlerleme göster
            if (i + 1) % 10 == 0:
                print(f"  İterasyon {i+1}/{self.num_iterations} tamamlandı")
        
        # Uint8'e dönüştür
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result, history


class ColorNonlinearDiffusion(NonlinearDiffusion):
    """
    Renkli görüntüler için nonlinear diffusion.
    
    Her renk kanalına difüzyon uygular, ancak difüzivite hesabı için
    tüm kanalların gradyanları birlikte kullanılır.
    """
    
    def apply_color(self, color_image):
        """
        Renkli görüntüye nonlinear difüzyon uygular.
        
        PDE: ∂u^i/∂t = ∇·(g(Σ|∇u_σ^k|)∇u^i)
        
        Burada:
        - i: renk kanalı (R, G, B)
        - k: tüm kanallar üzerinde toplama
        
        Parameters:
        -----------
        color_image : ndarray
            RGB görüntü (H x W x 3)
            
        Returns:
        --------
        result : ndarray
            Filtrelenmiş renkli görüntü
        history : dict
            Her kanal için ayrı istatistikler
        """
        # Görüntüyü float'a dönüştür
        image_float = color_image.astype(np.float64)
        
        # Kanalları ayır
        channels = [image_float[:, :, i] for i in range(3)]
        result_channels = [ch.copy() for ch in channels]
        
        # Her kanal için istatistikler
        history = {
            'mean_r': [], 'mean_g': [], 'mean_b': [],
            'variance_r': [], 'variance_g': [], 'variance_b': [],
            'gradient_magnitude': []
        }
        
        print(f"Renkli görüntü difüzyonu başlıyor...")
        print(f"  Difüzivite: {self.diffusivity_type}")
        print(f"  Lambda: {self.lambda_param}, Sigma: {self.sigma}")
        print(f"  İterasyon sayısı: {self.num_iterations}")
        
        # İterasyonlar
        for iteration in range(self.num_iterations):
            # Her kanal için Gaussian yumuşatma
            if self.sigma > 0:
                smoothed_channels = [gaussian_filter(ch, sigma=self.sigma) 
                                   for ch in result_channels]
            else:
                smoothed_channels = [ch.copy() for ch in result_channels]
            
            # Tüm kanalların gradyanlarını hesapla
            gradients = [self.compute_gradients(ch) for ch in smoothed_channels]
            
            # Gradyan büyüklüklerini topla (tüm kanallar için)
            total_gradient_mag = np.zeros_like(result_channels[0])
            for grad_x, grad_y in gradients:
                grad_mag = self.compute_gradient_magnitude(grad_x, grad_y)
                total_gradient_mag += grad_mag
            
            # Difüziviteyi hesapla (tüm kanalların gradyanları kullanılarak)
            g = self.compute_diffusivity(total_gradient_mag)
            
            # Her kanalı güncelle (aynı difüzivite ile)
            new_channels = []
            for i, channel in enumerate(result_channels):
                # Kanalın kendi gradyanları
                grad_x, grad_y = self.compute_gradients(channel)
                
                # g * ∇u hesapla
                diffusion_x = g * grad_x
                diffusion_y = g * grad_y
                
                # Divergence hesapla
                div_x, _ = self.compute_gradients(diffusion_x)
                _, div_y = self.compute_gradients(diffusion_y)
                divergence = div_x + div_y
                
                # Güncelle
                updated = channel + self.dt * divergence
                updated = np.clip(updated, 0, 255)
                new_channels.append(updated)
            
            result_channels = new_channels
            
            # İstatistikleri kaydet
            history['mean_r'].append(np.mean(result_channels[0]))
            history['mean_g'].append(np.mean(result_channels[1]))
            history['mean_b'].append(np.mean(result_channels[2]))
            
            history['variance_r'].append(np.var(result_channels[0]))
            history['variance_g'].append(np.var(result_channels[1]))
            history['variance_b'].append(np.var(result_channels[2]))
            
            history['gradient_magnitude'].append(np.sum(total_gradient_mag))
            
            # İlerleme göster
            if (iteration + 1) % 10 == 0:
                print(f"  İterasyon {iteration+1}/{self.num_iterations} tamamlandı")
        
        # Kanalları birleştir
        result = np.stack(result_channels, axis=2)
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result, history


# ============================================================================
# YARDIMCI FONKSİYONLAR VE GÖRSELLEŞTIRME
# ============================================================================

def linear_diffusion(image, num_iterations=50, dt=0.25):
    """
    Linear (Gaussian) diffusion - karşılaştırma için.
    
    ∂u/∂t = Δu (Laplacian)
    
    Parameters:
    -----------
    image : ndarray
        Girdi görüntüsü
    num_iterations : int
        İterasyon sayısı
    dt : float
        Zaman adımı
        
    Returns:
    --------
    ndarray
        Filtrelenmiş görüntü
    """
    result = image.astype(np.float64)
    
    for i in range(num_iterations):
        # Laplacian hesapla (ikinci türevlerin toplamı)
        laplacian = cv2.Laplacian(result, cv2.CV_64F)
        result = result + dt * laplacian
        result = np.clip(result, 0, 255)
    
    return result.astype(np.uint8)


def plot_comparison(original, linear, pm1, pm2, charbonnier, save_path=None):
    """
    Farklı difüzyon modellerinin sonuçlarını karşılaştırır.
    
    Parameters:
    -----------
    original : ndarray
        Orijinal görüntü
    linear : ndarray
        Linear difüzyon sonucu
    pm1, pm2, charbonnier : ndarrays
        Nonlinear difüzyon sonuçları
    save_path : str, optional
        Kaydedilecek dosya yolu
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    images = [original, linear, pm1, pm2, charbonnier]
    titles = ['Orijinal', 'Linear Diffusion', 'PM Type 1', 
              'PM Type 2', 'Charbonnier']
    
    for i, (img, title) in enumerate(zip(images, titles)):
        row = i // 3
        col = i % 3
        axes[row, col].imshow(img, cmap='gray')
        axes[row, col].set_title(title, fontsize=12, fontweight='bold')
        axes[row, col].axis('off')
    
    # Boş subplot'u gizle
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Karşılaştırma grafiği kaydedildi: {save_path}")
    
    plt.show()


def plot_statistics(history, title='Statistics Over Iterations', save_path=None):
    """
    İterasyonlar boyunca istatistikleri görselleştirir.
    
    Parameters:
    -----------
    history : dict
        İstatistik verilerini içeren dictionary
    title : str
        Grafik başlığı
    save_path : str, optional
        Kaydedilecek dosya yolu
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    iterations = range(1, len(history['mean']) + 1)
    
    # Ortalama yoğunluk
    axes[0].plot(iterations, history['mean'], 'b-', linewidth=2)
    axes[0].set_xlabel('İterasyon')
    axes[0].set_ylabel('Ortalama Yoğunluk')
    axes[0].set_title('Ortalama Yoğunluk Değişimi')
    axes[0].grid(True, alpha=0.3)
    
    # Varyans
    axes[1].plot(iterations, history['variance'], 'r-', linewidth=2)
    axes[1].set_xlabel('İterasyon')
    axes[1].set_ylabel('Varyans')
    axes[1].set_title('Yoğunluk Varyansı Değişimi')
    axes[1].grid(True, alpha=0.3)
    
    # Gradyan büyüklüğü
    axes[2].plot(iterations, history['gradient_magnitude'], 'g-', linewidth=2)
    axes[2].set_xlabel('İterasyon')
    axes[2].set_ylabel('Toplam Gradyan Büyüklüğü')
    axes[2].set_title('Toplam Gradyan Büyüklüğü Değişimi')
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"İstatistik grafiği kaydedildi: {save_path}")
    
    plt.show()


def compare_parameters(image, diff_type='pm1', lambdas=[5, 10, 20], 
                      sigmas=[0.5, 1.0, 2.0], save_dir='results'):
    """
    Farklı parametrelerle sonuçları karşılaştırır.
    
    Parameters:
    -----------
    image : ndarray
        Girdi görüntüsü
    diff_type : str
        Difüzivite tipi
    lambdas : list
        Test edilecek lambda değerleri
    sigmas : list
        Test edilecek sigma değerleri
    save_dir : str
        Sonuçların kaydedileceği dizin
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Lambda karşılaştırması
    print(f"\n{'='*60}")
    print(f"Lambda parametresi karşılaştırması ({diff_type})")
    print(f"{'='*60}")
    
    fig, axes = plt.subplots(1, len(lambdas) + 1, figsize=(15, 4))
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Orijinal')
    axes[0].axis('off')
    
    for i, lambda_val in enumerate(lambdas, 1):
        diffusion = NonlinearDiffusion(lambda_param=lambda_val, sigma=1.0)
        diffusion.set_diffusivity(diff_type)
        result, _ = diffusion.apply(image)
        
        axes[i].imshow(result, cmap='gray')
        axes[i].set_title(f'λ = {lambda_val}')
        axes[i].axis('off')
    
    plt.suptitle(f'{diff_type.upper()} - Lambda Karşılaştırması', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/lambda_comparison_{diff_type}.png', dpi=300)
    plt.close()
    
    # Sigma karşılaştırması
    print(f"\n{'='*60}")
    print(f"Sigma parametresi karşılaştırması ({diff_type})")
    print(f"{'='*60}")
    
    fig, axes = plt.subplots(1, len(sigmas) + 1, figsize=(15, 4))
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Orijinal')
    axes[0].axis('off')
    
    for i, sigma_val in enumerate(sigmas, 1):
        diffusion = NonlinearDiffusion(lambda_param=10.0, sigma=sigma_val)
        diffusion.set_diffusivity(diff_type)
        result, _ = diffusion.apply(image)
        
        axes[i].imshow(result, cmap='gray')
        axes[i].set_title(f'σ = {sigma_val}')
        axes[i].axis('off')
    
    plt.suptitle(f'{diff_type.upper()} - Sigma Karşılaştırması', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/sigma_comparison_{diff_type}.png', dpi=300)
    plt.close()
    
    print(f"\nKarşılaştırma grafikleri '{save_dir}' dizinine kaydedildi.")


# ============================================================================
# DEMO VE TEST FONKSİYONLARI
# ============================================================================

def demo_grayscale():
    """
    Gri tonlamalı görüntü için demo.
    """
    print("\n" + "="*70)
    print("GRİ TONLAMALI GÖRÜNTÜ İÇİN NONLINEAR DİFÜZYON DEMO")
    print("="*70)
    
    # Test görüntüsü oluştur (veya yükle)
    # Burada basit bir test görüntüsü oluşturuyoruz
    img = cv2.imread('test_image.png', cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print("Test görüntüsü bulunamadı. Sentetik görüntü oluşturuluyor...")
        img = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
        # Kenarlar ekle
        img[50:80, :] = 200
        img[:, 100:130] = 50
        # Gürültü ekle
        noise = np.random.normal(0, 25, img.shape)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
    
    print(f"Görüntü boyutu: {img.shape}")
    
    # Linear difüzyon
    print("\nLinear difüzyon uygulanıyor...")
    linear_result = linear_diffusion(img, num_iterations=50)
    
    # Nonlinear difüzyon modelleri
    results = {}
    
    for diff_type, name in [('pm1', 'PM Type 1'), 
                            ('pm2', 'PM Type 2'), 
                            ('charbonnier', 'Charbonnier')]:
        print(f"\n{name} difüzyonu uygulanıyor...")
        diffusion = NonlinearDiffusion(lambda_param=10.0, sigma=1.0, 
                                       dt=0.25, num_iterations=50)
        diffusion.set_diffusivity(diff_type)
        result, history = diffusion.apply(img)
        results[diff_type] = (result, history)
    
    # Sonuçları görselleştir
    print("\nSonuçlar görselleştiriliyor...")
    plot_comparison(img, linear_result, 
                   results['pm1'][0], results['pm2'][0], results['charbonnier'][0],
                   save_path='comparison_grayscale.png')
    
    # İstatistikleri göster
    for diff_type, name in [('pm1', 'PM Type 1'), 
                            ('pm2', 'PM Type 2'), 
                            ('charbonnier', 'Charbonnier')]:
        plot_statistics(results[diff_type][1], 
                       title=f'{name} - İstatistikler',
                       save_path=f'statistics_{diff_type}.png')


def demo_color():
    """
    Renkli görüntü için demo.
    """
    print("\n" + "="*70)
    print("RENKLİ GÖRÜNTÜ İÇİN NONLINEAR DİFÜZYON DEMO")
    print("="*70)
    
    # Renkli test görüntüsü yükle
    img = cv2.imread('test_image_color.png')
    
    if img is None:
        print("Renkli test görüntüsü bulunamadı. Sentetik görüntü oluşturuluyor...")
        img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        # Renkli desenler ekle
        img[50:80, :, 0] = 200  # Kırmızı çizgi
        img[:, 100:130, 1] = 200  # Yeşil çizgi
        img[150:180, 150:180, 2] = 200  # Mavi kare
        # Gürültü ekle
        noise = np.random.normal(0, 20, img.shape)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    print(f"Görüntü boyutu: {img.shape}")
    
    # Renkli nonlinear difüzyon
    print("\nRenkli PM Type 1 difüzyonu uygulanıyor...")
    color_diffusion = ColorNonlinearDiffusion(lambda_param=15.0, sigma=1.0, 
                                              dt=0.25, num_iterations=30)
    color_diffusion.set_diffusivity('pm1')
    result, history = color_diffusion.apply_color(img)
    
    # Sonuçları göster
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img)
    axes[0].set_title('Orijinal')
    axes[0].axis('off')
    
    axes[1].imshow(result)
    axes[1].set_title('PM Type 1 Difüzyon Sonucu')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('color_diffusion_result.png', dpi=300)
    plt.show()
    
    # İstatistikleri göster
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    iterations = range(1, len(history['mean_r']) + 1)
    
    # Her kanal için ortalama
    axes[0].plot(iterations, history['mean_r'], 'r-', label='Red', linewidth=2)
    axes[0].plot(iterations, history['mean_g'], 'g-', label='Green', linewidth=2)
    axes[0].plot(iterations, history['mean_b'], 'b-', label='Blue', linewidth=2)
    axes[0].set_xlabel('İterasyon')
    axes[0].set_ylabel('Ortalama Yoğunluk')
    axes[0].set_title('Kanal Ortalamaları')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Her kanal için varyans
    axes[1].plot(iterations, history['variance_r'], 'r-', label='Red', linewidth=2)
    axes[1].plot(iterations, history['variance_g'], 'g-', label='Green', linewidth=2)
    axes[1].plot(iterations, history['variance_b'], 'b-', label='Blue', linewidth=2)
    axes[1].set_xlabel('İterasyon')
    axes[1].set_ylabel('Varyans')
    axes[1].set_title('Kanal Varyansları')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Toplam gradyan
    axes[2].plot(iterations, history['gradient_magnitude'], 'k-', linewidth=2)
    axes[2].set_xlabel('İterasyon')
    axes[2].set_ylabel('Toplam Gradyan')
    axes[2].set_title('Toplam Gradyan Büyüklüğü')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('color_diffusion_statistics.png', dpi=300)
    plt.show()


# ============================================================================
# ANA PROGRAM
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("NONLINEAR DIFFUSION - PERONA-MALIK MODEL")
    print("CMP 717 - Practical Assignment 1")
    print("="*70)
    
    # Demo seçenekleri
    print("\nDemo Seçenekleri:")
    print("1. Gri tonlamalı görüntü demo")
    print("2. Renkli görüntü demo")
    print("3. Parametre karşılaştırması")
    print("4. Hepsi")
    
    choice = input("\nSeçiminiz (1-4): ")
    
    if choice == '1':
        demo_grayscale()
    elif choice == '2':
        demo_color()
    elif choice == '3':
        # Basit test görüntüsü oluştur
        img = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        img[30:50, :] = 200
        noise = np.random.normal(0, 20, img.shape)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
        
        compare_parameters(img, diff_type='pm1')
        compare_parameters(img, diff_type='pm2')
    elif choice == '4':
        demo_grayscale()
        demo_color()
        
        img = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        img[30:50, :] = 200
        noise = np.random.normal(0, 20, img.shape)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
        compare_parameters(img, diff_type='pm1')
    else:
        print("Geçersiz seçim!")
    
    print("\n" + "="*70)
    print("PROGRAM TAMAMLANDI")
    print("="*70)
