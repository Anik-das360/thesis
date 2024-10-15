#Gaussian
import cv2
import numpy as np
import pywt
import bm3d
from skimage.metrics import peak_signal_noise_ratio, mean_squared_error
from google.colab.patches import cv2_imshow
from google.colab import files

# Function to add Gaussian noise to an image
def add_gaussian_noise(image, sigma):
    noise = np.random.normal(0, sigma, image.shape)
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

def directional_nlm_filter(img, sigma, patch_size, search_window):
    # Convert the image to float32 for the wavelet transform
    img_float32 = np.float32(img)

    # Perform directional wavelet transform
    coeffs = pywt.dwt2(img_float32, 'bior1.3')
    cA, (cH, cV, cD) = coeffs

    # Apply NLM filter on each directional subband
    for subband in [cH, cV, cD]:
        # Convert subband to 8-bit unsigned format
        subband = cv2.convertScaleAbs(subband)
        subband_filtered = cv2.fastNlMeansDenoising(subband, None, sigma, patch_size, search_window)
        subband[:] = subband_filtered

    # Perform inverse wavelet transform
    denoised_img = pywt.idwt2((cA, (cH, cV, cD)), 'bior1.3')

    return denoised_img

def denoise_bm3d(img, sigma):
    # BM3D denoising on the whole image
    denoised_img = bm3d.bm3d(img, sigma_psd=sigma)

    return denoised_img

def adaptive_thresholding(coefficients, threshold_factor):
    # Apply adaptive thresholding on wavelet coefficients
    threshold = threshold_factor * np.median(np.abs(coefficients))
    coefficients = np.where(np.abs(coefficients) < threshold, 0, coefficients)

    return coefficients

def denoise_pipeline(noisy_img, sigma_nlm, patch_size_nlm, search_window_nlm, sigma_bm3d, threshold_factor):
    # Step 1: Directional NLM Filtering
    nlm_denoised_img = directional_nlm_filter(noisy_img, sigma_nlm, patch_size_nlm, search_window_nlm)

    # Step 2: BM3D Denoising
    bm3d_denoised_img = denoise_bm3d(nlm_denoised_img, sigma_bm3d)

    # Step 3: Adaptive Thresholding
    coeffs = pywt.wavedec2(bm3d_denoised_img, 'bior1.3', level=3)

    # Apply adaptive thresholding on each level
    for i in range(1, len(coeffs)):
        coeffs[i] = tuple(adaptive_thresholding(c, threshold_factor) for c in coeffs[i])

    # Step 4: Inverse Wavelet Transform
    final_denoised_img = pywt.waverec2(coeffs, 'bior1.3')

    return final_denoised_img

# Function to calculate Signal-to-Noise Ratio (SNR)
def calculate_snr(original_img, noisy_img):
    signal = np.sum(np.square(original_img))
    noise = np.sum(np.square(original_img - noisy_img))
    snr_value = 10 * np.log10(signal / noise)
    return snr_value

# Function to upload files in Colab
def upload_file():
    uploaded = files.upload()
    file_path = list(uploaded.keys())[0] if uploaded else None
    return file_path

# Step 0: Upload the image
original_image_path = upload_file()

# Check if the image is uploaded successfully
if original_image_path is None:
    print("Error: Please upload an image.")
else:
    # Load the uploaded image
    original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)

    # Check if the image is loaded successfully
    if original_image is None:
        print(f"Error: Unable to load the image from the path: {original_image_path}")
    else:
        # Set noise parameters
        noise_sigma = 20  # Adjust sigma value as standard deviation needed based on the noise level

        # Add Gaussian noise to the original image
        noisy_image = add_gaussian_noise(original_image, noise_sigma)

        # Set denoising parameters
        sigma_nlm = 10
        patch_size_nlm = 5
        search_window_nlm = 15
        sigma_bm3d = 20
        threshold_factor = 2.0

        # Apply the denoising pipeline
        denoised_image = denoise_pipeline(noisy_image, sigma_nlm, patch_size_nlm, search_window_nlm, sigma_bm3d, threshold_factor)

        # Calculate PSNR, MSE, and SNR
        psnr_value = peak_signal_noise_ratio(original_image.astype(np.float64), denoised_image.astype(np.float64), data_range=255)
        mse_value = mean_squared_error(original_image.astype(np.float64), denoised_image.astype(np.float64))
        snr_value = calculate_snr(original_image, denoised_image)

        # Display results
        cv2_imshow(original_image)
        cv2_imshow(noisy_image)
        cv2_imshow(denoised_image)
        print(f'PSNR: {psnr_value:.2f} dB')
        print(f'MSE: {mse_value:.2f}')
        print(f'SNR: {snr_value:.2f} dB')
