import torch
import numpy as np

def cepstrum(palpation):
    muestreo = 16000    
    mean = np.mean(palpation)
    standard_deviation = np.std(palpation)
    distance_from_mean = np.abs(palpation - mean)
    max_deviations = 0.76
    not_outlier = distance_from_mean < max_deviations * standard_deviation
    no_outliers = palpation[not_outlier]



    shift = np.mean(no_outliers)
    parte_shift = no_outliers-shift

    mult = 100000
    parte_mult = mult*parte_shift
    n_fft = 2**(14)
    tensor = torch.tensor(parte_mult) 
    tensor.to(device="cuda")
    fft_torch=torch.fft.fft(tensor,n_fft)
    m_fft_torch= np.abs(fft_torch)
    eps = 10**(-3)
    log_m_fft_torch = np.log(m_fft_torch+eps)
    ifft_t= torch.tensor(log_m_fft_torch)
    ifft_torch = torch.fft.ifft(ifft_t,n_fft)*muestreo
    ceps= np.abs(ifft_torch)

    return ceps