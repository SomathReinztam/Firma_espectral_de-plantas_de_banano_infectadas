"""
2_conv1d.py

Longitud de salida de la señal despues de la conv
"""

def L_out(L_in, padding, stride, kernel_size):
    d = L_in + 2*padding - (kernel_size - 1) - 1
    x = d/stride
    x = x + 1
    return int(x)

