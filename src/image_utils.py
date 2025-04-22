"""
Utilidades para el procesamiento de imágenes.
"""
import os
from pathlib import Path

def check_image_size(image_path, max_size_mb=4.5):
    """
    Verifica si el tamaño de una imagen está dentro del límite permitido.
    
    Args:
        image_path (str): Ruta a la imagen.
        max_size_mb (float): Tamaño máximo permitido en MB.
        
    Returns:
        tuple: (is_valid_size, file_size_mb)
    """
    if not os.path.exists(image_path):
        return (False, 0)
    
    file_size_bytes = os.path.getsize(image_path)
    file_size_mb = file_size_bytes / (1024 * 1024)
    
    return (file_size_mb <= max_size_mb, file_size_mb)

def get_image_format(image_path):
    """
    Determina el formato de una imagen a partir de su extensión.
    
    Args:
        image_path (str): Ruta a la imagen.
        
    Returns:
        str: Formato de la imagen (jpeg, png, etc.)
    """
    extension = Path(image_path).suffix.lower()
    
    if extension in ['.jpg', '.jpeg']:
        return 'jpeg'
    elif extension == '.png':
        return 'png'
    elif extension == '.gif':
        return 'gif'
    elif extension == '.webp':
        return 'webp'
    else:
        return 'jpeg'  # Formato por defecto

def is_valid_image(image_path, max_size_mb=4.5):
    """
    Verifica si una imagen existe y está dentro del tamaño permitido.
    
    Args:
        image_path (str): Ruta a la imagen.
        max_size_mb (float): Tamaño máximo permitido en MB.
        
    Returns:
        tuple: (is_valid, reason)
    """
    if not os.path.exists(image_path):
        return (False, f"La imagen no existe: {image_path}")
    
    valid_size, file_size_mb = check_image_size(image_path, max_size_mb)
    if not valid_size:
        return (False, f"La imagen supera el tamaño máximo permitido: {file_size_mb:.2f}MB > {max_size_mb}MB")
    
    return (True, "Imagen válida")

# Si el archivo se ejecuta directamente, realizar algunas pruebas
if __name__ == "__main__":
    # Ejemplo de uso
    test_image = "../imagenes/AnatomiaTopografica/Abdomen/Fig5-4-abd.jpg"
    
    valid, reason = is_valid_image(test_image)
    if valid:
        print(f"La imagen es válida.")
        _, size_mb = check_image_size(test_image)
        print(f"Tamaño: {size_mb:.2f}MB")
        print(f"Formato: {get_image_format(test_image)}")
    else:
        print(f"La imagen no es válida: {reason}")