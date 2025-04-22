#!/usr/bin/env python
"""
Script principal para ejecutar el experimento de consejo de modelos.
"""
import os
import sys
import argparse
import time
from datetime import datetime
from pathlib import Path

# Asegurar que el directorio src esté en la ruta de importación
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.orchestrator import ModelEnsemble

def parse_arguments():
    """
    Procesa los argumentos de línea de comandos.
    
    Returns:
        Namespace: Argumentos procesados.
    """
    parser = argparse.ArgumentParser(
        description="Sistema de consejo de modelos para análisis de imágenes médicas"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Ejecutar en modo prueba, mostrando información detallada"
    )
    
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Número de muestras a procesar (para testing)"
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Directorio de logs desde donde continuar, o ID de pregunta"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Ruta al archivo de configuración"
    )
    
    parser.add_argument(
        "--advisors",
        nargs="+",
        default=None,
        help="Lista de modelos consejeros a utilizar (sobreescribe la configuración)"
    )
    
    parser.add_argument(
        "--decision-model",
        type=str,
        default=None,
        help="Modelo decisor a utilizar (sobreescribe la configuración)"
    )
    
    return parser.parse_args()

def main():
    """
    Función principal.
    """
    # Obtener argumentos
    args = parse_arguments()
    
    # Validar y ajustar argumentos
    resume_from = None
    if args.resume:
        # Verificar si es un directorio o un ID
        if os.path.isdir(args.resume):
            print(f"Continuando desde el directorio: {args.resume}")
            # TODO: Implementar continuación desde directorio
        else:
            try:
                resume_from = int(args.resume)
                print(f"Continuando desde la pregunta ID: {resume_from}")
            except ValueError:
                print(f"Error: '{args.resume}' no es un ID de pregunta válido")
                return
    
    # Iniciar el experimento
    print("=" * 80)
    print(f"Iniciando experimento: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configuración: {args.config}")
    if args.test:
        print("Modo: PRUEBA")
    else:
        print("Modo: PRODUCCIÓN")
    if args.sample:
        print(f"Procesando {args.sample} muestras")
    print("=" * 80)
    
    # Crear y configurar el orquestador
    try:
        start_time = time.time()
        ensemble = ModelEnsemble(config_path=args.config)
        
        # Ejecutar el experimento
        results_dir = ensemble.run(
            test_mode=args.test,
            sample_size=args.sample,
            resume_from=resume_from
        )
        
        # Mostrar resultados
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        
        print("\n" + "=" * 80)
        print(f"Experimento completado en {minutes}m {seconds}s")
        print(f"Resultados guardados en: {results_dir}")
        print("=" * 80)
        
        # Abrir el directorio de resultados si estamos en un entorno gráfico
        try:
            # Solo en sistemas compatibles
            if sys.platform == 'darwin':  # macOS
                os.system(f"open {results_dir}")
            elif sys.platform == 'win32':  # Windows
                os.system(f"explorer {results_dir}")
            elif sys.platform.startswith('linux'):  # Linux
                os.system(f"xdg-open {results_dir}")
        except:
            pass
        
    except Exception as e:
        print(f"Error durante la ejecución: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())