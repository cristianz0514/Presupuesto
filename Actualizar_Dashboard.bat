@echo off
title Sistema de Gestion Presupuestal - Fortia Minerals
mode con: cols=80 lines=20
color 0B

echo ======================================================================
echo           SISTEMA DE ACTUALIZACION DE DATOS PRESUPUESTALES
echo ======================================================================
echo.
echo  [1/2] Procesando informacion desde Excel...
echo.

python update_dashboard.py

if %errorlevel% neq 0 (
    echo.
    echo  [ERROR] No se pudo actualizar la informacion.
    echo  Consulte 'process_logs.log' para identificar el problema.
    echo.
    pause
    exit /b
)

echo.
echo  [2/2] Sincronizacion completada con exito.
echo.
echo  Abriendo Dashboard en el navegador...
echo.

:: Abre el archivo HTML en el navegador predeterminado
start "" "presupuesto_ejecucion.html"

echo  ======================================================================
echo     PROCESO FINALIZADO. PUEDE CERRAR ESTA VENTANA.
echo  ======================================================================
timeout /t 5
