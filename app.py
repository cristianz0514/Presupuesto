from flask import Flask, send_from_directory
import os
import subprocess

app = Flask(__name__)

# Servir el dashboard principal
@app.route('/')
def index():
    # Intentar actualizar el dashboard antes de mostrarlo (opcional)
    # subprocess.run(["python", "update_dashboard.py"])
    return send_from_directory('.', 'presupuesto_ejecucion.html')

# Ruta para descargar la homologación si es necesario
@app.route('/homologations.json')
def get_homo():
    return send_from_directory('.', 'homologations.json')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
