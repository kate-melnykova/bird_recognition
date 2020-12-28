# pip install -q pyyaml h5py

from backend.api import views

if __name__ == '__main__':
    views.app.run(debug=True)