import zipfile
import os

def zip_code():
    zip_path = 'C:\\Users\\bonsh\\Desktop\\Projects\\FCCI\\Multi_GNN_Code_Latest.zip'
    base_dir = 'C:\\Users\\bonsh\\Desktop\\Projects\\FCCI\\Multi-GNN'
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(base_dir):
            if '.git' in root or '.venv' in root or 'logs' in root or '__pycache__' in root or 'data' in root or 'AML_dataset' in root or 'scratch' in root:
                continue
            
            for file in files:
                if file.endswith('.zip'):
                    continue
                    
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, base_dir)
                zipf.write(file_path, arcname)
                print(f"Added {arcname}")
                
if __name__ == '__main__':
    zip_code()
