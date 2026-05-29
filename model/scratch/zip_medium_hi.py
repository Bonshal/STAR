import zipfile
import os

def zip_project():
    zip_path = 'C:\\Users\\bonsh\\Desktop\\Projects\\FCCI\\Multi_GNN_MediumHI.zip'
    base_dir = 'C:\\Users\\bonsh\\Desktop\\Projects\\FCCI\\Multi-GNN'
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(base_dir):
            # Skip .git, .venv, logs, and huge __pycache__ folders
            if '.git' in root or '.venv' in root or 'logs' in root or '__pycache__' in root:
                continue
            
            for file in files:
                file_path = os.path.join(root, file)
                
                # We only want to include specific large datasets to avoid a 35GB zip
                if 'AML_dataset' in root:
                    if file != 'HI-Medium_Trans.csv':
                        continue # Skip all other heavy datasets
                
                # Also skip other large models except the one we just saved
                if file.endswith('.tar') and file != 'GIN_Small_LI_ROC0.958_Recall26pct.tar':
                    continue
                
                arcname = os.path.relpath(file_path, base_dir)
                zipf.write(file_path, arcname)
                print(f"Added {arcname}")
                
if __name__ == '__main__':
    zip_project()
    print("Zipping complete!")
