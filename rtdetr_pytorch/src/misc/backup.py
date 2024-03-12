import pickle
import os
import datetime
import shutil
import time

import traceback

class ManualBackup:
    def __init__(self, drive_dir='/content/gdrive', backup_dir=f'training_output_{str(datetime.datetime.now()).replace(" ", "|").replace(":", "_")}'):
        self.mydrive_dir = os.path.join(drive_dir, 'My Drive')
        self.backup_dir = os.path.join(self.mydrive_dir, backup_dir)
        if not os.path.isdir(self.backup_dir):
            os.mkdir(self.backup_dir)
        
        self.backup_names = {
            'dirs': 'directories_backups',
            'files': 'files_backups',
            'objs': 'objs_backups'
        }

        for _,value in self.backup_names.items():
            dst_backup_dir = os.path.join(self.backup_dir, value)
            if not os.path.isdir(dst_backup_dir):
                os.mkdir(dst_backup_dir)
        

    def backup(self, dirs_backup=[], files_backup=[], other_objects=[]):
        """
        Backups directories, files, and other objects to Google Drive.

        Args:
            dirs_backup (list): A list of directory paths to backup.
            files_backup (list): A list of file paths to backup.
            other_objects (list): A list of other objects (serializable with pickle) to backup.
        """
        print(f'({time.ctime()}) Backup of {dirs_backup}, {files_backup}, {other_objects}')
        # Backup directories
        for dir_path in dirs_backup:
            if not os.path.isdir(dir_path):
                print(f"Skipping directory '{dir_path}': does not exist")
                continue

            backup_dir_name = os.path.basename(dir_path)
            backup_dir = os.path.join(self.backup_dir, self.backup_names['dirs'], backup_dir_name)
            if not os.path.isdir(backup_dir):
                os.makedirs(backup_dir)

            for root, _, files in os.walk(dir_path):
                for filename in files:
                    src_file = os.path.join(root, filename)
                    dst_file = os.path.join(backup_dir, os.path.relpath(src_file, dir_path))
                    print('copy', dst_file)
                    if not os.path.isdir(os.path.dirname(dst_file)):
                        os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                    shutil.copy2(src_file, dst_file)  # Preserves metadata

        # Backup files
        for file_path in files_backup:
            if not os.path.isfile(file_path):
                print(f"Skipping file '{file_path}': does not exist")
                continue

            backup_file_name = os.path.basename(file_path)
            backup_file_dir = os.path.join(self.backup_dir, self.backup_names['files'])
            backup_file = os.path.join(backup_file_dir, backup_file_name)
            if not os.path.isdir(backup_file_dir):
                os.makedirs(backup_file_dir)
            shutil.copy2(file_path, backup_file)  # Preserves metadata

        # Backup other objects using pickle
        for obj in other_objects:
            backup_file_name = f"{os.path.basename(str(obj)[:25])}.pkl"
            backup_file_dir = os.path.join(self.backup_dir, self.backup_names['objs'])
            if not os.path.isdir(backup_file_dir):
                os.makedirs(backup_file_dir)
            backup_file = os.path.join(backup_file_dir, backup_file_name)
            with open(backup_file, 'wb') as f:
                pickle.dump(obj, f)

        print(f"Backup completed to: {self.backup_dir}")
