import pickle
import os
import datetime as dt
import shutil
import time
import json

class ManualBackup:
    def __init__(self, target_backup_dir='/content/gdrive/My Drive/manual_backup', backup_title='backup_training_output', initiate_backup_dir=False):
        self.target_backup_dir = target_backup_dir
        self.backup_title = backup_title
        self.info_name = 'backup_info.json'
        self.backup_index = 0

        self.backup_names = {
            'dirs': 'directories_backups',
            'files': 'files_backups',
            'objs': 'objs_backups'
        }

        # check if the target is accessible
        test_file = os.path.join(target_backup_dir, 'text.txt')
        if not os.path.isdir(target_backup_dir):
            os.makedirs(target_backup_dir)
        with open(test_file, 'w+') as tbp:
            tbp.write('.')
        os.remove(test_file)

        print(f'Manual Backup Driver Initiated: {str(self)}')

        if initiate_backup_dir:
            self.initiate_backup_dir(target_backup_dir, backup_title)
        
    def initiate_backup_dir(self, target_backup_dir, backup_title):
        self.backup_dir_path = os.path.join(target_backup_dir, f'{backup_title}_{self.backup_index}|{str(dt.datetime.now()).replace(" ", "-").replace(":", "_")}')
        if not os.path.isdir(self.backup_dir_path):
            os.makedirs(self.backup_dir_path)  
            self.backup_index += 1
        
        for _,value in self.backup_names.items():
            dst_backup_dir = os.path.join(self.backup_dir_path, value)
            if not os.path.isdir(dst_backup_dir):
                os.mkdir(dst_backup_dir)

    def copy_file(self, src, dst):
        if not os.path.isdir(dst):
            os.makedirs(dst)
        shutil.copy2(src, dst)  # Preserves metadata        

    def copy_dir(self, src, dst):
        for root, _, files in os.walk(src):
            for filename in files:
                src_file = os.path.join(root, filename)
                dst_file = os.path.join(dst, os.path.relpath(src_file, src))
                print('copy', dst_file)
                if not os.path.isdir(os.path.dirname(dst_file)):
                    os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                shutil.copy2(src_file, dst_file)  # Preserves metadata

    def backup_file(self, src):
        if not os.path.isfile(src):
            print(f"Skipping file '{src}': does not exist")
            return False

        backup_file_name = os.path.basename(src)
        backup_file_dir = os.path.join(self.backup_dir_path, self.backup_names['files'])
        backup_file = os.path.join(backup_file_dir, backup_file_name)
        print('copy dst', backup_file)
        with open(src, 'rb') as src_f:
          with open(backup_file, 'wb+') as dst_f:
            dst_f.write(src_f.read())
        return backup_file
    
    def backup_dir(self, src):
        if not os.path.isdir(src):
            print(f"Skipping directory '{src}': does not exist")
            return False

        backup_dir_name = os.path.basename(src)
        backup_dir = os.path.join(self.backup_dir_path, self.backup_names['dirs'], backup_dir_name)
        if not os.path.isdir(backup_dir):
            os.makedirs(backup_dir)

        self.copy_dir(src, backup_dir)
        return backup_dir
    

    def get_backups_sorted(self):
        date_backupdir = {
            dt.datetime.strptime(backup_dir.split('|')[-1], '%Y-%m-%d-%H_%M_%S.%f'): 
                backup_dir 
            for backup_dir in os.listdir(self.target_backup_dir)}
        date_backupdir = [date_backupdir[key] for key in sorted(date_backupdir)]
        return date_backupdir
    
    def backup(self, dirs_backup:list=[], files_backup:list=[], other_objects:dict=dict(), max_backup_count=3, initiate_backup_dir=True):
        if initiate_backup_dir:
            self.initiate_backup_dir(self.target_backup_dir, self.backup_title)

        if max_backup_count>0:
            date_backupdir = self.get_backups_sorted()
            if len(date_backupdir)>max_backup_count:
                to_delete_backupdirs = date_backupdir[:len(date_backupdir)-max_backup_count]
                print('Old backups will be removed:', to_delete_backupdirs)
                for td_backupdir in to_delete_backupdirs:
                    to_delete = os.path.join(self.target_backup_dir, td_backupdir)
                    shutil.rmtree(to_delete)

        dir_path = [str(dir_path)[:-1] if str(dir_path).endswith(os.path.sep) else str(dir_path) for dir_path in dirs_backup]

        print(f'({time.ctime()}) Backup of {dirs_backup}, {files_backup}, {[str(i) for i in other_objects]}')

        backup_mapping = {
            'dirs':dict(),
            'files':dict(),
            'objects':dict()
        }


        failed_backups = {
            'dirs':[],
            'files':[],
            'objects':[]
        }

        # Backup directories
        for dir_path in dirs_backup:
            backup_dir_result = self.backup_dir(dir_path)
            if not backup_dir_result:
                failed_backups['dirs'].append(str(dir_path))
            else:
                backup_mapping['dirs'][str(dir_path)] = str(backup_dir_result)

        # Backup files
        for file_path in files_backup:
            file_path = str(file_path)
            backup_file_result = self.backup_file(file_path)
            if not backup_file_result:
                failed_backups['files'].append(str(file_path))
            else:
                backup_mapping['files'][str(file_path)] = str(backup_file_result)

        # Backup other objects using pickle
        for obj_key, obj in other_objects.items():
            obj_name = os.path.basename(str(obj)[:25])
            backup_file_name = f"{obj_name}.pickle"
            backup_file_dir = os.path.join(self.backup_dir_path, self.backup_names['objs'])
            if not os.path.isdir(backup_file_dir):
                os.makedirs(backup_file_dir)
            backup_file = os.path.join(backup_file_dir, backup_file_name)
            with open(backup_file, 'wb') as f:
                pickle.dump(obj, f)
            if not os.path.isfile(backup_file):
                failed_backups['objects'].append(obj_name)
            else:
                backup_mapping['objects'][obj_key] = backup_file
        
        backup_info = {
            'datetime': str(dt.datetime.now()),
            'timestamp': time.time(),
            'backup_mapping': backup_mapping,
            'failed_backups': failed_backups,
        }

        with open(os.path.join(self.backup_dir_path, self.info_name), 'w+') as bu_file:
            bu_file.write(
                json.dumps(backup_info)
        )

        print(f"Backup completed to: {self.backup_dir_path}")

    def restore(self, src=None, dst=None):
        if src is None:
            date_backupdir = self.get_backups_sorted()
            src = os.path.join(self.target_backup_dir, date_backupdir[-1])

        if dst is None:
          dst = os.getcwd()

        if not os.path.isfile(os.path.join(src, self.info_name)):
            print(f'Backup of {src} is broken, {self.info_name} not found')
            return False
        
        info = None
        with open(os.path.join(src, self.info_name), 'r') as info_file:
            info = json.loads(info_file.read())

        restored = {
            'dirs':dict(),
            'files':dict(),
            'objects':dict()
        }

        # restore dirs
        for dst, dir_torestore in info['backup_mapping']['dirs'].items():
            self.copy_dir(dir_torestore, dst)
            restored['dirs'][dir_torestore] = dst
        
        # restore files
        for dst, file_torestore in info['backup_mapping']['files'].items():
            dst = os.path.split(dst)[0] 
            if dst=='':
              dst = os.getcwd()
            self.copy_file(file_torestore, dst)
            restored['files'][file_torestore] = dst
        
        # restore objects
        # for obj_key, obj_path in info['backup_mapping']['files'].items():
        #     with open(obj_path, 'rb') as obj_file:
        #         unpickler = pickle.Unpickler(obj_file)              
        #         restored['objects'][obj_key] = unpickler.load()
        
        return restored

    def __str__(self):
        return f'ManualBackup(target_backup_dir="{self.target_backup_dir}", backup_title="{self.backup_title}")'