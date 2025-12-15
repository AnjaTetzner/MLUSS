import os, re, shutil

SRC_DIR = 'Rohdaten/'
TGT_DIR = 'renamed'

# Verzeichnisse wurden umbenannt, XLS-Dateien in zugversuch.csv konvertiert

INCLUDES = ['ok', 'near_ok_sonowechsel', 
            'fehler_leitungsversatz', 'fehler_oel', 'fehler_terminversatz',
            'nolabel_ok_leoni','gg_group']

def handle_files(src_path, tgt_path):
    re_simple = re.compile(r'[0-9]+.csv')
    re_dat = re.compile(r'Dat[0-9]+_[0-9]+_([0-9]+)_[0-9].csv')
    re_leoni = re.compile(r'Dat[0-9]+_[0-9]+_([0-9]+)_[0-9]+\.[0-9]+.csv')
    re_gg = re.compile(r'CycleLab_[0-9]+_[0-9]+_([0-9]+)-1[^.]+.csv')
    for fname in os.listdir(src_path):
        #print(fname)
        new_name = fname
        if re_simple.match(fname):
            # copy file
            pass
        else:
            m = re_dat.match(fname)
            m_leoni = re_leoni.match(fname)
            m_gg= re_gg.match(fname)
            if m:
                new_name = m.group(1)+'.csv'
                #print(fname, new_name)
            elif m_leoni:
                new_name = m_leoni.group(1)+'.csv'
            elif m_gg:
                new_name = m_gg.group(1)+'.csv'
            else:
                print(f"Wrong pattern {fname}")
                continue
        shutil.copy(os.path.join(src_path, fname), os.path.join(tgt_path, new_name))

def run():
    for fname in INCLUDES:
        path = os.path.join(SRC_DIR, fname)
        print(path)
        tgt_dir = os.path.join(TGT_DIR, fname)
        os.makedirs(tgt_dir, exist_ok=True)
        if os.path.exists(os.path.join(path, 'zugversuch.csv')):
            shutil.copy(os.path.join(path, 'zugversuch.csv'),
                os.path.join(tgt_dir, 'zugversuch.csv'),
            )
        src_path = os.path.join(path, 'schweisskurven')
        tgt_path = os.path.join(TGT_DIR, fname, 'schweisskurven')
        os.makedirs(tgt_path, exist_ok=True)
        handle_files(src_path, tgt_path)

if __name__ == '__main__':
    run()
