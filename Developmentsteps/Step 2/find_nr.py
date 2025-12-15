import os, re, sys
import pandas as pd

def usage():
    print('Usage {} measure_file.csv curve_dir'.format(sys.argv[0]))
    
def run():
    if len(sys.argv) != 3:
        usage()
        return 1
    _, measure_file, curve_dir = sys.argv
    numbers = read_measure_numbers(measure_file)
    print('Measure numbers:', numbers)
    pos, len_nr = get_curve_nr_pos(numbers, curve_dir)
    print('Position and length of number in curve file', pos, len_nr)
    # check if all curves have measurment
    check_measurements_complete(numbers, curve_dir, pos, len_nr)

def read_measure_numbers(measure_file):
    '''
        Liest Spalte 'nr' aus der CSV-Datei der Messungen
        (Spaltentrenner: ';') und speichert sie in einem Set.
    '''
    measures = pd.read_csv(measure_file, sep=';')
    res = set()
    for nr in measures.nr:
        res.add(nr)
    return res

def get_curve_nr_pos(numbers, curve_dir):
    '''
        Ermittelt die Position der Nummer im Namen der
        Schweißkurvendateien. Es werden alle Zifferngruppen
        analysiert. Wenn diese (als Zahl interpretiert) mit
        einer Messungsnummer übereinstimmen, werden sie
        gespeichert (Startposition und Länge).
        Die am häufigsten vorkommende Startposition sollte
        die Zifferngruppe mit der Kurvennummer sein.
    '''
    re_nr = re.compile('[0-9]+')
    starts = dict()
    for f in os.listdir(curve_dir):
        #print(f)
        for m in re_nr.finditer(f):
            #print(m, m.group(0), m.start(0))
            # Gruppe 0 (komplatter Match) auswählen: value und pos
            nr = int(m.group(0))
            l_nr = len(m.group(0))
            pos = m.start(0)
            if nr not in numbers: continue
            if pos not in starts:
                starts[pos] = list()
            starts[pos].append((nr, l_nr))
    max_pos = None
    max_len = -1
    for pos in starts:
        if len(starts[pos]) > max_len:
            max_len = len(starts[pos])
            max_pos = pos
    # 2nd return value is length info of first entry
    return max_pos, starts[max_pos][0][1]

def check_measurements_complete(numbers, curve_dir, pos, nr_len):
    '''
        Vollständigkeitsanalyse:
        - Gibt es für jede Messung eine Kurve?
        - Gibt es für jede Kurve einen Messwert?
    '''
    pos_1 = pos + nr_len
    curve_nrs = dict()
    for f in os.listdir(curve_dir):
        #print(f)
        nr = int(f[pos:pos_1])
        curve_nrs[nr] = f
        if nr not in numbers:
            print('Curve {} (Nr. {}) has no measurement'.format(f, nr))
    for nr in numbers:
        if nr not in curve_nrs:
            print('Measurement {} has no curve'.format(nr))
        
if __name__ == '__main__':
    run()
