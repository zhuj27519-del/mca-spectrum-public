import csv
import numpy as np

def csv_read(b):
    lines=[s for s in b.decode('utf-8-sig').splitlines() if s.strip()]
    comments=[s for s in lines if s.startswith('#')]
    rows=list(csv.reader(s for s in lines if not s.startswith('#')))
    return comments,rows[0],np.array(rows[1:],float)
