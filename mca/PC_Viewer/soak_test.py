"""Bounded PC-only real TCP/Qt soak. Does not claim target Ethernet validation."""
import argparse
import json
import os
import time
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from PySide6.QtWidgets import QApplication
from viewer import Viewer
from mock_server import MockServer

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seconds',type=float,default=60)
    parser.add_argument('--output',default='soak-result.json')
    args=parser.parse_args()
    app=QApplication([])
    server=MockServer(port=0,interval=.1,disconnect_every=100).start()
    window=Viewer(False,'127.0.0.1',server.port)
    start=time.monotonic()
    try:
        while time.monotonic()-start<args.seconds:
            app.processEvents(); time.sleep(.002)
        result={'scope':'PC loopback TCP + Qt GUI, not STM32 hardware',
                'seconds':time.monotonic()-start,'frames':window.receiver.received,
                'connections':window.receiver.connections,'server_connections':server.connections,
                'crc_errors':window.receiver.parser.crc_errors,'header_errors':window.receiver.parser.header_errors,
                'display_channels':len(window.curve.getData()[0]) if window.current else 0,
                'peak_rows':window.table.rowCount(),'parser_pending_bytes':len(window.receiver.parser.buffer)}
        result['pass']=result['frames']>=args.seconds*8 and result['connections']>=2 and result['crc_errors']==0 and result['header_errors']==0 and result['display_channels']==2048 and result['peak_rows']==3
        with open(args.output,'w') as f: json.dump(result,f,indent=2)
        print(json.dumps(result,indent=2),flush=True)
        return 0 if result['pass'] else 1
    finally:
        window.close(); server.close()

if __name__=='__main__':
    raise SystemExit(main())
