#!/usr/bin/env python
# encoding: utf-8

from menpolmioserver.lmioapi import app_for_adapter
from menpolmioserver.menpoadapter import MenpoAdapter
import webbrowser

if __name__ == '__main__':
    gzip = False  # halves payload, increases server workload
    dev = False

    model_dir = '/Users/jab08/landmarkerdata/models'
    landmark_dir = '/Users/jab08/landmarkerdata/landmarks'
    template_dir = '/Users/jab08/landmarkerdata/templates'

    # Build a MenpoAdapter that will serve from the specified directories
    adapter = MenpoAdapter(model_dir, landmark_dir, template_dir)

    webbrowser.open("http://www.landmarker.io")
    app = app_for_adapter(adapter, gzip=gzip, dev=dev)
    app.run()
