"""Topology study add-on: 5 and 10 online iterations per word (25 offline minibatches)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vnet_study as vs
vs.log('===== extra: 5 and 10 online iterations (mb=25) =====')
vs.run_jobs([('affine', 25, 5), ('affine', 25, 10), ('100-58', 25, 5), ('100-58', 25, 10)])
vs.log('extra complete')
