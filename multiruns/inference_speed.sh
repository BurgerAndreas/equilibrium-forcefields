#!/bin/bash

launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-6
launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-5
launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-4
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-3
launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-2
launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-1
launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e0
launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e1
launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e2

launchrun evaluate=True model.num_layers=1
launchrun evaluate=True model.num_layers=4
launchrun evaluate=True model.num_layers=8