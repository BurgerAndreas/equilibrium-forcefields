#!/bin/bash

# training
launchrun torch_record_memory=True +use=deq +cfg=fpc_of epochs=1
launchrun torch_record_memory=True epochs=1 model.num_layers=1
launchrun torch_record_memory=True epochs=1 model.num_layers=4
launchrun torch_record_memory=True epochs=1 model.num_layers=8

launchrun torch_record_memory=True +use=deq +cfg=fpc_of evaluate=True
launchrun torch_record_memory=True evaluate=True model.num_layers=1
launchrun torch_record_memory=True evaluate=True model.num_layers=4
launchrun torch_record_memory=True evaluate=True model.num_layers=8

