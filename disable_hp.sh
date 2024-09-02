#!/bin/bash

sudo modprobe msr;

sudo wrmsr -a 0x1a4 0xf

sudo rdmsr 0x1a4
